import numpy as np
import torch
import torch.nn as nn
import gym
from copy import deepcopy
from typing import Dict, Union, Tuple
import torch.nn.functional as F

from policy_learning.base_policy import BasePolicy


class IPLIQLPolicy(BasePolicy):
    """
    Implicit Q-Learning <Ref: https://arxiv.org/abs/2110.06169>
    """

    def __init__(
        self,
        actor: nn.Module,
        critic_q1: nn.Module,
        critic_q2: nn.Module,
        critic_v: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic_q1_optim: torch.optim.Optimizer,
        critic_q2_optim: torch.optim.Optimizer,
        critic_v_optim: torch.optim.Optimizer,
        action_space: gym.spaces.Space,
        tau: float = 0.005,
        gamma: float = 0.99,
        expectile: float = 0.8,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()

        self.actor = actor
        self.critic_q1, self.critic_q1_old = critic_q1, deepcopy(critic_q1)
        self.critic_q1_old.eval()
        self.critic_q2, self.critic_q2_old = critic_q2, deepcopy(critic_q2)
        self.critic_q2_old.eval()
        self.critic_v = critic_v

        self.actor_optim = actor_optim
        self.critic_q1_optim = critic_q1_optim
        self.critic_q2_optim = critic_q2_optim
        self.critic_v_optim = critic_v_optim

        self.action_space = action_space
        self._tau = tau
        self._gamma = gamma
        self._expectile = expectile
        self._temperature = temperature

    def train(self) -> None:
        self.actor.train()
        self.critic_q1.train()
        self.critic_q2.train()
        self.critic_v.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critic_q1.eval()
        self.critic_q2.eval()
        self.critic_v.eval()

    def _sync_weight(self) -> None:
        for o, n in zip(self.critic_q1_old.parameters(), self.critic_q1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic_q2_old.parameters(), self.critic_q2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if len(obs.shape) == 1:
            obs = obs.reshape(1, -1)
        with torch.no_grad():
            dist = self.actor(obs)
            if deterministic:
                action = dist.mode().cpu().numpy()
            else:
                action = dist.sample().cpu().numpy()
        action = np.clip(action, self.action_space.low[0], self.action_space.high[0])
        return action

    def _expectile_regression(self, diff: torch.Tensor) -> torch.Tensor:
        weight = torch.where(diff > 0, self._expectile, (1 - self._expectile))
        return weight * (diff**2)

    def learn(self, batch: Dict, preference_batch: Dict) -> Dict[str, float]:
        obss, actions, next_obss, terminals = (
            batch["observations"],
            batch["actions"],
            batch["next_observations"],
            batch["terminals"],
        )

        # Compute target Qs with old critics
        with torch.no_grad():
            q1_old = self.critic_q1_old(obss, actions)
            q2_old = self.critic_q2_old(obss, actions)
            qs = torch.min(q1_old, q2_old)
            next_v = self.critic_v(next_obss)
            if True:  # target_clipping enabled
                q_lim = 1.0 / (0.5 * (1 - self._gamma))  # chi2_coeff=0.5
                next_v = torch.clamp(next_v, min=-q_lim, max=q_lim)

        # Compute value loss using expectile regression with preference and offline data distinction
        v = self.critic_v(obss)
        s0_obs, s0_act, s1_obs, s1_act, mu, _, _ = [
            x.to(self.actor.device) for x in preference_batch
        ]
        B, T = s0_obs.shape[:2]
        # Offline value loss over all obss
        v_loss_offline = self._expectile_regression(qs - v).mean()

        # Preference-based value loss
        s0_v = self.critic_v(s0_obs.view(B * T, -1))
        s1_v = self.critic_v(s1_obs.view(B * T, -1))
        s0_q1 = self.critic_q1_old(s0_obs.view(B * T, -1), s0_act.view(B * T, -1))
        s0_q2 = self.critic_q2_old(s0_obs.view(B * T, -1), s0_act.view(B * T, -1))
        s0_q = torch.min(s0_q1, s0_q2)
        s1_q1 = self.critic_q1_old(s1_obs.view(B * T, -1), s1_act.view(B * T, -1))
        s1_q2 = self.critic_q2_old(s1_obs.view(B * T, -1), s1_act.view(B * T, -1))
        s1_q = torch.min(s1_q1, s1_q2)

        v_loss_fb = self._expectile_regression(s0_q - s0_v).mean() + self._expectile_regression(s1_q - s1_v).mean()
        v_loss = 0.5 * v_loss_fb + 0.5 * v_loss_offline

        self.critic_v_optim.zero_grad()
        v_loss.backward()
        self.critic_v_optim.step()

        s0_obs, s0_act, s1_obs, s1_act, mu, _, _ = [
            x.to(self.actor.device) for x in preference_batch
        ]
        B, T = s0_obs.shape[:2]

        # Ensemble-aware q_loss: compute for both critics then average
        r1_q1 = self.critic_q1(s0_obs.view(B * T, -1), s0_act.view(B * T, -1)).view(B, T)
        r2_q1 = self.critic_q1(s1_obs.view(B * T, -1), s1_act.view(B * T, -1)).view(B, T)
        logits_q1 = r2_q1.sum(dim=1) - r1_q1.sum(dim=1)
        loss_q1 = F.binary_cross_entropy_with_logits(logits_q1, mu)

        r1_q2 = self.critic_q2(s0_obs.view(B * T, -1), s0_act.view(B * T, -1)).view(B, T)
        r2_q2 = self.critic_q2(s1_obs.view(B * T, -1), s1_act.view(B * T, -1)).view(B, T)
        logits_q2 = r2_q2.sum(dim=1) - r1_q2.sum(dim=1)
        loss_q2 = F.binary_cross_entropy_with_logits(logits_q2, mu)

        q_loss = 0.5 * (loss_q1 + loss_q2)

        # Compute chi2 loss using current critics
        chi2_coeff = 0.5
        q1 = self.critic_q1(obss, actions)
        q2 = self.critic_q2(obss, actions)
        q = torch.min(q1, q2)
        reward = q - self._gamma * next_v
        # Preference-based chi2 loss
        reward_pref = torch.cat([
            torch.min(
                self.critic_q1(s0_obs.view(B * T, -1), s0_act.view(B * T, -1)),
                self.critic_q2(s0_obs.view(B * T, -1), s0_act.view(B * T, -1))
            ) - self._gamma * self.critic_v(s0_obs.view(B * T, -1)),
            torch.min(
                self.critic_q1(s1_obs.view(B * T, -1), s1_act.view(B * T, -1)),
                self.critic_q2(s1_obs.view(B * T, -1), s1_act.view(B * T, -1))
            ) - self._gamma * self.critic_v(s1_obs.view(B * T, -1))
        ])
        chi2_loss_pref = (reward_pref**2).mean()
        chi2_loss_offline = (reward**2).mean()
        chi2_loss = chi2_coeff * (0.5 * chi2_loss_pref + 0.5 * chi2_loss_offline)

        critic_loss = q_loss + chi2_loss

        self.critic_q1_optim.zero_grad()
        self.critic_q2_optim.zero_grad()
        critic_loss.backward()
        self.critic_q1_optim.step()
        self.critic_q2_optim.step()

        # Update actor
        dist = self.actor(obss)
        log_probs = dist.log_prob(actions)
        with torch.no_grad():
            q1_old = self.critic_q1_old(obss, actions)
            q2_old = self.critic_q2_old(obss, actions)
            q = torch.min(q1_old, q2_old)
            v = self.critic_v(obss)
            exp_a = torch.exp((q - v) * self._temperature)
            exp_a = torch.clip(exp_a, None, 100.0)
        # Offline actor loss over all data
        actor_loss_offline = -(exp_a * log_probs).mean()

        # Preference-based actor loss
        s0_logp = self.actor(s0_obs.view(B * T, -1)).log_prob(s0_act.view(B * T, -1))
        s1_logp = self.actor(s1_obs.view(B * T, -1)).log_prob(s1_act.view(B * T, -1))
        with torch.no_grad():
            s0_q1 = self.critic_q1_old(s0_obs.view(B * T, -1), s0_act.view(B * T, -1))
            s0_q2 = self.critic_q2_old(s0_obs.view(B * T, -1), s0_act.view(B * T, -1))
            s0_q = torch.min(s0_q1, s0_q2)
            s0_v = self.critic_v(s0_obs.view(B * T, -1))
            s1_q1 = self.critic_q1_old(s1_obs.view(B * T, -1), s1_act.view(B * T, -1))
            s1_q2 = self.critic_q2_old(s1_obs.view(B * T, -1), s1_act.view(B * T, -1))
            s1_q = torch.min(s1_q1, s1_q2)
            s1_v = self.critic_v(s1_obs.view(B * T, -1))
            s0_adv = torch.exp((s0_q - s0_v) * self._temperature).clamp_max(100.0)
            s1_adv = torch.exp((s1_q - s1_v) * self._temperature).clamp_max(100.0)
        actor_loss_fb = -(s0_adv * s0_logp).mean() + -(s1_adv * s1_logp).mean()
        actor_loss = 0.5 * actor_loss_fb + 0.5 * actor_loss_offline

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self._sync_weight()

        return {
            "loss/actor": actor_loss.item(),
            "loss/q": critic_loss.item(),
            "loss/v": v_loss.item(),
        }
