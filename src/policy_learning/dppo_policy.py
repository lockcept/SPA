import numpy as np
import torch
import torch.nn as nn
import gym
from typing import Dict
import torch.nn.functional as F

from policy_learning.base_policy import BasePolicy
from policy_learning.pref_transformer import PrefTransformer

class PreferencePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, n_heads=4, n_layers=1, dropout=0.1):
        super().__init__()
        # use PrefTransformer encoder as in the attached PrefTransformer.py
        self.encoder = PrefTransformer(
            input_dim=input_dim,
            embed_dim=hidden_dim,
            num_heads=n_heads,
            num_layers=n_layers,
            dropout=dropout,
        )
        self.compare_layer = nn.Linear(hidden_dim * 2, 1)

    def encode_traj(self, traj):
        # traj shape: (B, T, obs+act)
        h = self.encoder(traj)   # PrefTransformer returns (B, T, hidden)
        return h.mean(dim=1)

    def forward(self, traj0, traj1):
        traj0_emb = self.encode_traj(traj0)
        traj1_emb = self.encode_traj(traj1)
        pair_emb = torch.cat([traj0_emb, traj1_emb], dim=-1)
        return self.compare_layer(pair_emb).squeeze(-1)


class DPPOPolicy(BasePolicy):
    """
    Direct Preference-based Policy Optimization without Reward Modeling
    """

    def __init__(
        self,
        actor: nn.Module,
        actor_optim: torch.optim.Optimizer,
        action_space: gym.spaces.Space,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.actor_optim = actor_optim
        self.action_space = action_space

    def train(self) -> None:
        self.actor.train()

    def eval(self) -> None:
        self.actor.eval()

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

    def learn_with_predictor(
        self, predictor: nn.Module, unlabeled_batch: Dict, lambda_reg: float = 0.5
    ) -> Dict[str, float]:
        # unlabeled_batch contains (traj0_obs, traj0_act, traj1_obs, traj1_act)
        s0_obs, s0_act, s1_obs, s1_act = [
            x.to(next(self.actor.parameters()).device) for x in unlabeled_batch
        ]
        # build traj for predictor
        traj0 = torch.cat([s0_obs, s0_act], dim=-1)  # (B, T, obs+act)
        traj1 = torch.cat([s1_obs, s1_act], dim=-1)
        with torch.no_grad():
            logits_pref = predictor(traj0, traj1)
            y_hat = (torch.sigmoid(logits_pref) > 0.5).float()

        # compute policy-segment L2 distance as in learner.py
        policy_act0 = self.actor(s0_obs).mode()
        policy_act1 = self.actor(s1_obs).mode()
        # normalize by action range
        action_range = (self.action_space.high[0] - self.action_space.low[0])
        step_diffs0 = (policy_act0 - s0_act) / action_range
        step_diffs1 = (policy_act1 - s1_act) / action_range
        d0 = torch.norm(step_diffs0, dim=2).mean(dim=1)  # average over T
        d1 = torch.norm(step_diffs1, dim=2).mean(dim=1)
        s01 = -F.softplus(d0 - lambda_reg * d1)
        s10 = -F.softplus(d1 - lambda_reg * d0)
        score = (1 - y_hat) * s01 + y_hat * s10
        policy_loss = score.mean()

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()
        return {"loss/policy": policy_loss.item()}


def train_preference_predictor(
    predictor,
    optimizer,
    pref_batch,
    buffer=None,
    smoothness_weight=1.0,
    seg_len=25,
    overlap_shift=5,
):
    s0_obs, s0_act, s1_obs, s1_act, mu, _, _ = [
        x.to(next(predictor.parameters()).device) for x in pref_batch
    ]
    traj0 = torch.cat([s0_obs, s0_act], dim=-1)
    traj1 = torch.cat([s1_obs, s1_act], dim=-1)
    label = mu
    logits = predictor(traj0, traj1)
    ce_loss = F.binary_cross_entropy_with_logits(logits, label)

    smoothness_loss = torch.tensor(0.0, device=logits.device)
    if buffer is not None:
        # sample nearly overlapping segments sigma, sigma'
        sigma, sigma_prime = buffer.sample_overlapping_segments(
            batch_size=label.shape[0], seg_len=seg_len, overlap_shift=overlap_shift
        )
        sigma_traj = torch.cat([sigma["observations"].to(label.device), sigma["actions"].to(label.device)], dim=-1)
        sigma_prime_traj = torch.cat([sigma_prime["observations"].to(label.device), sigma_prime["actions"].to(label.device)], dim=-1)
        logits_overlap = predictor(sigma_traj, sigma_prime_traj)
        probs_overlap = torch.sigmoid(logits_overlap)
        smoothness_loss = ((probs_overlap - 0.5) ** 2).mean()

    total_loss = ce_loss + smoothness_weight * smoothness_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    return {
        "loss/pref_ce": ce_loss.item(),
        "loss/smoothness": smoothness_loss.item(),
        "loss/total": total_loss.item(),
    }
