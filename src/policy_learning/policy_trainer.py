import time
import os
import csv

import numpy as np
import torch
import gym
import wandb

from typing import Optional, Dict, List
from tqdm import tqdm

from policy_learning.base_policy import BasePolicy
from policy_learning.logger import Logger
from policy_learning.replay_buffer import ReplayBuffer


SAVE_POINTS = [250000, 500000, 750000, 1000000]


class MFPolicyTrainer:
    def __init__(
        self,
        policy: BasePolicy,
        eval_env: gym.Env,
        buffer: ReplayBuffer,
        logger: Logger,
        epoch: int = 1000,
        step_per_epoch: int = 1000,
        batch_size: int = 256,
        eval_episodes: int = 10,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        self.policy = policy
        self.eval_env = eval_env
        self.buffer = buffer
        self.logger = logger

        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._batch_size = batch_size
        self._eval_episodes = eval_episodes
        self.lr_scheduler = lr_scheduler

        self.log_path = os.path.join(logger._dir, "train_log.csv")

    def train(self) -> Dict[str, float]:
        start_time = time.time()
        num_timesteps = 0
        best_norm_ep_rew_mean = -float("inf")

        with open(self.log_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Timesteps", "Reward", "Length", "Success"])

        for e in range(1, self._epoch + 1):
            wandb_log = {}
            self.policy.train()
            pbar = tqdm(range(self._step_per_epoch), desc=f"Epoch #{e}/{self._epoch}")

            for _ in pbar:
                batch = self.buffer.sample(self._batch_size)
                loss = self.policy.learn(batch)
                pbar.set_postfix(**loss)

                for k, v in loss.items():
                    self.logger.logkv_mean(k, v)

                wandb_log.update(loss)
                num_timesteps += 1

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            eval_info = self._evaluate()

            normalized_rewards = [
                self.eval_env.get_normalized_score(r)
                for r in eval_info["eval/episode_reward"]
            ]
            norm_ep_rew_mean, norm_ep_rew_std = np.mean(normalized_rewards), np.std(
                normalized_rewards
            )
            ep_length_mean = np.mean(eval_info["eval/episode_length"])
            ep_length_std = np.std(eval_info["eval/episode_length"])

            self.logger.logkv("eval/normalized_episode_reward", norm_ep_rew_mean)
            self.logger.logkv("eval/normalized_episode_reward_std", norm_ep_rew_std)
            self.logger.logkv("eval/episode_length", ep_length_mean)
            self.logger.logkv("eval/episode_length_std", ep_length_std)
            self.logger.set_timestep(num_timesteps)
            self.logger.dumpkvs()

            with open(self.log_path, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                for r, l, s in zip(
                    eval_info["eval/episode_reward"],
                    eval_info["eval/episode_length"],
                    eval_info["eval/episode_success"],
                ):
                    writer.writerow([num_timesteps, r, l, s])

            wandb_log.update(
                {
                    "eval/episode_reward": np.mean(eval_info["eval/episode_reward"]),
                    "eval/episode_success": np.mean(
                        np.array(eval_info["eval/episode_success"]) > 0
                    ),
                    "eval/episode_length": ep_length_mean,
                }
            )
            wandb.log(wandb_log, step=num_timesteps)

            # Save latest policy
            torch.save(
                self.policy.state_dict(),
                os.path.join(self.logger.checkpoint_dir, "policy.pth"),
            )

            # Save best policy
            if norm_ep_rew_mean > best_norm_ep_rew_mean:
                best_norm_ep_rew_mean = norm_ep_rew_mean
                torch.save(
                    self.policy.state_dict(),
                    os.path.join(self.logger.model_dir, "best_policy.pth"),
                )

            if e * self._step_per_epoch in SAVE_POINTS:
                torch.save(
                    self.policy.state_dict(),
                    os.path.join(
                        self.logger.model_dir, f"policy_{e * self._step_per_epoch}.pth"
                    ),
                )

        self.logger.log(f"total time: {time.time() - start_time:.2f}s")
        torch.save(
            self.policy.state_dict(),
            os.path.join(self.logger.model_dir, "last_policy.pth"),
        )
        self.logger.close()
        return

    def _evaluate(self) -> Dict[str, List[float]]:
        self.policy.eval()
        obs = self.eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length, episode_success = 0, 0, 0

        while num_episodes < self._eval_episodes:
            action = self.policy.select_action(obs.reshape(1, -1), deterministic=True)
            next_obs, reward, terminal, info = self.eval_env.step(action.flatten())
            episode_reward += reward
            episode_length += 1
            if "success" in info:
                episode_success += info["success"]

            obs = next_obs
            if terminal:
                eval_ep_info_buffer.append(
                    {
                        "episode_reward": episode_reward,
                        "episode_length": episode_length,
                        "episode_success": episode_success,
                    }
                )
                num_episodes += 1
                episode_reward, episode_length, episode_success = 0, 0, 0
                obs = self.eval_env.reset()

        return {
            "eval/episode_reward": [ep["episode_reward"] for ep in eval_ep_info_buffer],
            "eval/episode_length": [ep["episode_length"] for ep in eval_ep_info_buffer],
            "eval/episode_success": [
                ep["episode_success"] for ep in eval_ep_info_buffer
            ],
        }
