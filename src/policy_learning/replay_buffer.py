import numpy as np
import torch

from typing import Optional, Union, Tuple, Dict


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple,
        obs_dtype: np.dtype,
        action_dim: int,
        action_dtype: np.dtype,
        device: str = "cpu",
    ) -> None:
        self._max_size = buffer_size
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.action_dim = action_dim
        self.action_dtype = action_dtype

        self._ptr = 0
        self._size = 0

        self.observations = np.zeros(
            (self._max_size,) + self.obs_shape, dtype=obs_dtype
        )
        self.next_observations = np.zeros(
            (self._max_size,) + self.obs_shape, dtype=obs_dtype
        )
        self.actions = np.zeros((self._max_size, self.action_dim), dtype=action_dtype)
        self.rewards = np.zeros((self._max_size, 1), dtype=np.float32)
        self.terminals = np.zeros((self._max_size, 1), dtype=np.float32)

        self.device = torch.device(device)

    def load_dataset(self, dataset: Dict[str, np.ndarray]) -> None:
        observations = np.array(dataset["observations"], dtype=self.obs_dtype)
        next_observations = np.array(dataset["next_observations"], dtype=self.obs_dtype)
        actions = np.array(dataset["actions"], dtype=self.action_dtype)
        rewards = np.array(dataset["rewards"], dtype=np.float32).reshape(-1, 1)
        terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1)

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals

        self._ptr = len(observations)
        self._size = len(observations)

        # additionally segment into trajectories based on terminal flag
        self.trajectories = []
        start_idx = 0
        for idx, term in enumerate(terminals):
            if term > 0.5:  # terminal reached
                end_idx = idx + 1
                traj_obs = observations[start_idx:end_idx]
                traj_act = actions[start_idx:end_idx]
                self.trajectories.append(
                    {"observations": traj_obs, "actions": traj_act}
                )
                start_idx = end_idx
        # if leftover (not terminal at end)
        if start_idx < len(observations):
            traj_obs = observations[start_idx:]
            traj_act = actions[start_idx:]
            self.trajectories.append({"observations": traj_obs, "actions": traj_act})

    def sample_trajectory_pair(
        self, batch_size: int = 32, seg_len: int = 25
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Sample batch_size pairs of completely random segments from possibly different trajectories."""
        assert hasattr(self, "trajectories"), "Dataset not segmented yet."
        obs0_list, act0_list, obs1_list, act1_list = [], [], [], []
        for _ in range(batch_size):
            # first segment from one random trajectory
            traj_a = self.trajectories[np.random.randint(0, len(self.trajectories))]
            traj_obs_a = traj_a["observations"]
            traj_act_a = traj_a["actions"]
            traj_len_a = len(traj_obs_a)
            if traj_len_a < seg_len:
                obs_seq0 = traj_obs_a[:seg_len]
                act_seq0 = traj_act_a[:seg_len]
            else:
                max_start_a = traj_len_a - seg_len
                start_a = np.random.randint(0, max_start_a + 1)
                end_a = start_a + seg_len
                obs_seq0 = traj_obs_a[start_a:end_a]
                act_seq0 = traj_act_a[start_a:end_a]

            # second segment from another random trajectory (could be same, sampled with replacement)
            traj_b = self.trajectories[np.random.randint(0, len(self.trajectories))]
            traj_obs_b = traj_b["observations"]
            traj_act_b = traj_b["actions"]
            traj_len_b = len(traj_obs_b)
            if traj_len_b < seg_len:
                obs_seq1 = traj_obs_b[:seg_len]
                act_seq1 = traj_act_b[:seg_len]
            else:
                max_start_b = traj_len_b - seg_len
                start_b = np.random.randint(0, max_start_b + 1)
                end_b = start_b + seg_len
                obs_seq1 = traj_obs_b[start_b:end_b]
                act_seq1 = traj_act_b[start_b:end_b]

            obs0_list.append(obs_seq0)
            act0_list.append(act_seq0)
            obs1_list.append(obs_seq1)
            act1_list.append(act_seq1)
        obs0 = torch.tensor(np.stack(obs0_list), dtype=torch.float32).to(self.device)
        act0 = torch.tensor(np.stack(act0_list), dtype=torch.float32).to(self.device)
        obs1 = torch.tensor(np.stack(obs1_list), dtype=torch.float32).to(self.device)
        act1 = torch.tensor(np.stack(act1_list), dtype=torch.float32).to(self.device)
        return {"obs0": obs0, "act0": act0, "obs1": obs1, "act1": act1}

    def sample_overlapping_pair(
        self, batch_size: int = 32, seg_len: int = 250, overlap_shift: int = 20
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Sample batch_size pairs of nearly overlapping segments sigma and sigma'."""
        assert hasattr(self, "trajectories"), "Dataset not segmented yet."
        obs0_list, act0_list, obs1_list, act1_list = [], [], [], []
        for _ in range(batch_size):
            traj = self.trajectories[np.random.randint(0, len(self.trajectories))]
            traj_obs = traj["observations"]
            traj_act = traj["actions"]
            traj_len = len(traj_obs)
            if traj_len < seg_len + overlap_shift + 1:
                obs_seq = traj_obs[:seg_len]
                act_seq = traj_act[:seg_len]
                obs0_list.append(obs_seq)
                act0_list.append(act_seq)
                obs1_list.append(obs_seq)
                act1_list.append(act_seq)
                continue
            max_start = traj_len - seg_len - overlap_shift
            start = np.random.randint(0, max_start)
            end = start + seg_len
            obs_seq0 = traj_obs[start:end]
            act_seq0 = traj_act[start:end]
            # sample shift from normal distribution N(0, overlap_shift^2)
            shift = int(np.random.normal(loc=0.0, scale=overlap_shift))
            shift_start = min(max(0, start + shift), traj_len - seg_len)
            shift_end = shift_start + seg_len
            obs_seq1 = traj_obs[shift_start:shift_end]
            act_seq1 = traj_act[shift_start:shift_end]
            obs0_list.append(obs_seq0)
            act0_list.append(act_seq0)
            obs1_list.append(obs_seq1)
            act1_list.append(act_seq1)
        obs0 = torch.tensor(np.stack(obs0_list), dtype=torch.float32).to(self.device)
        act0 = torch.tensor(np.stack(act0_list), dtype=torch.float32).to(self.device)
        obs1 = torch.tensor(np.stack(obs1_list), dtype=torch.float32).to(self.device)
        act1 = torch.tensor(np.stack(act1_list), dtype=torch.float32).to(self.device)
        return (
            {"observations": obs0, "actions": act0},
            {"observations": obs1, "actions": act1},
        )

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:

        batch_indexes = np.random.randint(0, self._size, size=batch_size)

        return {
            "observations": torch.tensor(self.observations[batch_indexes]).to(
                self.device
            ),
            "actions": torch.tensor(self.actions[batch_indexes]).to(self.device),
            "next_observations": torch.tensor(self.next_observations[batch_indexes]).to(
                self.device
            ),
            "terminals": torch.tensor(self.terminals[batch_indexes]).to(self.device),
            "rewards": torch.tensor(self.rewards[batch_indexes]).to(self.device),
        }
