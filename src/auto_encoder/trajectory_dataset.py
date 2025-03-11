import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """
    Custom Dataset for trajectory autoencoder
    """

    def __init__(self, dataset, indices, device="cpu"):
        """
        dataset: dict with "observations" and "actions"
        indices: list of (start, end) tuples for trajectory slicing
        device: Target device ("cpu" or "cuda")
        """
        self.device = device
        self.indices = indices

        self.observations = torch.tensor(dataset["observations"], dtype=torch.float32)
        self.actions = torch.tensor(dataset["actions"], dtype=torch.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Pre-processed Tensors를 사용하여 Trajectory를 빠르게 가져옴
        """
        start, end = self.indices[idx]

        obs = self.observations[start:end]  # (traj_len, obs_dim)
        act = self.actions[start:end]  # (traj_len, act_dim)

        traj = torch.cat((obs, act), dim=-1)  # (traj_len, obs_dim + act_dim)

        return traj.view(-1).to(self.device)