import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """
    Custom Dataset for trajectory autoencoder
    """

    def __init__(self, dataset, indices, device="cpu"):
        self.dataset = dataset
        self.indices = indices
        self.device = device

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start, end = self.indices[idx]

        # ✅ numpy 배열 → Tensor 변환
        obs = torch.tensor(self.dataset["observations"][start:end], dtype=torch.float32)
        act = torch.tensor(self.dataset["actions"][start:end], dtype=torch.float32)

        traj = torch.cat((obs, act), dim=-1)  # (traj_len, obs_dim + act_dim)

        return traj.view(-1).to(self.device)
