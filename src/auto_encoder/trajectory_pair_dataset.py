import torch
from torch.utils.data import Dataset


class TrajectoryPairDataset(Dataset):
    """
    Dataset class for trajectory pairs.
    """

    def __init__(self, dataset, traj_pairs, device):
        self.dataset = dataset
        self.traj_pairs = traj_pairs
        self.obs_dim = dataset["observations"].shape[1]
        self.act_dim = dataset["actions"].shape[1]
        self.observations = torch.tensor(dataset["observations"], dtype=torch.float32)
        self.actions = torch.tensor(dataset["actions"], dtype=torch.float32)
        self.device = device

    def __len__(self):
        return len(self.traj_pairs)

    def __getitem__(self, idx):
        (s1, e1), (s2, e2) = self.traj_pairs[idx]

        obs_1 = self.observations[s1:e1]
        act_1 = self.actions[s1:e1]
        obs_2 = self.observations[s2:e2]
        act_2 = self.actions[s2:e2]

        traj_1 = torch.cat((obs_1, act_1), dim=-1)
        traj_2 = torch.cat((obs_2, act_2), dim=-1)

        traj_pair = torch.cat((traj_1, traj_2), dim=0)

        return traj_pair.view(-1).to(self.device)
