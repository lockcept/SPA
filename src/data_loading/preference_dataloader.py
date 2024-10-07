import os
import sys
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.utils.rnn as rnn_utils

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from data_loading.load_dataset import get_processed_data


# Custom Dataset for handling structured (s0, s1, mu) pairs
class PreferenceDataset(Dataset):
    def __init__(self, processed_data):
        self.processed_data = processed_data

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        s0 = self.processed_data[idx]["s0"]
        s1 = self.processed_data[idx]["s1"]
        mu = self.processed_data[idx]["mu"]

        s0_obs = s0["observations"]
        s0_act = s0["actions"]
        s1_obs = s1["observations"]
        s1_act = s1["actions"]

        s0_obs_next = s0_obs[1:]
        s1_obs_next = s1_obs[1:]

        return (
            torch.tensor(s0_obs[:-1], dtype=torch.float32),
            torch.tensor(s0_act[:-1], dtype=torch.float32),
            torch.tensor(s0_obs_next, dtype=torch.float32),
            torch.tensor(s1_obs[:-1], dtype=torch.float32),
            torch.tensor(s1_act[:-1], dtype=torch.float32),
            torch.tensor(s1_obs_next, dtype=torch.float32),
            torch.tensor(mu, dtype=torch.float32),
        )

    def get_dimensions(self):
        s0 = self.processed_data[0]["s0"]
        obs_dim = s0["observations"].shape[-1]
        act_dim = s0["actions"].shape[-1]
        return obs_dim, act_dim


# Collate function to handle variable-length sequences
def collate_fn(batch):
    s0_obs, s0_act, s1_obs, s1_act, mu = zip(*batch)

    # Pad sequences to the same length in the batch
    s0_obs_padded = rnn_utils.pad_sequence(s0_obs, batch_first=True)
    s0_act_padded = rnn_utils.pad_sequence(s0_act, batch_first=True)
    s1_obs_padded = rnn_utils.pad_sequence(s1_obs, batch_first=True)
    s1_act_padded = rnn_utils.pad_sequence(s1_act, batch_first=True)

    mu = torch.stack(mu)  # Stack mu values into a tensor

    return s0_obs_padded, s0_act_padded, s1_obs_padded, s1_act_padded, mu


def get_dataloader(env_name, pair_name, batch_size=32, shuffle=True):
    # Get the processed data (structured array of s0, s1, mu)
    processed_data = get_processed_data(env_name, pair_name)

    # Create a custom dataset
    dataset = PreferenceDataset(processed_data)

    # Create the DataLoader with a custom collate_fn to handle padding
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )

    obs_dim, act_dim = dataset.get_dimensions()

    return dataloader, obs_dim, act_dim
