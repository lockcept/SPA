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
        self.processed_data = [
            {
                "s0": {
                    "observations": torch.tensor(item["s0"]["observations"], dtype=torch.float32),
                    "actions": torch.tensor(item["s0"]["actions"], dtype=torch.float32)
                },
                "s1": {
                    "observations": torch.tensor(item["s1"]["observations"], dtype=torch.float32),
                    "actions": torch.tensor(item["s1"]["actions"], dtype=torch.float32)
                },
                "mu": torch.tensor(item["mu"], dtype=torch.float32)
            }
            for item in processed_data
        ]

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        s0 = self.processed_data[idx]["s0"]
        s1 = self.processed_data[idx]["s1"]
        mu = self.processed_data[idx]["mu"]

        return (
            s0["observations"],
            s0["actions"],
            s1["observations"],
            s1["actions"],
            mu,
        )

    def get_dimensions(self):
        s0 = self.processed_data[0]["s0"]
        obs_dim = s0["observations"].shape[-1]
        act_dim = s0["actions"].shape[-1]
        return obs_dim, act_dim


# Collate function to handle variable-length sequences
def collate_fn(batch):
    s0_obs, s0_act, s1_obs, s1_act, mu = zip(*batch)

    s0_obs_padded = rnn_utils.pad_sequence(s0_obs, batch_first=True)
    s0_act_padded = rnn_utils.pad_sequence(s0_act, batch_first=True)
    s1_obs_padded = rnn_utils.pad_sequence(s1_obs, batch_first=True)
    s1_act_padded = rnn_utils.pad_sequence(s1_act, batch_first=True)

    mu = torch.stack(mu)

    return (
        s0_obs_padded,
        s0_act_padded,
        s1_obs_padded,
        s1_act_padded,
        mu,
    )


def get_dataloader(
    env_name,
    pair_name,
    batch_size=32,
    shuffle=True,
    drop_last=True,
):
    processed_data = get_processed_data(env_name, pair_name)

    dataset = PreferenceDataset(processed_data)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )

    obs_dim, act_dim = dataset.get_dimensions()

    print(f"Loaded {pair_name} dataset with {len(dataset)} samples")

    return dataloader, obs_dim, act_dim
