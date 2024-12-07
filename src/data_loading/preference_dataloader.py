import torch
from torch.utils.data import DataLoader, Dataset

from data_loading.load_data import get_processed_data
from helper import get_pair_path


class PreferenceDataset(Dataset):
    """
    Custom Dataset for handling structured (s0, s1, mu) pairs
    """

    def __init__(self, processed_data):
        max_len = max(len(item["s0"]["observations"]) for item in processed_data)

        self.processed_data = []
        for item in processed_data:
            item_len = len(item["s0"]["observations"])
            new_item = {
                "s0": {
                    "observations": None,
                    "actions": None,
                },
                "s1": {
                    "observations": None,
                    "actions": None,
                },
                "mu": torch.tensor(item["mu"], dtype=torch.float32),
                "mask": None,
            }

            s0_obs = torch.tensor(item["s0"]["observations"], dtype=torch.float32)
            s0_act = torch.tensor(item["s0"]["actions"], dtype=torch.float32)
            s1_obs = torch.tensor(item["s1"]["observations"], dtype=torch.float32)
            s1_act = torch.tensor(item["s1"]["actions"], dtype=torch.float32)
            mask = torch.zeros(max_len, dtype=torch.float32)
            mask[item_len:] = 1

            new_item["s0"]["observations"] = self.pad_sequence(s0_obs, max_len)
            new_item["s0"]["actions"] = self.pad_sequence(s0_act, max_len)
            new_item["s1"]["observations"] = self.pad_sequence(s1_obs, max_len)
            new_item["s1"]["actions"] = self.pad_sequence(s1_act, max_len)
            new_item["mask"] = mask.unsqueeze(1)

            self.processed_data.append(new_item)

    def pad_sequence(self, seq, max_len):
        pad_size = max_len - seq.size(0)
        if pad_size > 0:
            padding = torch.zeros(pad_size, seq.size(1), dtype=torch.float32)
            seq = torch.cat([seq, padding], dim=0)
        return seq

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        s0 = self.processed_data[idx]["s0"]
        s1 = self.processed_data[idx]["s1"]
        mu = self.processed_data[idx]["mu"]
        mask = self.processed_data[idx]["mask"]

        return (
            s0["observations"],
            s0["actions"],
            s1["observations"],
            s1["actions"],
            mu,
            mask,
        )

    def get_dimensions(self):
        s0 = self.processed_data[0]["s0"]
        obs_dim = s0["observations"].shape[-1]
        act_dim = s0["actions"].shape[-1]
        return obs_dim, act_dim


def get_dataloader(
    env_name,
    exp_name,
    pair_type,
    pair_algo,
    batch_size=32,
    shuffle=True,
    drop_last=True,
):
    """
    Returns a DataLoader object for the given pair data
    """
    processed_data = get_processed_data(env_name, exp_name, pair_type, pair_algo)

    dataset = PreferenceDataset(processed_data)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    pair_path = get_pair_path(
        env_name=env_name, exp_name=exp_name, pair_type=pair_type, pair_algo=pair_algo
    )

    print(f"Loaded {pair_path} dataset with {len(dataset)} samples")

    return dataloader
