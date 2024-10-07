import os
import numpy as np
import argparse

import torch
from torch.utils.data import Dataset, DataLoader

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from config import DEFAULT_ENV_NAME


def load_d4rl_dataset(env_name):
    dir_path = f"dataset/{env_name}"
    dataset_name = "d4rl_dataset.npz"
    dataset = np.load(os.path.join(dir_path, dataset_name))

    return dataset


def load_pair(env_name, pair_name="full_preference_pairs.npz"):
    dir_path = f"dataset/{env_name}"
    pair = np.load(os.path.join(dir_path, pair_name), allow_pickle=True)

    return pair


class PreferenceDataset(Dataset):
    def __init__(
        self, s0_observations, s0_actions, s1_observations, s1_actions, mu_labels
    ):
        self.s0_observations = s0_observations
        self.s0_actions = s0_actions
        self.s1_observations = s1_observations
        self.s1_actions = s1_actions
        self.mu_labels = mu_labels

    def __len__(self):
        return len(self.mu_labels)

    def get_dimensions(self):
        obs_dim = self.s0_observations.shape[1:]
        act_dim = self.s0_actions.shape[1:]
        return obs_dim, act_dim

    def __getitem__(self, idx):
        s0_obs = self.s0_observations[idx]
        s0_act = self.s0_actions[idx]
        s1_obs = self.s1_observations[idx]
        s1_act = self.s1_actions[idx]
        mu = self.mu_labels[idx]

        return s0_obs, s0_act, s1_obs, s1_act, mu


def get_processed_data(env_name, pair_name="full_preference_pairs.npz"):
    dataset = load_d4rl_dataset(env_name)
    observations = dataset["observations"]
    actions = dataset["actions"]

    pair = load_pair(env_name, pair_name)

    processed_data = []

    for entry in pair["data"]:
        s0_idx, s1_idx, mu = entry["s0"], entry["s1"], entry["mu"]

        s0_obs = observations[s0_idx[0] : s0_idx[1]]
        s0_act = actions[s0_idx[0] : s0_idx[1]]
        s1_obs = observations[s1_idx[0] : s1_idx[1]]
        s1_act = actions[s1_idx[0] : s1_idx[1]]
        mu = mu

        s0 = np.array(
            list(zip(observations, s0_act)),
            dtype=[
                ("observations", "f4", (s0_obs.shape[1],)),
                ("actions", "f4", (s0_act.shape[1],)),
            ],
        )
        s1 = np.array(
            list(zip(observations, s1_act)),
            dtype=[
                ("observations", "f4", (s1_obs.shape[1],)),
                ("actions", "f4", (s1_act.shape[1],)),
            ],
        )

        processed_data.append(
            (
                s0,
                s1,
                mu,
            )
        )

    return np.array(processed_data, dtype=[("s0", "O"), ("s1", "O"), ("mu", "f4")])


def get_dataloader(
    env_name, pair_name="full_preference_pairs.npz", batch_size=32, shuffle=True
):
    s0_observations = []
    s0_actions = []
    s1_observations = []
    s1_actions = []
    mu_labels = []

    processed_data = get_processed_data(env_name, pair_name)

    for entry in processed_data:
        s0_obs, s0_act = (
            entry["s0"]["observations"],
            entry["s0"]["actions"],
        )
        s1_obs, s1_act = (
            entry["s1"]["observations"],
            entry["s1"]["actions"],
        )
        mu = entry["mu"]

        s0_observations.append(torch.tensor(s0_obs, dtype=torch.float32))
        s0_actions.append(torch.tensor(s0_act, dtype=torch.float32))
        s1_observations.append(torch.tensor(s1_obs, dtype=torch.float32))
        s1_actions.append(torch.tensor(s1_act, dtype=torch.float32))
        mu_labels.append(torch.tensor(mu, dtype=torch.float32))

    s0_observations = torch.stack(s0_observations)
    s0_actions = torch.stack(s0_actions)
    s1_observations = torch.stack(s1_observations)
    s1_actions = torch.stack(s1_actions)
    mu_labels = torch.stack(mu_labels)

    dataset = PreferenceDataset(
        s0_observations, s0_actions, s1_observations, s1_actions, mu_labels
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load and save D4RL dataset for a given environment."
    )

    parser.add_argument(
        "--env_name",
        type=str,
        default=DEFAULT_ENV_NAME,
        help="Name of the environment to load the dataset for",
    )

    args = parser.parse_args()

    dataloader = get_dataloader(args.env_name)
    print(dataloader)
