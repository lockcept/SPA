import os
import numpy as np
import argparse


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


def load_processed_data(env_name, pair_name="full_preference_pairs.npz"):
    dataset = load_d4rl_dataset(env_name)
    pair = load_pair(env_name, pair_name)

    # todo: check if the dataset and pair are compatible
    return dataset, pair


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

    pairs = load_pair(args.env_name)
