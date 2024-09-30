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

    processed_data = []

    for preference_data in pair["data"]:
        s0, s1, mu = preference_data["s0"], preference_data["s1"], preference_data["mu"]
        print(s0)

        s0 = [dataset["observations"][s0[0] : s0[1]], dataset["actions"][s0[0] : s0[1]]]
        s1 = [dataset["observations"][s1[0] : s1[1]], dataset["actions"][s1[0] : s1[1]]]
        processed_data.append(np.array([s0, s1, mu]))

    return processed_data


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

    pairs = load_processed_data(args.env_name)
