import os
import gym
import d4rl  # import 해야 gym.make()에서 d4rl 환경을 불러올 수 있음
import numpy as np
import argparse


def save_dataset(env_name, dataset):
    save_dir = f"dataset/d4rl/{env_name}"
    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(save_dir, "dataset.npz")

    save_data = {
        "observations": dataset["observations"],
        "actions": dataset["actions"],
        "rewards": dataset["rewards"],
        "terminals": dataset["terminals"],
        "timeouts": dataset["timeouts"],
    }

    for key in dataset.keys():
        if key.startswith("infos"):
            save_data[key] = dataset[key]

    np.savez(file_path, **save_data)

    print(f"Dataset saved with keys: {save_data.keys()}")


def load(env_name):
    antmaze_env = gym.make(env_name)
    antmaze_dataset = antmaze_env.get_dataset()

    save_dataset(env_name, antmaze_dataset)

    return antmaze_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load and save D4RL dataset for a given environment."
    )

    parser.add_argument(
        "--env_name",
        type=str,
        default="maze2d-medium-dense-v1",
        help="Name of the environment to load the dataset for",
    )

    args = parser.parse_args()

    load(args.env_name)
