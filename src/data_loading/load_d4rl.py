import os
import gym
import d4rl  # import 해야 gym.make()에서 d4rl 환경을 불러올 수 있음
import numpy as np


def save_d4rl_dataset(env_name):
    env = gym.make(env_name)
    dataset = env.get_dataset()

    save_dir = f"dataset/{env_name}"
    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(save_dir, "d4rl.npz")

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
