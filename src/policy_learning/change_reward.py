import torch
import numpy as np

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from data_loading.load_dataset import load_d4rl_dataset


def change_reward(env_name, model, dataset_path):
    dataset = load_d4rl_dataset(env_name)

    observations = torch.tensor(dataset["observations"])
    actions = torch.tensor(dataset["actions"])
    terminals = torch.tensor(dataset["terminals"])
    timeouts = torch.tensor(dataset["timeouts"])

    s_t = observations[:-1]
    s_t_next = observations[1:]
    a_t = actions[:-1]

    predicted_rewards = model(s_t, a_t, s_t_next)

    mask = (terminals[:-1] == 1) | (timeouts[:-1] == 1)
    predicted_rewards[mask] = 0.0

    predicted_rewards = predicted_rewards.detach().cpu().numpy()

    predicted_rewards = list(predicted_rewards)
    predicted_rewards.append(0.0)

    rewards_array = np.array(predicted_rewards)

    print("reward diff example", rewards_array[:10] - dataset["rewards"][:10])

    save_data = {
        "observations": dataset["observations"],
        "actions": dataset["actions"],
        "rewards": rewards_array,
        "terminals": dataset["terminals"],
        "timeouts": dataset["timeouts"],
    }

    np.savez(dataset_path, **save_data)
