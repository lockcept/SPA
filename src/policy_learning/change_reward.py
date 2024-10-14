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

    s_t = observations[:-1]
    s_t_next = observations[1:]
    a_t = actions[:-1]

    predicted_rewards = model(s_t, a_t, s_t_next)
    predicted_rewards = predicted_rewards.detach().cpu().numpy()
    rewards = np.append(np.array(predicted_rewards), 0)

    terminals = dataset["terminals"] | dataset["timeouts"]
    mask = terminals == 1
    observations = dataset["observations"]
    actions = dataset["actions"]
    rewards[mask] = 0

    print(
        observations.shape,
        actions.shape,
        rewards.shape,
        terminals.shape,
    )

    save_data = {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
    }

    np.savez(dataset_path, **save_data)
