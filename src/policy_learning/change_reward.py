import torch
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from data_loading.load_dataset import load_d4rl_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def change_reward(env_name, model_list, dataset_path):
    dataset = load_d4rl_dataset(env_name)

    observations = torch.tensor(dataset["observations"]).to(device)
    actions = torch.tensor(dataset["actions"]).to(device)

    s_t = observations[:-1]
    s_t_next = observations[1:]
    a_t = actions[:-1]

    model_outputs = []
    for model in model_list:
        model_output = model(s_t, a_t, s_t_next)
        model_outputs.append(model_output.detach().cpu().numpy())

    predicted_rewards = np.mean(model_outputs, axis=0)
    rewards = np.append(predicted_rewards, 0)

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