from typing import List
import torch
import numpy as np
import os
import sys

from reward_learning.reward_model_base import RewardModelBase

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from data_loading.load_data import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def change_reward(env_name, model_list: List[RewardModelBase], dataset_path):
    dataset = load_dataset(env_name)

    observations = torch.tensor(dataset["observations"], dtype=torch.float32).to(device)
    actions = torch.tensor(dataset["actions"], dtype=torch.float32).to(device)

    model_outputs = []

    for model in model_list:
        rewards = model.batched_forward_trajectory(
            obs_batch=observations, act_batch=actions
        )
        model_outputs.append(rewards.detach().cpu().numpy())

    predicted_rewards = np.mean(model_outputs, axis=0)

    terminals = dataset["terminals"] | dataset["timeouts"]
    observations = dataset["observations"]
    actions = dataset["actions"]
    rewards = predicted_rewards.squeeze()

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
