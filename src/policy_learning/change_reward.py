import glob
from typing import List
import torch
import numpy as np

from data_loading import get_dataloader, load_dataset
from reward_learning import MR, RewardModelBase

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


def change_reward_from_all_datasets(
    env_name, pair_name, reward_model_algo, dataset_name, new_dataset_path
):
    """
    change reward and save it to new_dataset_path
    use all reward models in model/{env_name}/reward/{dataset_name}_*.pth
    """

    _, obs_dim, act_dim = get_dataloader(env_name=env_name, pair_name=pair_name)

    print("obs_dim:", obs_dim, "act_dim:", act_dim)
    model_path_pattern = f"model/{env_name}/reward/{dataset_name}_*.pth"
    model_files = glob.glob(model_path_pattern)
    model_list = []

    if reward_model_algo == "MR":
        for model_file in model_files:
            model, _ = MR.initialize(
                config={"obs_dim": obs_dim, "act_dim": act_dim}, path=model_file
            )
            model_list.append(model)
    elif reward_model_algo == "MR-linear":
        for model_file in model_files:
            model, _ = MR.initialize(
                config={"obs_dim": obs_dim, "act_dim": act_dim},
                path=model_file,
                linear_loss=True,
            )
            model_list.append(model)

    change_reward(
        env_name=env_name, model_list=model_list, dataset_path=new_dataset_path
    )
