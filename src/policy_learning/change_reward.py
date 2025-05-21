import glob
from typing import List
import torch
import numpy as np

from data_loading import load_dataset
from reward_learning import RewardModelBase, get_reward_model
from utils import get_reward_model_path, get_new_dataset_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def change_reward(env_name, model_list: List[RewardModelBase], dataset_path):
    dataset = load_dataset(env_name)

    num_samples = len(dataset["observations"])
    batch_size = num_samples // 20
    model_outputs = []

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)

        obs_batch = torch.tensor(
            dataset["observations"][start_idx:end_idx], dtype=torch.float32
        ).to(device)
        act_batch = torch.tensor(
            dataset["actions"][start_idx:end_idx], dtype=torch.float32
        ).to(device)

        batch_model_outputs = []
        for model in model_list:
            rewards = model.batched_forward_trajectory(
                obs_batch=obs_batch, act_batch=act_batch
            )
            batch_model_outputs.append(rewards.detach().cpu().numpy())

        batch_predicted_rewards = np.mean(batch_model_outputs, axis=0)
        model_outputs.append(batch_predicted_rewards)

    predicted_rewards = np.concatenate(model_outputs, axis=0).squeeze()

    terminals = dataset["terminals"] | dataset["timeouts"]
    observations = dataset["observations"]
    actions = dataset["actions"]
    rewards = predicted_rewards

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


def change_reward_from_all_datasets(env_name, exp_name, pair_algo, reward_model_algo):
    """
    change reward and save it to new_dataset_path
    use all reward models in model/{env_name}/reward/{dataset_name}_*.pth
    """

    dataset = load_dataset(env_name)
    obs_dim = dataset["observations"].shape[1]
    act_dim = dataset["actions"].shape[1]

    print("obs_dim:", obs_dim, "act_dim:", act_dim)
    model_path_pattern = get_reward_model_path(
        env_name=env_name,
        exp_name=exp_name,
        pair_algo=pair_algo,
        reward_model_algo=reward_model_algo,
        reward_model_tag="*",
    )

    model_files = glob.glob(model_path_pattern)
    model_list = []

    # 사전순 정렬
    model_files_sorted = sorted(model_files)

    # 상위 n개만 사용
    n = 3  # 원하는 개수로 설정
    model_files = model_files_sorted[:n]

    print(f"Using top {n} models:")
    for path in model_files:
        print(" -", path)

    for model_file in model_files:
        model, _ = get_reward_model(
            reward_model_algo=reward_model_algo,
            obs_dim=obs_dim,
            act_dim=act_dim,
            model_path=model_file,
            allow_existing=True,
        )
        model_list.append(model)

    new_dataset_path = get_new_dataset_path(
        env_name, exp_name, pair_algo, reward_model_algo
    )

    change_reward(
        env_name=env_name, model_list=model_list, dataset_path=new_dataset_path
    )
