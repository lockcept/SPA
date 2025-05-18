import glob
import torch
import numpy as np
from tqdm import tqdm

from data_loading import load_dataset
from reward_learning.get_model import get_reward_model
from utils import get_new_dataset_path
from utils.path import get_reward_model_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def change_reward_and_save_pt(env_name, exp_name, pair_algo, is_linear=False, seq_len=25):
    # 원본 dataset 불러오기
    dataset = load_dataset(env_name)
    obs = dataset["observations"]
    act = dataset["actions"]
    terminals = dataset["terminals"] | dataset["timeouts"]
    obs_dim = obs.shape[1]
    act_dim = act.shape[1]

    # 모델 불러오기
    if is_linear:
        reward_model_algo = "PT-linear"
    else:
        reward_model_algo = "PT-exp"
    model_path_pattern = get_reward_model_path(
        env_name=env_name,
        exp_name=exp_name,
        pair_algo=pair_algo,
        reward_model_algo=reward_model_algo,
        reward_model_tag="*",
    )
    model_files = glob.glob(model_path_pattern)
    model_list = []

    for model_file in model_files:
        model, _ = get_reward_model(
            reward_model_algo=reward_model_algo,
            obs_dim=obs_dim,
            act_dim=act_dim,
            model_path=model_file,
            allow_existing=True,
        )
        model_list.append(model.to(device))

    assert len(model_list) > 0, "No reward model found."

    # trajectory 분할
    trajectories = []
    start = 0
    for i in range(len(obs)):
        if terminals[i] == 1 or i == len(obs) - 1:
            end = i + 1
            trajectories.append((obs[start:end], act[start:end]))
            start = end

    predicted_rewards = []

    for traj_obs, traj_act in tqdm(trajectories, desc="Relabel Trajectory"):
        T = len(traj_obs)
        rewards_per_model = []

        for reward_model in model_list:
            padded_obs_list = []
            padded_act_list = []
            timestep_list = []
            attn_mask_list = []

            for t in range(T):
                if t < seq_len - 1:
                    padded_obs = np.concatenate(
                        [np.zeros((seq_len - 1 - t, obs_dim)), traj_obs[: t + 1]],
                        axis=0,
                    )
                    padded_act = np.concatenate(
                        [np.zeros((seq_len - 1 - t, act_dim)), traj_act[: t + 1]],
                        axis=0,
                    )
                    timestep = np.concatenate(
                        [
                            np.zeros(seq_len - 1 - t, dtype=np.int32),
                            np.arange(1, t + 2),
                        ],
                        axis=0,
                    )
                    attn_mask = np.concatenate(
                        [np.zeros(seq_len - 1 - t), np.ones(t + 1)], axis=0
                    )
                else:
                    padded_obs = traj_obs[t - seq_len + 1 : t + 1]
                    padded_act = traj_act[t - seq_len + 1 : t + 1]
                    timestep = np.arange(1, seq_len + 1)
                    attn_mask = np.ones(seq_len)

                padded_obs_list.append(padded_obs)
                padded_act_list.append(padded_act)
                timestep_list.append(timestep)
                attn_mask_list.append(attn_mask)

            input = {
                "observations": torch.tensor(
                    np.array(padded_obs_list), dtype=torch.float32
                ).to(device),
                "actions": torch.tensor(
                    np.array(padded_act_list), dtype=torch.float32
                ).to(device),
                "timestep": torch.tensor(np.array(timestep_list), dtype=torch.int32).to(
                    device
                ),
                "attn_mask": torch.tensor(
                    np.array(attn_mask_list), dtype=torch.float32
                ).to(device),
            }

            with torch.no_grad():
                reward = reward_model(
                    input["observations"],  # shape [T, seq_len, obs_dim]
                    input["actions"],  # shape [T, seq_len, act_dim]
                    input["timestep"],  # shape [T, seq_len]
                    input["attn_mask"],  # shape [T, seq_len]
                )
                rewards = reward.squeeze(-1).cpu().numpy()  # shape [T]

            rewards_per_model.append(rewards)

        mean_rewards = np.mean(rewards_per_model, axis=0)
        predicted_rewards.append(mean_rewards)

    # 전부 이어붙이기
    new_rewards = np.concatenate(predicted_rewards)
    assert new_rewards.shape[0] == obs.shape[0], "Reward length mismatch"

    save_data = {
        "observations": obs,
        "actions": act,
        "rewards": new_rewards,
        "terminals": terminals,
    }
    dataset_path = get_new_dataset_path(
        env_name=env_name,
        exp_name=exp_name,
        pair_algo=pair_algo,
        reward_model_algo=reward_model_algo,
    )
    np.savez(dataset_path, **save_data)
