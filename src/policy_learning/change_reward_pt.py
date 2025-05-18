import glob
import torch
import numpy as np
from tqdm import tqdm

from data_loading import load_dataset
from reward_learning.get_model import get_reward_model
from utils import get_new_dataset_path
from utils.path import get_reward_model_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def change_reward_and_save_pt(env_name, exp_name, pair_algo, seq_len=25):
    # 원본 dataset 불러오기
    dataset = load_dataset(env_name)
    obs = dataset["observations"]
    act = dataset["actions"]
    terminals = dataset["terminals"] | dataset["timeouts"]
    obs_dim = obs.shape[1]
    act_dim = act.shape[1]

    # 모델 불러오기
    reward_model_algo = "PT"
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
            rewards = []
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

                input = {
                    "observations": torch.tensor(
                        padded_obs[None], dtype=torch.float32
                    ).to(device),
                    "actions": torch.tensor(padded_act[None], dtype=torch.float32).to(
                        device
                    ),
                    "timestep": torch.tensor(timestep[None], dtype=torch.long).to(
                        device
                    ),
                    "attn_mask": torch.tensor(attn_mask[None], dtype=torch.float32).to(
                        device
                    ),
                }
                if torch.isnan(input["observations"]).any():
                    print(f"⚠️ NaN in obs at t={t}")
                if torch.isnan(input["actions"]).any():
                    print(f"⚠️ NaN in act at t={t}")
                if torch.isnan(input["timestep"]).any():
                    print(f"⚠️ NaN in timestep at t={t}")
                if torch.isnan(input["attn_mask"]).any():
                    print(f"⚠️ NaN in attn_mask at t={t}")
                if input["attn_mask"].sum() == 0:
                    print(f"⚠️ All attn_mask=0 at t={t}")

                with torch.no_grad():
                    reward = reward_model(
                        input["observations"],
                        input["actions"],
                        input["timestep"],
                        input["attn_mask"],
                    )
                    rewards.append(reward.item())

            rewards_per_model.append(np.array(rewards))

        # 모델 평균 reward
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
