import numpy as np
import torch
from tqdm import tqdm

from data_loading.load_data import load_dataset, load_pair
from reward_learning.get_model import get_reward_model
from reward_learning.train_model import train_reward_model
from utils import get_reward_model_path
from data_generation.utils import generate_pairs_from_indices

TRAJECTORY_LENGTH = 25
PAIR_COUNT = 100000


def predict_rewards_with_mc_dropout(
    model, obs, act, device, batch_size=1024, mc_passes=10
):
    """
    전체 obs, act에 대해 MC Dropout을 적용한 reward mean 및 std 계산 (batch 단위 처리)

    Args:
        model: MR-dropout reward model
        obs: numpy array of shape [T, obs_dim]
        act: numpy array of shape [T, act_dim]
        device: torch device
        batch_size: int, batch size for prediction
        mc_passes: int, number of MC forward passes

    Returns:
        mean_rewards: [T] torch.FloatTensor
        std_rewards: [T] torch.FloatTensor
    """
    model.train()  # Dropout 활성화
    total_len = obs.shape[0]

    mean_rewards = []
    std_rewards = []

    for start in tqdm(range(0, total_len, batch_size), desc="MC Dropout Batching"):
        end = min(start + batch_size, total_len)

        obs_batch = torch.tensor(obs[start:end], dtype=torch.float32, device=device)
        act_batch = torch.tensor(act[start:end], dtype=torch.float32, device=device)

        with torch.no_grad():
            mean_batch, std_batch = model.forward_mc(
                obs_batch, act_batch, mc_passes=mc_passes
            )
            mean_batch = mean_batch.squeeze(-1)
            std_batch = std_batch.squeeze(-1)

        mean_rewards.append(mean_batch)
        std_rewards.append(std_batch)

    mean_rewards = torch.cat(mean_rewards, dim=0)  # [T]
    std_rewards = torch.cat(std_rewards, dim=0)  # [T]

    return mean_rewards, std_rewards


def mr_dropout_test(env_name, exp_name, pair_algo, device="cuda"):
    reward_model_algo = "MR-dropout"

    # 1. 학습
    train_reward_model(
        env_name=env_name,
        exp_name=exp_name,
        pair_algo=pair_algo,
        reward_model_algo=reward_model_algo,
        reward_model_tag="00",
        num_epoch=100,
    )

    # 2. 모델 로드
    dataset = load_dataset(env_name)
    obs, act = dataset["observations"], dataset["actions"]
    obs_dim, act_dim = obs.shape[1], act.shape[1]

    model_path = get_reward_model_path(
        env_name, exp_name, pair_algo, reward_model_algo, "00"
    )

    model, _ = get_reward_model(
        reward_model_algo=reward_model_algo,
        obs_dim=obs_dim,
        act_dim=act_dim,
        model_path=model_path,
        allow_existing=True,
    )
    model.to(device)
    model.eval()

    # 3. pair 생성
    train_all_pairs_with_mu = load_pair(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train_all",
        pair_algo="raw",
    )

    all_traj_set = []

    for p in train_all_pairs_with_mu:
        all_traj_set.append(p[0])
        all_traj_set.append(p[1])

    pair_candidates = generate_pairs_from_indices(
        trajectories=all_traj_set,
        pair_count=100000,
        trajectory_length=TRAJECTORY_LENGTH,
    )

    # 4. MC Dropout 예측
    mean_rewards, std_rewards = predict_rewards_with_mc_dropout(
        model=model, obs=obs, act=act, device=device, batch_size=1024, mc_passes=10
    )

    predicted_mean_cumsum = np.cumsum(
        mean_rewards.cpu().numpy(), axis=0, dtype=np.float64
    )
    predicted_std_cumsum = np.cumsum(
        std_rewards.cpu().numpy(), axis=0, dtype=np.float64
    )

    reward_cumsum = np.cumsum(dataset["rewards"], axis=0, dtype=np.float64)

    data = []

    for i0, i1 in pair_candidates:
        s0, e0 = i0
        s1, e1 = i1

        predicted_sum_of_rewards_0 = predicted_mean_cumsum[e0 - 1] - (
            predicted_mean_cumsum[s0 - 1] if s0 > 0 else 0
        )
        predicted_sum_of_rewards_1 = predicted_mean_cumsum[e1 - 1] - (
            predicted_mean_cumsum[s1 - 1] if s1 > 0 else 0
        )

        predicted_mu = (predicted_sum_of_rewards_1 + 1e-6) / (
            predicted_sum_of_rewards_0 + predicted_sum_of_rewards_1 + 2e-6
        )

        predicted_std_0 = predicted_std_cumsum[e0 - 1] - (
            predicted_std_cumsum[s0 - 1] if s0 > 0 else 0
        )

        predicted_std_1 = predicted_std_cumsum[e1 - 1] - (
            predicted_std_cumsum[s1 - 1] if s1 > 0 else 0
        )

        sum_of_rewards_0 = reward_cumsum[e0 - 1] - (
            reward_cumsum[s0 - 1] if s0 > 0 else 0
        )
        sum_of_rewards_1 = reward_cumsum[e1 - 1] - (
            reward_cumsum[s1 - 1] if s1 > 0 else 0
        )
        mu = (sum_of_rewards_1 + 1e-6) / (sum_of_rewards_0 + sum_of_rewards_1 + 2e-6)

        data.append(
            (
                (s0, e0),
                (s1, e1),
                predicted_mu,
                mu,
                predicted_std_0,
                predicted_std_1,
            )
        )

    return data
