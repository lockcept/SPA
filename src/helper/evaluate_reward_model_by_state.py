import numpy as np
import matplotlib.pyplot as plt
import random
import os
import csv
import glob

import torch

from data_loading.load_data import load_dataset
from reward_learning.get_model import get_reward_model
from utils.path import get_reward_model_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_rewards(
    env_name,
    exp_name,
    pair_algo,
    reward_model_algo,
):
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

    for model_file in model_files:
        model, _ = get_reward_model(
            reward_model_algo=reward_model_algo,
            obs_dim=obs_dim,
            act_dim=act_dim,
            model_path=model_file,
            allow_existing=True,
        )
        model_list.append(model)

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

    return predicted_rewards

def evaluate_reward_by_state(
    env_name,
    exp_name,
    pair_algo,
    reward_model_algo,
):
    """
    Evaluate the reward model by comparing predicted rewards with true rewards.
    """
    # Load the dataset

    dataset = load_dataset(env_name)

    save_dir = "reward_eval_stats"
    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(save_dir, "reward_eval_summary.csv")

    write_header = not os.path.exists(csv_path)
    if write_header:
        with open(csv_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["env_name", "exp_name", "pair_algo", "pcc", "order_agreement"])

    predicted_rewards = predict_rewards(
        env_name=env_name,
        exp_name=exp_name,
        pair_algo=pair_algo,
        reward_model_algo=reward_model_algo,
    )

    true_rewards = dataset["rewards"]

    true_rewards = np.array(true_rewards).flatten()
    predicted_rewards = np.array(predicted_rewards).flatten()

    plt.figure(figsize=(6, 6))
    plt.scatter(true_rewards, predicted_rewards, alpha=0.1, s=10, label="Samples")
    plt.xlabel("True Reward")
    plt.ylabel("Predicted Reward")
    plt.title(f"Predicted vs True Rewards ({exp_name} / {pair_algo})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    try:
        pcc = np.corrcoef(true_rewards, predicted_rewards)[0, 1]
    except Exception as e:
        print(f"[{exp_name}] PCC 계산 중 오류 발생: {e}")
        pcc = float('nan')

    num_samples = 100000
    agree = 0
    n = len(true_rewards)

    for _ in range(num_samples):
        i1, i2 = random.sample(range(n), 2)
        gt_diff = true_rewards[i1] - true_rewards[i2]
        pred_diff = predicted_rewards[i1] - predicted_rewards[i2]

        if gt_diff * pred_diff > 0 or (gt_diff == 0 and pred_diff == 0):
            agree += 1

    order_agreement = agree / num_samples

    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            env_name,
            exp_name,
            pair_algo,
            f"{pcc:.6f}",
            f"{order_agreement:.6f}",
        ])