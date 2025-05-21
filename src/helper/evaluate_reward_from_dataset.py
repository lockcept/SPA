import numpy as np
import os
import csv
import random

from data_loading import load_dataset, load_pair
from utils import get_new_dataset_path

def evaluate_existing_reward_dataset(env_name, exp_name, pair_algo, reward_model_algo):
    dataset_path = get_new_dataset_path(env_name, exp_name, pair_algo, reward_model_algo)
    dataset_npz = np.load(dataset_path)
    dataset = {key: dataset_npz[key] for key in dataset_npz}

    predicted_rewards = dataset["rewards"]
    predicted_cumsum = np.cumsum(predicted_rewards, dtype=np.float64)

    # ---- accuracy on test pairs ----
    test_pairs = load_pair(env_name, exp_name, pair_type="test", pair_algo="binary-10000")

    def get_total_reward(s, e):
        return predicted_cumsum[e - 1] - (predicted_cumsum[s - 1] if s > 0 else 0)

    correct_predictions = 0
    total_samples = 0

    for (s0, e0), (s1, e1), mu in test_pairs:
        r0 = get_total_reward(s0, e0)
        r1 = get_total_reward(s1, e1)
        pred_winner = 0 if r0 > r1 else 1
        if mu == pred_winner:
            correct_predictions += 1
        total_samples += 1

    accuracy = correct_predictions / total_samples

    # ---- load raw dataset ----
    raw_dataset = load_dataset(env_name)
    raw_rewards = raw_dataset["rewards"]

    # ---- state-level PCC ----
    pcc = np.corrcoef(np.array(raw_rewards).flatten(), np.array(predicted_rewards).flatten())[0, 1]

    # ---- random state pair ranking agreement ----
    N = len(predicted_rewards)
    num_pairs = 100000
    agreement_count = 0

    for _ in range(num_pairs):
        i, j = random.sample(range(N), 2)
        gt_order = raw_rewards[i] > raw_rewards[j]
        pred_order = predicted_rewards[i] > predicted_rewards[j]
        if gt_order == pred_order:
            agreement_count += 1

    agreement_ratio = agreement_count / num_pairs

    # ---- logging ----
    log_path = "log/eval_existing_dataset.csv"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow([
                "EnvName", "ExpName", "PairAlgo", "RewardModelAlgo",
                "TrajACC", "StatePCC", "StateACC"
            ])
        writer.writerow([
            env_name, exp_name, pair_algo, reward_model_algo,
            f"{accuracy:.4f}", f"{pcc:.4f}", f"{agreement_ratio:.4f}"
        ])