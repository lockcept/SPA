import csv
import itertools
import os
import torch
import numpy as np
from data_generation.picker.mr_dropout import mr_dropout_test
from data_generation.utils import generate_pairs_from_indices
from data_loading.load_data import load_dataset, load_pair
from reward_learning.get_model import get_reward_model
from reward_learning.train_model import train_reward_model
from utils.path import get_reward_model_path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def categorize_mu(mu_value):
    if mu_value < 1 / 3:
        return 0.0
    elif mu_value > 2 / 3:
        return 1.0
    else:
        return 0.5

def test1(env_name, exp_name):
    """
    Research function for data generation and analysis.
    """
    data = mr_dropout_test(
        env_name=env_name,
        exp_name=exp_name,
        pair_algo="ternary-500",
        device=device,
    )

    save_dir = "reward_eval_stats"
    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(save_dir, "MCdropout.csv")

    write_header = not os.path.exists(csv_path)
    if write_header:
        with open(csv_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "env_name",
                    "exp_name",
                    "pair_algo",
                    "correct_ratio",
                    "incorrect_ratio",
                    "correct_reward_mean",
                    "incorrect_reward_mean",
                    "correct_uncertainty_mean",
                    "incorrect_uncertainty_mean",
                    "correct_uncertainty_std",
                    "incorrect_uncertainty_std",
                ]
            )

    correct = []
    incorrect = []

    correct_reward = []
    incorrect_reward = []

    for datum in data:
        (
            (s0, e0),
            (s1, e1),
            predicted_mu,
            mu,
            uncertainty_0,
            uncertainty_1,
            predicted_sum_of_rewards_0,
            predicted_sum_of_rewards_1,
        ) = datum
        
        if categorize_mu(predicted_mu) == 0.5:
            continue

        if categorize_mu(predicted_mu) == categorize_mu(mu):
            correct.append(uncertainty_0)
            correct.append(uncertainty_1)
            correct_reward.append(predicted_sum_of_rewards_0)
            correct_reward.append(predicted_sum_of_rewards_1)
        else:
            incorrect.append(uncertainty_0)
            incorrect.append(uncertainty_1)
            incorrect_reward.append(predicted_sum_of_rewards_0)
            incorrect_reward.append(predicted_sum_of_rewards_1)

    print("Correct predictions:", len(correct) / 2, np.mean(correct))
    print("Incorrect predictions:", len(incorrect) / 2, np.mean(incorrect))

    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                env_name,
                exp_name,
                "ternary-500",
                round(len(correct) / (len(correct) + len(incorrect)), 4),
                round(len(incorrect) / (len(correct) + len(incorrect)), 4),
                round(np.mean(correct_reward), 4),
                round(np.mean(incorrect_reward), 4),
                round(np.mean(correct), 4),
                round(np.mean(incorrect), 4),
                round(np.std(correct), 4),
                round(np.std(incorrect), 4),
            ]
        )


def test2(env_name, exp_name):
    """
    Research function for uncertainty-based top-k accuracy evaluation.
    Selects top-K high uncertainty pairs and evaluates correctness ratio.
    """
    data = mr_dropout_test(
        env_name=env_name,
        exp_name=exp_name,
        pair_algo="ternary-500",
        device=device,
    )



    filtered_data = [d for d in data if categorize_mu(d[2]) != 0.5]

    print("data length:", len(data))
    print("Filtered data length:", len(filtered_data))


    # 필터 기준: uncertainty 합
    ranked_data = sorted(
        filtered_data,
        key=lambda d: (d[4] + d[5]),  # predicted_std_0 + predicted_std_1
        reverse=False
    )

    # Top-k 정확도 계산
    top_k_values = [100, 1000, 10000, 100000]
    # top_k_values = [100]
    results = []

    for k in top_k_values:
        top_k = ranked_data[:k]

        correct_count = 0
        total_count = 0

        for datum in top_k:
            (s0, e0), (s1, e1), predicted_mu, mu, _, _, predicted_sum_of_rewards_0, predicted_sum_of_rewards_1 = datum
            if categorize_mu(predicted_mu) == mu:
                correct_count += 1
            total_count += 1

            # print(predicted_sum_of_rewards_0, predicted_sum_of_rewards_1, np.sum(dataset["rewards"][s0:e0]), np.sum(dataset["rewards"][s1:e1]), predicted_mu, mu)

        accuracy = correct_count / total_count if total_count > 0 else 0.0
        print(f"Top-{k} accuracy: {accuracy:.4f} ({correct_count}/{total_count})")
        results.append((k, accuracy, correct_count, total_count))

    # CSV 저장
    save_dir = "reward_eval_stats"
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "MCdropout_topk.csv")

    write_header = not os.path.exists(csv_path)
    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["env_name", "exp_name", "pair_algo", "top_k", "accuracy", "correct", "total"])

        for k, acc, correct, total in results:
            writer.writerow([env_name, exp_name, "ternary-500", k, acc, correct, total])

def test3(env_name, exp_name):
    data = mr_dropout_test(
        env_name=env_name,
        exp_name=exp_name,
        pair_algo="ternary-500",
        device=device,
    )

    trajectories = []
    for i in range(10000):
        traj = data[i][0]
        predicted_reward = data[i][6]
        uncertainty = data[i][4]
        trajectories.append((traj, predicted_reward, uncertainty))


    dataset = load_dataset(env_name)
    reward_cumsum = np.cumsum(dataset["rewards"], axis=0, dtype=np.float64)

    top_100_certainty = sorted(
        trajectories,
        key=lambda x: x[2],  # uncertainty
        reverse=True
    )[:100]

    # 100C2 쌍 만들기
    trajectory_pairs = list(itertools.combinations(top_100_certainty, 2))

    correct = 0
    total = 0

    for (traj0, pred_r0, _), (traj1, pred_r1, _) in trajectory_pairs:
        total += 1

        # 예측 label 계산
        pred_mu = (pred_r1 + 1e-6) / (pred_r0 + pred_r1 + 2e-6)
        pred_label = int(pred_mu > 0.5)

        # 실제 label 계산
        s0, e0 = traj0
        s1, e1 = traj1
        sum_r0 = reward_cumsum[e0 - 1] - (reward_cumsum[s0 - 1] if s0 > 0 else 0)
        sum_r1 = reward_cumsum[e1 - 1] - (reward_cumsum[s1 - 1] if s1 > 0 else 0)
        true_label = int(sum_r1 > sum_r0)

        if pred_label == true_label:
            correct += 1


    # CSV 저장
    save_dir = "reward_eval_stats"
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "MCdropout_topk.csv")

    write_header = not os.path.exists(csv_path)
    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["env_name", "exp_name", "pair_algo", "top_k", "accuracy", "correct", "total"])

        writer.writerow([env_name, exp_name, "ternary-500", 103, round(correct/total,4), correct, total])
    
def test4(env_name, exp_name):
    for i in range(7):
        train_reward_model(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo="ternary-500",
            reward_model_algo="MR-exp",
            reward_model_tag=f"{i:02d}",
            num_epoch=500,
        )

    # Load the trained model
    models = []

    dataset = load_dataset(env_name)

    for i in range(7):
        model_path = get_reward_model_path(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo="ternary-500",
            reward_model_algo="MR-exp",
            reward_model_tag=f"{i:02d}",
        )
        model, _ = get_reward_model(
            reward_model_algo="MR-exp",
            model_path=model_path,
            allow_existing=True,
            obs_dim=dataset["observations"].shape[1],
            act_dim=dataset["actions"].shape[1],
        )
        models.append(model)
    
    # Load the dataset
    dataset = load_dataset(env_name)

    true_rewards = dataset["rewards"]

    def compute_rewards(model, dataset):
        num_samples = len(dataset["observations"])
        batch_size = num_samples // 20

        model_rewards = []
        model_logits = []

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)

            obs_batch = torch.tensor(
                dataset["observations"][start_idx:end_idx], dtype=torch.float32
            ).to(device)
            act_batch = torch.tensor(
                dataset["actions"][start_idx:end_idx], dtype=torch.float32
            ).to(device)

            rewards, logits = model(
                obs_batch, act_batch, return_logit=True
            )
            model_rewards.append(rewards.cpu().detach().numpy())
            model_logits.append(logits.cpu().detach().numpy())

        return np.concatenate(model_rewards, axis=0), np.concatenate(model_logits, axis=0)

    predicted_rewards = []
    predicted_logits = []

    for model in models:
        pred_rewards, pred_logits = compute_rewards(model, dataset)
        predicted_rewards.append(pred_rewards)
        predicted_logits.append(pred_logits)

    def normalize_preds(preds, method="z"):
        preds = preds.astype(np.float64)
        if method == "z":
            mean = preds.mean()
            std = preds.std() + 1e-8
            return (preds - mean) / std
        elif method == "minmax":
            minv = preds.min()
            maxv = preds.max() + 1e-8
            print(minv, maxv)
            return (preds - minv) / (maxv - minv)
        elif method == "none":
            return preds
        else:
            raise ValueError("Unknown normalization method")

    normalized_preds = []
    for preds in predicted_rewards:
        print (preds.min(), preds.max())
        norm_preds = normalize_preds(preds, method="none")
        print(norm_preds.min(), norm_preds.max())
        normalized_preds.append(norm_preds)

    normalized_preds_array = np.stack(normalized_preds, axis=0)  # [N, T]

    pred_mean = normalized_preds_array.mean(axis=0)  # [T]
    pred_std = normalized_preds_array.std(axis=0)    # [T]

    logit_std = np.array(predicted_logits).std(axis=0)  # [T]

    result = list(zip(true_rewards, pred_mean, pred_std, logit_std))  # [(true, mean, std, logit_std), ...]

    return result


def data_research(env_name, exp_name):
    """
    Research function for data generation and analysis.
    """
    test4(env_name, exp_name)
