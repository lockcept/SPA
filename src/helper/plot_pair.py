import csv
import os
from matplotlib import pyplot as plt
import numpy as np
from data_loading import load_dataset, load_pair
from utils import get_pair_log_path


def plot_pair(env_name_list, exp_name, pair_algo_list):
    for env_name in env_name_list:
        for pair_algo in pair_algo_list:
            pairs = load_pair(
                env_name=env_name,
                exp_name=exp_name,
                pair_type="train",
                pair_algo=pair_algo,
            )["data"]

            log_path = get_pair_log_path(
                env_name=env_name,
                exp_name=exp_name,
                pair_type="train",
                pair_algo=pair_algo,
                log_file="mu_histogram.png",
            )

            # histogram of mu values
            mu_values = [item["mu"] for item in pairs]
            plt.figure(figsize=(10, 6))
            plt.hist(mu_values, bins=50, alpha=0.75)
            plt.xlabel("Mu Values")
            plt.ylabel("Frequency")
            plt.title(f"{env_name}_{exp_name}_{pair_algo}")
            plt.grid(True)
            plt.savefig(log_path)


def evaluate_pair(env_name, exp_name, pair_type, pair_algo):
    dataset = load_dataset(env_name)
    data = load_pair(env_name, exp_name, pair_type, pair_algo)

    answer_count = 0
    total_count = 0

    true_values = []
    eval_values = []

    for s0, s1, mu in data["data"]:
        rewards_sum_0 = np.sum(dataset["rewards"][s0[0] : s0[1]])
        rewards_sum_1 = np.sum(dataset["rewards"][s1[0] : s1[1]])

        total_count += 1

        true_values.append(1 / (1 + np.exp(rewards_sum_0 - rewards_sum_1)))
        eval_values.append(mu)

        if rewards_sum_0 <= rewards_sum_1 and mu >= 0.5:
            answer_count += 1
        elif rewards_sum_0 >= rewards_sum_1 and mu <= 0.5:
            answer_count += 1

    plt.figure(figsize=(8, 6))
    plt.scatter(true_values, eval_values, alpha=0.7, edgecolors="k")

    plt.fill_betweenx([0, 0.5], 0, 0.5, color="red", alpha=0.2, label="(0~0.5, 0~0.5)")
    plt.fill_betweenx(
        [0.5, 1.0], 0.5, 1.0, color="red", alpha=0.2, label="(0.5~1.0, 0.5~1.0)"
    )

    plt.xlabel("true_mu")
    plt.ylabel("eval_mu")
    plt.title("Mu Relationship")
    plt.grid(True)
    plt.savefig(
        get_pair_log_path(
            env_name=env_name,
            exp_name=exp_name,
            pair_type="train",
            pair_algo=pair_algo,
            log_file="vs_true_mu.png",
        ),
        format="png",
    )
    plt.close()

    log_path = "log/main_evaluate_pair.csv"
    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    accuracy = answer_count / total_count

    with open(log_path, "a", encoding="utf-8", newline="") as log_file:
        writer = csv.writer(log_file)

        if log_file.tell() == 0:
            writer.writerow(
                [
                    "EnvName",
                    "ExpName",
                    "PairAlgo",
                    "Accuracy",
                ]
            )

        formatted_accuracy = f"{accuracy:.4f}"

        writer.writerow(
            [
                env_name,
                exp_name,
                pair_algo,
                formatted_accuracy,
            ]
        )
