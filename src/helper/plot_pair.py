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
            )

            # histogram of mu values
            filtered_data = [item for item in pairs]

            # Extract "mu" values for the filtered data
            filtered_mu_values = [item["mu"] for item in filtered_data]

            # Plot histogram of mu values
            plt.figure(figsize=(10, 6))
            plt.hist(filtered_mu_values, bins=50, alpha=0.75)
            plt.xlabel("Mu Values")
            plt.ylabel("Frequency")
            plt.title(f"Filtered {env_name}_{exp_name}_{pair_algo}")
            plt.grid(True)
            plt.savefig(
                get_pair_log_path(
                    env_name=env_name,
                    exp_name=exp_name,
                    pair_type="train",
                    pair_algo=pair_algo,
                    log_file="mu_histogram.png",
                )
            )
            plt.close()


def evaluate_pair(env_name, exp_name, pair_type, pair_algo):
    dataset = load_dataset(env_name)
    data = load_pair(env_name, exp_name, pair_type, pair_algo)

    answer_count = 0
    total_count = 0

    cumulative_rewards = np.cumsum(dataset["rewards"], dtype=np.float64)

    def segment_sum(start, end):
        return cumulative_rewards[end - 1] - (
            cumulative_rewards[start - 1] if start > 0 else 0
        )

    s0_starts = np.array([s0[0] for s0, _, _ in data])
    s0_ends = np.array([s0[1] for s0, _, _ in data])
    s1_starts = np.array([s1[0] for _, s1, _ in data])
    s1_ends = np.array([s1[1] for _, s1, _ in data])

    rewards_sum_0 = np.array(
        [segment_sum(start, end) for start, end in zip(s0_starts, s0_ends)]
    )
    rewards_sum_1 = np.array(
        [segment_sum(start, end) for start, end in zip(s1_starts, s1_ends)]
    )

    mu_values = np.where(rewards_sum_0 < rewards_sum_1, 1.0, 0.0)

    true_feedbacks = [(data[i][0], data[i][1], mu) for i, mu in enumerate(mu_values)]
    true_feedbacks = np.array(
        true_feedbacks,
        dtype=[
            ("s0", "i4", (2,)),
            ("s1", "i4", (2,)),
            ("mu", "f"),
        ],
    )

    answer_count = 0
    total_count = 0

    for i in range(len(true_feedbacks)):
        truth, mean = (true_feedbacks["mu"][i], data["mu"][i])

        if mean == 0.5:
            continue

        total_count += 1
        if truth == 0 and mean < 0.5:
            answer_count += 1
        elif truth == 1 and mean > 0.5:
            answer_count += 1

    log_path = "log/main_evaluate_pair.csv"
    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    accuracy = answer_count / total_count
    print(accuracy, answer_count, total_count)

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
