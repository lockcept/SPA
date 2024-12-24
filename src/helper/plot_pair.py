from matplotlib import pyplot as plt
import numpy as np
from data_loading import get_processed_data, load_dataset, load_pair
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
            mu_values = [item[2] for item in pairs]
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
    for s0, s1, mu in data["data"]:
        rewards_sum_0 = np.sum(dataset["rewards"][s0[0] : s0[1]])
        rewards_sum_1 = np.sum(dataset["rewards"][s1[0] : s1[1]])

        if rewards_sum_0 <= rewards_sum_1 and mu >= 0.5:
            answer_count += 1
        elif rewards_sum_0 >= rewards_sum_1 and mu <= 0.5:
            answer_count += 1
    print(answer_count / len(data["data"]))
