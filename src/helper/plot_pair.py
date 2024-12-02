from matplotlib import pyplot as plt
import numpy as np
from data_loading import get_processed_data, load_dataset, load_pair


def plot_pair(env_name_list, pair_algo_list, pair_name_base):
    for env_name in env_name_list:
        for pair_algo in pair_algo_list:
            pair_name = f"{pair_name_base}_{pair_algo}"
            data = get_processed_data(env_name, pair_name)

            # histogram of mu values
            mu_values = [item["mu"] for item in data]
            plt.figure(figsize=(10, 6))
            plt.hist(mu_values, bins=50, alpha=0.75)
            plt.xlabel("Mu Values")
            plt.ylabel("Frequency")
            plt.title(f"{env_name}_{pair_name}")
            plt.grid(True)
            plt.savefig(f"log/mu_histogram_{env_name}_{pair_name}.png")


def evaluate_pair(env_name, pair_name):
    dataset = load_dataset(env_name)
    data = load_pair(env_name, pair_name)

    answer_count = 0
    for s0, s1, mu in data["data"]:
        rewards_sum_0 = np.sum(dataset["rewards"][s0[0] : s0[1]])
        rewards_sum_1 = np.sum(dataset["rewards"][s1[0] : s1[1]])

        if rewards_sum_0 < rewards_sum_1 and mu > 0.5:
            answer_count += 1
        elif rewards_sum_0 > rewards_sum_1 and mu < 0.5:
            answer_count += 1
    print(answer_count / len(data["data"]))
