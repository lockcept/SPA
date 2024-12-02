import json
import numpy as np
import matplotlib.pyplot as plt

from data_loading import load_dataset


def save_trajectory_lengths(dataset, env_name):
    terminals, timeouts = dataset["terminals"], dataset["timeouts"]
    indices = []
    length = len(terminals)
    start = 0
    for i in range(length):
        if terminals[i] or timeouts[i]:
            indices.append((start, i + 1))
            start = i + 1

    lengths = [end - start for start, end in indices]

    with open(f"dataset/{env_name}/trajectory_lengths.json", "w") as f:
        json.dump(lengths, f)

    print("Number of trajectories: ", len(lengths))
    print("Average length: ", np.mean(lengths))
    print("Median length: ", np.median(lengths))
    print("Max length: ", np.max(lengths))
    print("Min length: ", np.min(lengths))

    plt.hist(lengths, bins=50)
    plt.title("Histogram of trajectory lengths")
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.show()


def save_reward_graph(dataset_path, dataset_name, log_path=None):
    dataset_npz = np.load(dataset_path)
    dataset = {key: dataset_npz[key] for key in dataset_npz}

    log_path = f"log/dataset_reward_distribution_{dataset_name}.png"

    rewards = dataset["rewards"]
    plt.hist(rewards, bins=100)
    plt.title(f"Reward graph of {dataset_name}")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    if log_path is not None:
        plt.savefig(log_path, format="png")
    else:
        plt.show()


def analyze_env_dataset(env_name):
    """
    Analyze the raw dataset of the given environment.
    """
    data = load_dataset(env_name)
    print(data["observations"].shape)
    save_trajectory_lengths(data, env_name)
    save_reward_graph(data, env_name)
