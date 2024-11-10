import json
import numpy as np
import matplotlib.pyplot as plt

import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from data_loading import load_dataset


def load_d4rl(env_name):
    return load_dataset.load_d4rl_dataset(env_name)


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


def show_reward_graph(dataset, env_name):
    rewards = dataset["rewards"]
    plt.hist(rewards, bins=50)
    plt.title(f"Reward graph of {env_name}")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.show()


def analyze(env_name):
    data = load_d4rl(env_name)
    print(data["observations"].shape)
    save_trajectory_lengths(data, env_name)
    show_reward_graph(data, env_name)
