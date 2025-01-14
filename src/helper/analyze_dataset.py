import json
import numpy as np
import matplotlib.pyplot as plt

from data_loading import load_dataset
from utils import get_new_dataset_path, get_new_dataset_log_path


def save_trajectory_lengths(dataset, env_name):
    terminals, timeouts = dataset["terminals"], dataset["timeouts"]
    indices = []
    length = len(terminals)
    start = 0
    success_count = 0
    for i in range(length):
        if terminals[i] or timeouts[i]:
            if "success" in dataset:
                success = np.sum(dataset["success"][start : i + 1])
                if success > 0:
                    success_count += 1
            indices.append((start, i + 1))
            start = i + 1

    lengths = [end - start for start, end in indices]
    print(success_count, len(indices))

    with open(
        f"dataset/{env_name}/trajectory_lengths.json", "w", encoding="utf-8"
    ) as f:
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
    plt.savefig(f"log/dataset_trajectory_length_{env_name}.png")


def save_reward_graph_from_dataset(dataset, log_path, title):
    """
    Save the reward graph from the given dataset.
    """
    rewards = dataset["rewards"]
    plt.hist(rewards)
    plt.title(f"Reward graph of {title}")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.savefig(log_path, format="png")
    plt.close()


def save_reward_scatter_from_raw_dataset(dataset, raw_dataset, log_path, title):
    """
    Save the reward scatter plot from the given dataset and raw dataset.
    """
    raw_rewards = raw_dataset["rewards"]
    rewards = dataset["rewards"]

    plt.scatter(raw_rewards, rewards)
    plt.title(f"Reward scatter plot of {title}")
    plt.xlabel("Raw reward")
    plt.ylabel("Reward")
    plt.savefig(log_path, format="png")
    plt.close()


def save_reward_graph(
    env_name,
    exp_name,
    pair_algo,
    reward_model_algo,
):
    """
    Save the reward graph of the given environment, experiment, pair algorithm, and reward model algorithm.
    """
    dataset_path = get_new_dataset_path(
        env_name, exp_name, pair_algo, reward_model_algo
    )
    dataset_npz = np.load(dataset_path)
    dataset = {key: dataset_npz[key] for key in dataset_npz}

    raw_dataset = load_dataset(env_name)

    title = f"{env_name} {exp_name} {pair_algo} {reward_model_algo}"

    save_reward_graph_from_dataset(
        dataset,
        get_new_dataset_log_path(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo=pair_algo,
            reward_model_algo=reward_model_algo,
            log_file="reward_distribution.png",
        ),
        title,
    )
    save_reward_scatter_from_raw_dataset(
        dataset,
        raw_dataset,
        get_new_dataset_log_path(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo=pair_algo,
            reward_model_algo=reward_model_algo,
            log_file="reward_scatter.png",
        ),
        title,
    )


def analyze_env_dataset(env_name):
    """
    Analyze the raw dataset of the given environment.
    """
    dataset = load_dataset(env_name)
    print(dataset["observations"].shape)
    save_reward_graph_from_dataset(dataset, f"log/{env_name}.png", env_name)
    save_trajectory_lengths(dataset, env_name)
