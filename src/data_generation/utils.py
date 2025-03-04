import os
import numpy as np

from utils import get_pair_path


def extract_trajectory_indices(dataset):
    terminals, timeouts = dataset["terminals"], dataset["timeouts"]
    indices = []
    length = len(terminals)
    start = 0
    for i in range(length):
        if terminals[i] or timeouts[i]:
            indices.append((start, i + 1))
            start = i + 1

    print("trajectory counts", len(indices))
    return indices


def generate_pairs_from_using_all(trajectories):
    """
    generate pairs using all trajectories
    """

    pairs = []
    total_trajectory_count = len(trajectories)

    for i in range(total_trajectory_count // 2):
        pairs.append((trajectories[2 * i], trajectories[2 * i + 1]))

    return pairs


def generate_pairs_from_indices(
    dataset, trajectories, pair_count, trajectory_length, except_same=False
):
    """
    choose pairs from indices and cut them to have a fixed length with same starting point
    To utilize as many pairs as possible, start by using pairs from the beginning.
    """

    pairs = []
    valid_trajectories = [t for t in trajectories if (t[1] - t[0]) >= trajectory_length]
    total_trajectory_count = len(valid_trajectories)

    i = 0

    while len(pairs) < pair_count:
        if i * 2 < total_trajectory_count:
            if i * 2 + 1 < total_trajectory_count:
                first_pair_index = i * 2
                second_pair_index = i * 2 + 1
            else:
                first_pair_index = i * 2
                second_pair_index = np.random.randint(0, total_trajectory_count - 1)
            i += 1
        else:
            first_pair_index, second_pair_index = np.random.randint(
                0, total_trajectory_count - 1, 2
            )

        first_trajectory = valid_trajectories[first_pair_index]
        second_trajectory = valid_trajectories[second_pair_index]
        min_length = min(
            (
                first_trajectory[1] - first_trajectory[0],
                second_trajectory[1] - second_trajectory[0],
            )
        )

        if min_length == trajectory_length:
            first_start_point = 0
            second_start_point = 0
        else:
            first_start_point = np.random.randint(0, min_length - trajectory_length)
            second_start_point = np.random.randint(0, min_length - trajectory_length)

        first_pair = (
            first_trajectory[0] + first_start_point,
            first_trajectory[0] + first_start_point + trajectory_length,
        )
        second_pair = (
            second_trajectory[0] + second_start_point,
            second_trajectory[0] + second_start_point + trajectory_length,
        )

        if except_same:
            first_rewards = dataset["rewards"][first_pair[0] : first_pair[1]]
            second_rewards = dataset["rewards"][second_pair[0] : second_pair[1]]
            if np.sum(first_rewards) == np.sum(second_rewards):
                continue

        pairs.append((first_pair, second_pair))

    return pairs


def save_feedbacks_npz(env_name, exp_name, pair_type, pair_name, feedbacks):
    """
    Save preference feedbacks using np.savez

    Args:
        env_name: str, Environment name
        exp_name: str, Experiment name
        pair_type: str, Type of pairs
        pair_name: str, Name of pairs
        feedbacks: list of ((int, int), (int, int))
    """
    save_path = get_pair_path(env_name, exp_name, pair_type, pair_name)
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)

    feedbacks_np = np.array(
        feedbacks, dtype=[("s0", "i4", (2,)), ("s1", "i4", (2,)), ("mu", "f")]
    )

    np.savez(save_path, data=feedbacks_np)

    print(f"Saved feedbacks at {save_path}")
