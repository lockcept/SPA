import argparse
import numpy as np
import random


def load_dataset(file_path):
    dataset = np.load(file_path)
    return dataset


def extract_trajectory_indices(dataset):
    terminals, timeouts = dataset["terminals"], dataset["timeouts"]
    indices = []
    length = len(terminals)
    start = 0
    for i in range(length):
        if terminals[i] or timeouts[i]:
            indices.append((start, i + 1))
            start = i + 1
    return indices


def trajectory_from_index(dataset, start, end):
    trajectory = {
        "observations": dataset["observations"][start:end],
        "actions": dataset["actions"][start:end],
        "rewards": dataset["rewards"][start:end],
    }
    return trajectory


def compare_trajectories(traj0, traj1):
    reward_sum_0 = np.sum(traj0["rewards"])
    reward_sum_1 = np.sum(traj1["rewards"])

    if reward_sum_0 > reward_sum_1:
        return 0
    else:
        return 1


def generate_preference_pair(dataset, indices):
    min_length = 10

    while True:
        index0, index1 = random.sample(range(len(indices)), 2)
        (start0, end0), (start1, end1) = indices[index0], indices[index1]

        length0 = end0 - start0
        length1 = end1 - start1

        if length0 < min_length or length1 < min_length:
            continue

        if length0 > length1:
            end0 = start0 + length1
        else:
            end1 = start1 + length0

        traj0 = trajectory_from_index(dataset, start0, end0)
        traj1 = trajectory_from_index(dataset, start1, end1)

        winner = compare_trajectories(traj0, traj1)

        preference_pair = np.array(
            ((start0, end0), (start1, end1), winner),
            dtype=[("s0", "i4", (2,)), ("s1", "i4", (2,)), ("mu", "i4")],
        )

        return preference_pair


def save_preference_pairs(file_path, preference_pairs):
    np.savez(file_path, data=preference_pairs)
    print(f"Preference pairs saved at {file_path}")


def generate_and_save(env, num_pairs=1000):
    dataset_file_path = f"dataset/{env}/d4rl_dataset.npz"
    dataset = load_dataset(dataset_file_path)

    indices = extract_trajectory_indices(dataset)

    preference_pairs = []
    for _ in range(num_pairs):
        preference_pair = generate_preference_pair(dataset, indices)
        preference_pairs.append(preference_pair)

    save_path = f"dataset/{env}/full_preference_pairs.npz"
    save_preference_pairs(save_path, preference_pairs)
