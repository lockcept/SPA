import numpy as np
import numpy.lib.recfunctions as rfn

import random


import os
import sys

from tqdm import tqdm


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from data_loading.load_dataset import load_d4rl_dataset


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
        reward_sum_0 = np.sum(traj0["rewards"])
        reward_sum_1 = np.sum(traj1["rewards"])
        mu = reward_sum_0 < reward_sum_1
        avg_diff = (reward_sum_1 - reward_sum_0) / min(length0, length1)

        preference_pair = ((start0, end0), (start1, end1), mu, avg_diff)

        return preference_pair


def save_preference_pairs(file_path, preference_pairs):
    np.savez(file_path, data=preference_pairs)
    print(f"Preference pairs saved at {file_path}")


def generate_and_save(env, path, num_pairs=10):
    dataset = load_d4rl_dataset(env)

    indices = extract_trajectory_indices(dataset)
    print("trajectory counts", len(indices))

    preference_pairs = []
    for _ in tqdm(range(num_pairs), desc="Generating preference pairs"):
        preference_pair = generate_preference_pair(dataset, indices)
        preference_pairs.append(preference_pair)

    preference_pairs_np = np.array(
        preference_pairs,
        dtype=[
            ("s0", "i4", (2,)),
            ("s1", "i4", (2,)),
            ("mu", "i4"),
            ("avg_diff", "f4"),
        ],
    )

    print(preference_pairs_np[:5])

    max_abs_diff = np.max(np.abs(preference_pairs_np["avg_diff"]))
    normalized_diff = preference_pairs_np["avg_diff"] / max_abs_diff
    mu_values = 0.5 + 0.5 * normalized_diff
    preference_pairs_np = rfn.append_fields(
        preference_pairs_np, "normalized_mu", mu_values, dtypes=float
    )
    preference_pairs_np = rfn.drop_fields(preference_pairs_np, "avg_diff")

    print(preference_pairs_np[:5])

    save_path = f"dataset/{env}/{path}.npz"
    save_preference_pairs(save_path, preference_pairs_np)
