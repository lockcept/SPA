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

        preference_pair = ((start0, end0), (start1, end1), reward_sum_0, reward_sum_1)

        return preference_pair


def save_pairs_by_mu_type(env, pair, mu_type, pair_data):
    if mu_type == "binary":
        mu_values = np.where(
            pair_data["reward_sum_0"] > pair_data["reward_sum_1"], 0, 1
        )
        pair_data = rfn.append_fields(pair_data, "mu", mu_values, dtypes=float)
    elif mu_type == "continuous":
        length_values = pair_data["s0"][:, 1] - pair_data["s0"][:, 0]
        diff = (pair_data["reward_sum_1"] - pair_data["reward_sum_0"]) / length_values
        max_diff = np.max(np.abs(diff))
        normalized_diff = diff / max_diff
        mu_values = 0.5 + 0.5 * normalized_diff
        pair_data = rfn.append_fields(pair_data, "mu", mu_values, dtypes=float)
    elif mu_type == "sigmoid":
        diff_values = pair_data["reward_sum_1"] - pair_data["reward_sum_0"]
        sigmoid_values = 1 / (1 + np.exp(-diff_values))
        pair_data = rfn.append_fields(pair_data, "mu", sigmoid_values, dtypes=float)

    pair_data = rfn.drop_fields(pair_data, "reward_sum_0")
    pair_data = rfn.drop_fields(pair_data, "reward_sum_1")

    save_path = f"pair/{env}/{pair}_{mu_type}.npz"
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.savez(save_path, data=pair_data)
    print(f"Preference pairs saved at {save_path}")


def generate_pairs(env, pair_name_base, num_pairs, mu_types=["binary"]):
    dataset = load_d4rl_dataset(env)

    print("start generating preference pairs", env, pair_name_base, num_pairs)

    indices = extract_trajectory_indices(dataset)
    print("trajectory counts", len(indices))

    preference_pairs = []
    for _ in tqdm(range(num_pairs), desc="Generating preference pairs"):
        preference_pair = generate_preference_pair(dataset, indices)
        preference_pairs.append(preference_pair)

    print("generating finished, start saving by mu type")

    preference_pairs_np = np.array(
        preference_pairs,
        dtype=[
            ("s0", "i4", (2,)),
            ("s1", "i4", (2,)),
            ("reward_sum_0", "f4"),
            ("reward_sum_1", "f4"),
        ],
    )

    for mu_type in mu_types:
        save_pairs_by_mu_type(env, pair_name_base, mu_type, preference_pairs_np)
