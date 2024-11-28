import numpy as np
import numpy.lib.recfunctions as rfn

import random

import os
import sys

from tqdm import tqdm


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from data_generation.utils import extract_trajectory_indices
from data_loading.load_data import load_dataset


def rewards_from_index(dataset, start, end):
    return dataset["rewards"][start:end]


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

        rewards_0 = rewards_from_index(dataset, start0, end0)
        rewards_1 = rewards_from_index(dataset, start1, end1)

        preference_pair = ((start0, end0), (start1, end1), rewards_0, rewards_1)

        return preference_pair


def get_pairs_by_mu_type(mu_type, pair_data, reward_info=(0, 1)):
    reward_min, reward_max = reward_info
    rewards_0 = pair_data["rewards_0"]
    rewards_1 = pair_data["rewards_1"]
    reward_sum_0 = np.array([np.sum(rewards) for rewards in rewards_0])
    reward_sum_1 = np.array([np.sum(rewards) for rewards in rewards_1])
    normalized_rewards_0 = np.array(
        [(rewards - reward_min) / (reward_max - reward_min) for rewards in rewards_0]
    )
    normalized_rewards_1 = np.array(
        [(rewards - reward_min) / (reward_max - reward_min) for rewards in rewards_1]
    )
    normalized_reward_sum_0 = np.array(
        [np.sum(rewards) + np.finfo(float).eps for rewards in normalized_rewards_0]
    )
    normalized_reward_sum_1 = np.array(
        [np.sum(rewards) + np.finfo(float).eps for rewards in normalized_rewards_1]
    )

    if mu_type == "binary":
        mu_values = np.where(reward_sum_0 > reward_sum_1, 0, 1)
        pair_data = rfn.append_fields(pair_data, "mu", mu_values, dtypes=float)
    elif mu_type == "minus-binary":
        mu_values = np.where(reward_sum_0 < reward_sum_1, 0, 1)
        pair_data = rfn.append_fields(pair_data, "mu", mu_values, dtypes=float)
    elif mu_type == "random":
        mu_values = np.random.uniform(0, 1, len(reward_sum_0))
        pair_data = rfn.append_fields(pair_data, "mu", mu_values, dtypes=float)
    elif mu_type == "continuous":
        length_values = pair_data["s0"][:, 1] - pair_data["s0"][:, 0]
        diff = (reward_sum_1 - reward_sum_0) / length_values
        max_diff = np.max(np.abs(diff))
        normalized_diff = diff / max_diff
        mu_values = 0.5 + 0.5 * normalized_diff
        pair_data = rfn.append_fields(pair_data, "mu", mu_values, dtypes=float)
    elif mu_type == "sigmoid":
        diff_values = reward_sum_1 - reward_sum_0
        sigmoid_values = 1 / (1 + np.exp(-diff_values))
        pair_data = rfn.append_fields(pair_data, "mu", sigmoid_values, dtypes=float)
    elif mu_type == "sigmoid-0.1":
        diff_values = reward_sum_1 - reward_sum_0
        sigmoid_values = 1 / (1 + np.exp(-diff_values))
        round_unit = 0.1
        rounded_sigmoid_values = np.round(sigmoid_values / round_unit) * round_unit
        pair_data = rfn.append_fields(
            pair_data, "mu", rounded_sigmoid_values, dtypes=float
        )
    elif mu_type == "sigmoid-0.25":
        diff_values = reward_sum_1 - reward_sum_0
        sigmoid_values = 1 / (1 + np.exp(-diff_values))
        round_unit = 0.25
        rounded_sigmoid_values = np.round(sigmoid_values / round_unit) * round_unit
        pair_data = rfn.append_fields(
            pair_data, "mu", rounded_sigmoid_values, dtypes=float
        )
    elif mu_type == "sigmoid-0.5":
        diff_values = reward_sum_1 - reward_sum_0
        sigmoid_values = 1 / (1 + np.exp(-diff_values))
        round_unit = 0.5
        rounded_sigmoid_values = np.round(sigmoid_values / round_unit) * round_unit
        pair_data = rfn.append_fields(
            pair_data, "mu", rounded_sigmoid_values, dtypes=float
        )
    elif mu_type == "linear":
        mu_values = normalized_reward_sum_1 / (
            normalized_reward_sum_0 + normalized_reward_sum_1
        )
        pair_data = rfn.append_fields(pair_data, "mu", mu_values, dtypes=float)

    pair_data = rfn.drop_fields(pair_data, "rewards_0")
    pair_data = rfn.drop_fields(pair_data, "rewards_1")

    return pair_data


def generate_full_pairs(env_name, pair_name_base, num_pairs, mu_types=["binary"]):
    for mu_type in mu_types:
        save_path = f"pair/{env_name}/{pair_name_base}_full-{mu_type}.npz"
        is_already_exist = os.path.exists(save_path)
        if is_already_exist:
            print(f"Pair already exists at {save_path}, cancel generating")
            return

    dataset = load_dataset(env_name=env_name)

    reward_min = np.min(dataset["rewards"])
    reward_max = np.max(dataset["rewards"])
    reward_info = (reward_min, reward_max)

    print("start generating preference pairs", env_name, pair_name_base, num_pairs)

    indices = extract_trajectory_indices(dataset)

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
            ("rewards_0", "O"),
            ("rewards_1", "O"),
        ],
    )

    for mu_type in mu_types:
        pair_data = get_pairs_by_mu_type(
            env_name=env_name,
            pair=pair_name_base,
            mu_type=mu_type,
            pair_data=preference_pairs_np,
            reward_info=reward_info,
        )

        save_path = f"pair/{env_name}/{pair_name_base}_full-{mu_type}.npz"
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.savez(save_path, data=pair_data)
        print(f"Preference pairs saved at {save_path}")
