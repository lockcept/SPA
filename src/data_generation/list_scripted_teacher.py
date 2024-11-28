import numpy as np
import numpy.lib.recfunctions as rfn

import random

import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from data_loading.load_data import load_dataset


def generate_trajectories(dataset, trajectory_len, num_trajectories):
    trajectories = []

    dataset_length = len(dataset["observations"])
    is_terminal = dataset["terminals"] | dataset["timeouts"]

    count = 0
    while count < num_trajectories:
        start_idx = random.randint(0, dataset_length - trajectory_len)
        end_idx = start_idx + trajectory_len

        if any(is_terminal[start_idx:end_idx]):
            continue

        sum_of_rewards = np.sum(dataset["rewards"][start_idx:end_idx])
        trajectories.append((start_idx, end_idx, sum_of_rewards))
        count += 1

    return np.array(
        trajectories, dtype=[("start", int), ("end", int), ("sum_of_rewards", float)]
    )


def divide_into_groups(trajectories, num_group):
    sorted_trajectories = np.sort(trajectories, order="sum_of_rewards")

    group_size = len(sorted_trajectories) / num_group
    result = []

    for i in range(num_group):
        start_idx = int(np.floor(i * group_size))
        end_idx = (
            int(np.floor((i + 1) * group_size))
            if i < num_group - 1
            else len(sorted_trajectories)
        )
        for traj in sorted_trajectories[start_idx:end_idx]:
            result.append((*traj, i))

    return np.array(
        result,
        dtype=[
            ("start", int),
            ("end", int),
            ("sum_of_rewards", float),
            ("group_index", int),
        ],
    )


def generate_pairs(trajectories_with_groups, num_group):
    np.random.shuffle(trajectories_with_groups)
    pairs = []
    num_trajectories = len(trajectories_with_groups)

    for i in range(num_trajectories):
        for j in range(i + 1, num_trajectories):
            traj0 = trajectories_with_groups[i]
            traj1 = trajectories_with_groups[j]

            group_diff = (traj1["group_index"] - traj0["group_index"]) / (num_group - 1)
            mu = group_diff * 0.5 + 0.5

            pairs.append(
                ((traj0["start"], traj0["end"]), (traj1["start"], traj1["end"]), mu)
            )

    # Typed numpy array로 반환
    return np.array(pairs, dtype=[("s0", "i4", (2,)), ("s1", "i4", (2,)), ("mu", "f")])


def save_pairs(env_name, pair, pair_algo, pair_data):
    save_path = f"pair/{env_name}/{pair}_list-{pair_algo}.npz"
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.savez(save_path, data=pair_data)
    print(f"Preference pairs saved at {save_path}")


def generate_list_pairs(
    env_name,
    pair_name_base,
    num_trajectories,
    pair_algos=[2, 3, 5, 11],
):
    for pair_algo in pair_algos:
        save_path = f"pair/{env_name}/{pair_name_base}_list-{pair_algo}.npz"
        is_already_exist = os.path.exists(save_path)
        if is_already_exist:
            print(f"Pair already exists at {save_path}, cancel generating")
            return

    dataset = load_dataset(env_name=env_name)
    trajectory_len = 25

    trajectories = generate_trajectories(
        dataset=dataset,
        trajectory_len=trajectory_len,
        num_trajectories=num_trajectories,
    )

    # group num_pairs into M groups, with is defined by pair_algos
    for pair_algo in pair_algos:
        num_group = pair_algo

        trajectories_with_groups = divide_into_groups(trajectories, num_group)

        pairs = generate_pairs(
            trajectories_with_groups=trajectories_with_groups, num_group=num_group
        )

        save_pairs(
            env_name=env_name, pair=pair_name_base, pair_algo=pair_algo, pair_data=pairs
        )

    print("finish generating preference pairs", env_name, pair_name_base)
