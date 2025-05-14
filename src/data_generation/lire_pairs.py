import os
import numpy as np
import numpy.lib.recfunctions as rfn
import random
import itertools
from data_generation.utils import save_feedbacks_npz
from data_loading.load_data import load_pair
from utils.path import get_pair_path


def get_traj_reward_sum(start, end, cum_rewards):
    return cum_rewards[end - 1] - (cum_rewards[start - 1] if start > 0 else 0)


def binary_compare(traj0, traj1, cum_rewards, threshold=12.5):
    r0 = get_traj_reward_sum(traj0[0], traj0[1], cum_rewards)
    r1 = get_traj_reward_sum(traj1[0], traj1[1], cum_rewards)
    diff = r1 - r0
    if abs(diff) <= threshold:
        return 0.5
    return 1.0 if diff > 0 else 0.0


def insert_into_buckets(traj, buckets, cum_rewards):
    low, high = 0, len(buckets) - 1
    compare_count = 0

    while low <= high:
        mid = (low + high) // 2
        sample = random.choice(buckets[mid])
        mu = binary_compare(traj, sample, cum_rewards)
        compare_count += 1

        if mu == 0.5:
            buckets[mid].append(traj)
            return compare_count
        elif mu == 0.0:
            high = mid - 1
        else:
            low = mid + 1

    buckets.insert(low, [traj])
    return compare_count


def generate_and_save_lire_pairs(
    dataset, env_name, exp_name, pair_type, max_comparisons=500
):
    pair_name = f"lire-{max_comparisons}"
    save_path = get_pair_path(env_name, exp_name, pair_type, pair_name)

    # 이미 존재하는 경우 스킵
    if os.path.exists(save_path):
        print(f"Already exists: {save_path} — skipping generation.")
        return

    feedbacks = load_pair(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_algo="ternary-500",
    )

    trajectory_list = []
    for traj0, traj1, mu in feedbacks:
        trajectory_list.append(traj0)
        trajectory_list.append(traj1)

    cum_rewards = np.cumsum(dataset["rewards"], dtype=np.float64)
    buckets = []
    comparison_count = 0

    candidate_trajs = trajectory_list.copy()
    random.shuffle(candidate_trajs)

    while candidate_trajs and comparison_count < max_comparisons:
        traj = candidate_trajs.pop()
        comparison_count += insert_into_buckets(traj, buckets, cum_rewards)

    print(f"Generated {len(buckets)} buckets from ~{comparison_count} comparisons.")

    traj_with_bucket = []

    for bucket_idx, trajs in enumerate(buckets):
        for traj in trajs:
            traj_with_bucket.append((traj, bucket_idx))

    print(f"Generated {len(traj_with_bucket)} trajectories with buckets.")

    feedbacks = []
    for (traj0, b0), (traj1, b1) in itertools.combinations(traj_with_bucket, 2):
        if b0 == b1:
            mu = 0.5
        elif b0 < b1:
            mu = 1.0
        else:
            mu = 0.0
        feedbacks.append((traj0, traj1, mu))

    save_feedbacks_npz(
        env_name=env_name,
        exp_name=exp_name,
        pair_type=pair_type,
        pair_name=pair_name,
        feedbacks=feedbacks,
    )
