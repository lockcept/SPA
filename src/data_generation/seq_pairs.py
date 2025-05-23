import os
import numpy as np
import random
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


def generate_and_save_seq_pairs(
    dataset, env_name, exp_name, pair_type, max_compares=500, threshold=12.5
):
    pair_name = f"seq-{max_compares}"
    save_path = get_pair_path(env_name, exp_name, pair_type, pair_name)

    if os.path.exists(save_path):
        print(f"Already exists: {save_path} — skipping generation.")
        # return

    feedbacks = load_pair(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_algo="ternary-500",
    )

    trajectory_list = []
    for traj0, traj1, _ in feedbacks:
        trajectory_list.append(traj0)
        trajectory_list.append(traj1)

    cum_rewards = np.cumsum(dataset["rewards"], dtype=np.float64)
    random.shuffle(trajectory_list)

    compares = 0
    seq_feedbacks = []

    root = trajectory_list.pop()
    node_list = []
    print(root)

    while trajectory_list and compares < max_compares:
        challenger = trajectory_list.pop()
        mu = binary_compare(root, challenger, cum_rewards, threshold=threshold)
        compares += 1

        if mu == 1.0:
            node_list.append(root)
            for node in node_list:
                seq_feedbacks.append((node, challenger, 1.0))
            print(f"feedback appended {len(node_list)}")
            root = challenger
        elif mu == 0.0:
            seq_feedbacks.append((root, challenger, 0.0))
            node_list.append(challenger)
        else:
            for node in node_list:
                seq_feedbacks.append((node, challenger, 1.0))
            seq_feedbacks.append((root, challenger, 0.5))
            node_list.append(root)
            root = challenger

    print(f"Generated {len(seq_feedbacks)} feedbacks using root pairwise strategy.")

    save_feedbacks_npz(
        env_name=env_name,
        exp_name=exp_name,
        pair_type=pair_type,
        pair_name=pair_name,
        feedbacks=seq_feedbacks,
    )
