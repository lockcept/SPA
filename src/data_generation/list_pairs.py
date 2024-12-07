import numpy as np

from utils import get_pair_path


def divide_into_groups(trajectories, num_group):
    """
    return np.array of
        ("start", "i4"),
        ("end", "i4"),
        ("sum_of_rewards", "f"),
        ("group_index", "i4"),
    """
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


def generate_pairs(trajectory_pairs, trajectories_with_groups, num_group):
    """
    Args:
        trajectory_pairs: list of ((int, int), (int, int)),
        trajectories_with_groups: np.array of
            ("start", "i4"),
            ("end", "i4"),
            ("sum_of_rewards", "f"),
            ("group_index", "i4"),
        num_group: int,

    Returns:
        pairs: np.array of
            ("s0", "i4", (2,)),
            ("s1", "i4", (2,)),
            ("mu", "f"),
    """
    pairs = []

    for i0, i1 in trajectory_pairs:
        # search trajectory_with_groups with start=s0, end=e0
        s0, e0 = i0
        traj0 = trajectories_with_groups[
            (trajectories_with_groups["start"] == s0)
            & (trajectories_with_groups["end"] == e0)
        ][0]
        traj1 = trajectories_with_groups[
            (trajectories_with_groups["start"] == i1[0])
            & (trajectories_with_groups["end"] == i1[1])
        ][0]

        group_diff = (traj1["group_index"] - traj0["group_index"]) / (num_group - 1)
        mu = group_diff * 0.5 + 0.5

        pairs.append(
            ((traj0["start"], traj0["end"]), (traj1["start"], traj1["end"]), mu)
        )

    # Typed numpy array로 반환
    return np.array(pairs, dtype=[("s0", "i4", (2,)), ("s1", "i4", (2,)), ("mu", "f")])


def generate_and_save_list_pairs(
    dataset,
    env_name,
    exp_name,
    pair_type,
    pairs,
    all_indices,
    num_groups=None,
):
    """
    Args:
        dataset,
        env_name: str,
        pair_name_base: str,
        pairs: list of ((int, int), (int, int)),
        all_indices: list of (int, int, float),
        group_nums: list of int,
    """
    # make same length of all trajectories
    min_length = np.min([e - s for s, e in all_indices])
    new_index_pairs = []
    new_all_indices = []

    for i0, i1 in pairs:
        s0, e0 = i0
        s1, e1 = i1

        e0 = s0 + min_length
        e1 = s1 + min_length

        new_index_pairs.append(((s0, e0), (s1, e1)))

    for s, e in all_indices:
        e = s + min_length
        sum_of_rewards = np.sum(dataset["rewards"][s:e])
        new_all_indices.append((s, e, sum_of_rewards))

    new_all_indices = np.array(
        new_all_indices,
        dtype=[("start", int), ("end", int), ("sum_of_rewards", float)],
    )

    # group num_pairs into M groups, with is defined by pair_algos
    for num_group in num_groups:
        trajectories_with_groups = divide_into_groups(new_all_indices, num_group)

        pairs = generate_pairs(
            trajectory_pairs=new_index_pairs,
            trajectories_with_groups=trajectories_with_groups,
            num_group=num_group,
        )

        save_path = get_pair_path(
            env_name=env_name,
            exp_name=exp_name,
            pair_type=pair_type,
            pair_algo=f"list-{num_group}",
        )

        np.savez(save_path, data=pairs)
        print("finish saving preference pair", save_path)
