import numpy as np
import numpy.lib.recfunctions as rfn

from data_generation.utils import save_feedbacks_npz


def get_pairs_by_mu_type(mu_type, pair_data, average_reward):
    """
    Args:
        mu_type: str, type of mu
        pair_data: np.darray of,
            ("s0", "i4", (2,)),
            ("s1", "i4", (2,)),
            ("reward_sum_0", "f"),
            ("reward_sum_1", "f"),
        reward_info: tuple, reward min and max
    Returns:
        pair_data: np.darray of,
            ("s0", "i4", (2,)),
            ("s1", "i4", (2,)),
            ("mu", "f"),
    """
    reward_sum_0 = pair_data["reward_sum_0"]
    reward_sum_1 = pair_data["reward_sum_1"]

    # warning: calculate legnth_values only on the first pair (s0)
    length_values = pair_data["s0"][:, 1] - pair_data["s0"][:, 0]
    # mu_values = np.where(
    #     np.abs(reward_sum_0 - reward_sum_1) < average_reward * length_values * 0.1,
    #     0.5,
    #     np.where(reward_sum_0 > reward_sum_1, 0, 1),
    # )
    mu_values = np.where(
        np.abs(reward_sum_0 - reward_sum_1) < length_values * 0.5,
        0.5,
        np.where(reward_sum_0 > reward_sum_1, 0, 1),
    )

    if "flip" in mu_type:
        # 10% of the pairs are flipped
        flip_mask = np.random.choice([0, 1], len(pair_data), p=[0.9, 0.1])
        mu_values = np.where(flip_mask, 1 - mu_values, mu_values)
        pair_data = rfn.append_fields(pair_data, "mu", mu_values, dtypes=float)
    elif mu_type == "random":
        mu_values = np.random.uniform(0, 1, len(reward_sum_0))
        pair_data = rfn.append_fields(pair_data, "mu", mu_values, dtypes=float)
    else:
        pair_data = rfn.append_fields(pair_data, "mu", mu_values, dtypes=float)

    pair_data = rfn.drop_fields(pair_data, "reward_sum_0")
    pair_data = rfn.drop_fields(pair_data, "reward_sum_1")

    if "100000" in mu_type:
        pair_data = pair_data[:100000]
    elif "10000" in mu_type:
        pair_data = pair_data[:10000]
    elif "1000" in mu_type:
        pair_data = pair_data[:1000]
    elif "500" in mu_type:
        pair_data = pair_data[:500]
    elif "200" in mu_type:
        pair_data = pair_data[:200]
    elif "100" in mu_type:
        pair_data = pair_data[:100]
    else:
        pair_data = pair_data[:1000]

    return pair_data


def generate_and_save_ternary_pairs(
    dataset,
    env_name,
    exp_name,
    pair_type,
    pairs,
    mu_types,
):
    """
    Args:
        dataset,
        env_name: str,
        save_pair_dir: str,
        pairs: list of ((int, int), (int, int)),
        mu_types: list of str,
    """

    cumulative_rewards = np.cumsum(dataset["rewards"], dtype=np.float64)

    pairs_with_rewards = []

    for i0, i1 in pairs:
        s0, e0 = i0
        s1, e1 = i1
        len0 = e0 - s0
        len1 = e1 - s1
        if len0 > len1:
            e0 = s0 + len1
        elif len0 < len1:
            e1 = s1 + len0

        reward_sum_0 = cumulative_rewards[e0 - 1] - (
            cumulative_rewards[s0 - 1] if s0 > 0 else 0
        )
        reward_sum_1 = cumulative_rewards[e1 - 1] - (
            cumulative_rewards[s1 - 1] if s1 > 0 else 0
        )
        pairs_with_rewards.append(((s0, e0), (s1, e1), reward_sum_0, reward_sum_1))

    pairs_with_rewards_np = np.array(
        pairs_with_rewards,
        dtype=[
            ("s0", "i4", (2,)),
            ("s1", "i4", (2,)),
            ("reward_sum_0", "f"),
            ("reward_sum_1", "f"),
        ],
    )

    average_reward = np.mean(dataset["rewards"])

    for mu_type in mu_types:
        feedbacks = get_pairs_by_mu_type(
            mu_type=mu_type,
            pair_data=pairs_with_rewards_np,
            average_reward=average_reward,
        )

        save_feedbacks_npz(
            env_name=env_name,
            exp_name=exp_name,
            pair_type=pair_type,
            pair_name=f"ternary-{mu_type}",
            feedbacks=feedbacks,
        )
