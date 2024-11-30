import os

import numpy as np
import numpy.lib.recfunctions as rfn


def get_pairs_by_mu_type(mu_type, pair_data, reward_info=(0, 1)):
    """
    Args:
        mu_type: str, type of mu
        pair_data: np.darray of,
            ("s0", "i4", (2,)),
            ("s1", "i4", (2,)),
            ("rewards_0", "O"),
            ("rewards_1", "O"),
        reward_info: tuple, reward min and max
    Returns:
        pair_data: np.darray of,
            ("s0", "i4", (2,)),
            ("s1", "i4", (2,)),
            ("mu", "f"),
    """
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


def generate_and_save_full_pairs(
    dataset, env_name, pair_name_base, pairs, mu_types=None
):
    """
    Args:
        dataset,
        env_name: str,
        pair_name_base: str,
        pairs: list of ((int, int), (int, int)),
        mu_types: list of str,
    """

    reward_min = np.min(dataset["rewards"])
    reward_max = np.max(dataset["rewards"])
    reward_info = (reward_min, reward_max)

    preference_pairs = []

    for i0, i1 in pairs:
        s0, e0 = i0
        s1, e1 = i1
        len0 = e0 - s0
        len1 = e1 - s1
        if len0 > len1:
            e0 = s0 + len1
        elif len0 < len1:
            e1 = s1 + len0

        rewards_0 = dataset["rewards"][s0:e0]
        rewards_1 = dataset["rewards"][s1:e1]
        preference_pairs.append((s0, s1, rewards_0, rewards_1))

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
