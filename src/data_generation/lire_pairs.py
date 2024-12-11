from typing import Literal
import numpy as np

from utils import get_pair_path


def generate_and_save_lire_pairs(
    dataset,
    env_name,
    exp_name,
    pair_type,
    pair_algo: Literal["with-0.5", "without-0.5"],
    threshold,
    pairs,
):
    """
    Args:
        dataset,
        env_name: str,
        exp_name: str,
        pair_type: str,
        cut_type: str,
        threshold: float,
        pairs: list of ((int, int), (int, int)),
    """

    lire_pairs = []

    for i0, i1 in pairs:
        sum_of_rewards_0 = np.sum(dataset["rewards"][i0[0] : i0[1]])
        sum_of_rewards_1 = np.sum(dataset["rewards"][i1[0] : i1[1]])
        if np.abs(sum_of_rewards_0 - sum_of_rewards_1) < threshold:
            mu = 0.5
        elif sum_of_rewards_0 < sum_of_rewards_1:
            mu = 1.0
        else:
            mu = 0.0

        if mu == 0.5 and pair_algo == "without-0.5":
            continue

        lire_pairs.append(((i0[0], i0[1]), (i1[0], i1[1]), mu))

    pairs_np = np.array(
        lire_pairs, dtype=[("s0", "i4", (2,)), ("s1", "i4", (2,)), ("mu", "f")]
    )

    pair_path = get_pair_path(
        env_name=env_name,
        exp_name=exp_name,
        pair_type=pair_type,
        pair_algo=f"lire-{pair_algo}",
    )
    np.savez(pair_path, data=pairs_np)
