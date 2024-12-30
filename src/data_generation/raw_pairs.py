import numpy as np

from utils import get_pair_path


def save_raw_pairs(
    env_name,
    exp_name,
    pair_type,
    pairs,
):
    """
    Args:
        dataset,
        env_name: str,
        exp_name: str,
        pair_type: str,
        pairs: list of ((int, int), (int, int)),
    """

    raw_pairs = []

    for i0, i1 in pairs:
        s0, e0 = i0
        s1, e1 = i1
        raw_pairs.append(((s0, e0), (s1, e1), 0))

    pairs_np = np.array(
        raw_pairs, dtype=[("s0", "i4", (2,)), ("s1", "i4", (2,)), ("mu", "f")]
    )

    pair_path = get_pair_path(
        env_name=env_name,
        exp_name=exp_name,
        pair_type=pair_type,
        pair_algo="raw",
    )
    np.savez(pair_path, data=pairs_np)
    print(f"Preference pairs saved at {pair_path}")
