from typing import Literal
import numpy as np

from utils import get_pair_path


def generate_and_save_cut_pairs(
    dataset,
    env_name,
    exp_name,
    pair_type,
    cut_type: Literal["0.5", "0.25", "half-random", "random"],
    pairs,
):
    """
    Args:
        dataset,
        env_name: str,
        exp_name: str,
        pair_type: str,
        cut_type: str,
        pairs: list of ((int, int), (int, int)),
    """

    length = len(pairs)
    cut_pairs = []
    pair_index = 0

    valid_feedback = 0

    mu = 0.75

    while valid_feedback < length:
        i0, i1 = pairs[pair_index]

        s0, e0 = i0
        s1, e1 = i1

        if cut_type == "0.5":
            ratio = 0.5
        elif cut_type == "0.25":
            ratio = 0.25
        elif cut_type == "half-random":
            ratio = np.random.uniform(0.25, 0.75)
        elif cut_type == "random":
            ratio = np.random.uniform(0.1, 0.9)
        else:
            raise ValueError(f"Invalid cut type: {cut_type}")

        m0 = s0 + int((e0 - s0) * ratio)
        m1 = s1 + int((e1 - s1) * ratio)

        i0_0 = (s0, m0)
        i0_1 = (m0, e0)
        i1_0 = (s1, m1)
        i1_1 = (m1, e1)

        r0_0 = np.sum(dataset["rewards"][s0:m0])
        r0_1 = np.sum(dataset["rewards"][m0:e0])
        r1_0 = np.sum(dataset["rewards"][s1:m1])
        r1_1 = np.sum(dataset["rewards"][m1:e1])

        r0 = r0_0 + r0_1
        r1 = r1_0 + r1_1

        if r0 < r1:
            if r0_0 < r1_0:
                if r0_1 < r1_1:
                    cut_pairs.append((i0, i1, 1.0))
                    cut_pairs.append((i0_0, i1_0, mu))
                    cut_pairs.append((i0_1, i1_1, mu))
                    valid_feedback += 3
                else:
                    cut_pairs.append((i0, i1, mu))
                    cut_pairs.append((i0_0, i1_0, 1.0))
                    cut_pairs.append((i0_1, i1_1, 1 - mu))
                    valid_feedback += 3
            else:
                cut_pairs.append((i0, i1, mu))
                cut_pairs.append((i0_0, i1_0, 1 - mu))
                cut_pairs.append((i0_1, i1_1, 1.0))
                valid_feedback += 2
        else:
            if r0_0 < r1_0:
                cut_pairs.append((i0, i1, 1 - mu))
                cut_pairs.append((i0_0, i1_0, mu))
                cut_pairs.append((i0_1, i1_1, 0))
                valid_feedback += 2
            else:
                if r0_1 < r1_1:
                    cut_pairs.append((i0, i1, 1 - mu))
                    cut_pairs.append((i0_0, i1_0, 0))
                    cut_pairs.append((i0_1, i1_1, mu))
                    valid_feedback += 3
                else:
                    cut_pairs.append((i0, i1, 0))
                    cut_pairs.append((i0_0, i1_0, 1 - mu))
                    cut_pairs.append((i0_1, i1_1, 1 - mu))
                    valid_feedback += 3
        pair_index += 1
    print(length, valid_feedback, pair_index)

    pairs_np = np.array(
        cut_pairs, dtype=[("s0", "i4", (2,)), ("s1", "i4", (2,)), ("mu", "f")]
    )

    pair_path = get_pair_path(
        env_name=env_name,
        exp_name=exp_name,
        pair_type=pair_type,
        pair_algo=f"cut-{cut_type}",
    )
    np.savez(pair_path, data=pairs_np)
