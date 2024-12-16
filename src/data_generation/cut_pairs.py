from typing import Literal
import numpy as np

from utils import get_pair_path


def generate_and_save_cut_pairs(
    dataset,
    env_name,
    exp_name,
    pair_type,
    cut_type: Literal["0.5", "0.25", "half-random", "random"],
    mu_scale,
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

    mu = mu_scale

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

        i0_head = (s0, m0)
        i0_tail = (m0, e0)
        i1_head = (s1, m1)
        i1_tail = (m1, e1)

        r0_head = np.sum(dataset["rewards"][s0:m0])
        r0_tail = np.sum(dataset["rewards"][m0:e0])
        r1_head = np.sum(dataset["rewards"][s1:m1])
        r1_tail = np.sum(dataset["rewards"][m1:e1])

        r0 = r0_head + r0_tail
        r1 = r1_head + r1_tail

        is_total_better = r1 > r0
        is_head_better = r1_head > r0_head
        is_tail_better = r1_tail > r0_tail

        # if total_better and head_better is different, valid feedback is 3
        if is_total_better == is_head_better:
            valid_feedback += 3
        else:
            valid_feedback += 2

        # append basic pairs
        cut_pairs.append((i0, i1, 1.0 if is_total_better else 0.0))
        cut_pairs.append((i0_head, i1_head, 1.0 if is_head_better else 0.0))
        cut_pairs.append((i0_tail, i1_tail, 1.0 if is_tail_better else 0.0))

        if (is_total_better == is_head_better) and (is_total_better == is_tail_better):
            # 0, 0, 0 or 1, 1, 1
            cut_pairs.append((i0, i0_head, 1 - mu if is_total_better else mu))
            cut_pairs.append((i0, i0_tail, 1 - mu if is_total_better else mu))
            cut_pairs.append((i1, i1_head, mu if is_total_better else 1 - mu))
            cut_pairs.append((i1, i1_tail, mu if is_total_better else 1 - mu))
        else:
            if is_total_better == is_head_better:
                # 0, 0, 1 or 1, 1, 0
                i0_same = i0_head
                i0_diff = i0_tail
                i1_same = i1_head
                i1_diff = i1_tail
            elif is_total_better == is_tail_better:
                # 0, 1, 0 or 1, 0, 1
                i0_same = i0_tail
                i0_diff = i0_head
                i1_same = i1_tail
                i1_diff = i1_head
            else:
                # 0, 1, 1 or 1, 0, 0
                print("Impossible case")
                raise ValueError()

            # same part is more powerful than total
            cut_pairs.append((i0, i0_same, 1 - mu if is_total_better else mu))
            cut_pairs.append((i0, i0_diff, 1.0 if is_total_better else 0.0))
            cut_pairs.append((i1, i1_same, mu if is_total_better else 1 - mu))
            cut_pairs.append((i1, i1_diff, 0.0 if is_total_better else 1.0))

        pair_index += 1
    print(length, valid_feedback, pair_index, len(cut_pairs))

    pairs_np = np.array(
        cut_pairs, dtype=[("s0", "i4", (2,)), ("s1", "i4", (2,)), ("mu", "f")]
    )

    pair_path = get_pair_path(
        env_name=env_name,
        exp_name=exp_name,
        pair_type=pair_type,
        pair_algo=f"cut-{mu_scale}",
    )
    np.savez(pair_path, data=pairs_np)
