import numpy as np


def generate_and_save_cut_pairs(dataset, env_name, pair_name_base, pairs):
    """
    Args:
        dataset,
        env_name: str,
        pair_name_base: str,
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

        m0 = (s0 + e0) // 2
        m1 = (s1 + e1) // 2

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
    np.savez(f"pair/{env_name}/{pair_name_base}_cut-binary.npz", data=pairs_np)

    return pairs_np
