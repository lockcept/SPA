from data_generation.utils import save_feedbacks_npz


def save_raw_pairs(
    env_name,
    exp_name,
    pair_type,
    pairs,
    raw_name,
):
    """
    Args:
        dataset,
        env_name: str,
        exp_name: str,
        pair_type: str,
        pairs: list of ((int, int), (int, int)),
    """

    feedbacks = []

    for i0, i1 in pairs:
        s0, e0 = i0
        s1, e1 = i1
        feedbacks.append(((s0, e0), (s1, e1), 0))

    save_feedbacks_npz(
        env_name=env_name,
        exp_name=exp_name,
        pair_type=pair_type,
        pair_name=raw_name,
        feedbacks=feedbacks,
    )
