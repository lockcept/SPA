import numpy as np

from data_generation.cut_pairs import generate_and_save_cut_pairs
from data_generation.full_pairs import generate_and_save_full_pairs
from data_generation.list_pairs import generate_and_save_list_pairs
from data_generation.raw_pairs import save_raw_pairs
from data_generation.scored_pairs import generate_score_pairs
from data_generation.utils import extract_trajectory_indices
from data_loading.load_data import load_dataset, load_pair


def generate_pairs_from_indices(trajectories, pair_count, trajectory_length):
    """
    choose pairs from indices and cut them to have a fixed length with same starting point
    To utilize as many pairs as possible, start by using pairs from the beginning.
    """

    pairs = []
    valid_trajectories = [t for t in trajectories if (t[1] - t[0]) >= trajectory_length]
    total_trajectory_count = len(valid_trajectories)

    for i in range(pair_count):
        if i * 2 < total_trajectory_count:
            if i * 2 + 1 < total_trajectory_count:
                first_pair_index = i * 2
                second_pair_index = i * 2 + 1
            else:
                first_pair_index = i * 2
                second_pair_index = np.random.randint(0, total_trajectory_count - 1)
        else:
            first_pair_index, second_pair_index = np.random.randint(
                0, total_trajectory_count - 1, 2
            )

        first_trajectory = valid_trajectories[first_pair_index]
        second_trajectory = valid_trajectories[second_pair_index]
        min_length = min(
            (
                first_trajectory[1] - first_trajectory[0],
                second_trajectory[1] - second_trajectory[0],
            )
        )

        if min_length == trajectory_length:
            first_start_point = 0
            second_start_point = 0
        else:
            first_start_point = np.random.randint(0, min_length - trajectory_length)
            second_start_point = np.random.randint(0, min_length - trajectory_length)
        pairs.append(
            (
                (
                    first_trajectory[0] + first_start_point,
                    first_trajectory[0] + first_start_point + trajectory_length,
                ),
                (
                    second_trajectory[0] + second_start_point,
                    second_trajectory[0] + second_start_point + trajectory_length,
                ),
            )
        )

    return pairs


def generate_all_algo_pairs(env_name, exp_name, include_score_pairs=False):
    """
    generate all algo pairs with hard-coded values
    """
    dataset = load_dataset(env_name=env_name)
    indices = extract_trajectory_indices(dataset)
    np.random.shuffle(indices)

    trajectory_length = 25

    train_trajectories_cnt = 1000
    val_trajectories_cnt = 1000
    test_trajectories_cnt = 1000

    # hard coded values
    if len(indices) == 600:
        train_trajectories_cnt = 500
        val_trajectories_cnt = 50
        test_trajectories_cnt = 50
        print("Using 0.1 dataset")
    elif len(indices) == 1800:
        train_trajectories_cnt = 1000
        val_trajectories_cnt = 400
        test_trajectories_cnt = 400
        print("Using 0.3 dataset")
    elif len(indices) == 2100:
        train_trajectories_cnt = 1000
        val_trajectories_cnt = 500
        test_trajectories_cnt = 500
        print("Using 0.7 dataset")

    print(
        f"train_trajectories_cnt: {train_trajectories_cnt}, val_trajectories_cnt: {val_trajectories_cnt}, test_trajectories_cnt: {test_trajectories_cnt}"
    )

    train_pairs_cnt = 500
    val_pairs_cnt = 500
    test_pairs_cnt = 500

    try:
        _ = load_pair(
            env_name=env_name,
            exp_name=exp_name,
            pair_type="train",
            pair_algo="full-binary",
        )["data"]
        is_already_exist = True
    except FileNotFoundError:
        is_already_exist = False

    if is_already_exist:
        print("full-binary Pair already exists, use it for generating")

        train_pairs_with_mu = load_pair(
            env_name=env_name,
            exp_name=exp_name,
            pair_type="train",
            pair_algo="raw",
        )["data"]
        val_pairs_with_mu = load_pair(
            env_name=env_name,
            exp_name=exp_name,
            pair_type="val",
            pair_algo="raw",
        )["data"]
        test_pairs_with_mu = load_pair(
            env_name=env_name,
            exp_name=exp_name,
            pair_type="test",
            pair_algo="raw",
        )["data"]

        train_pairs = [
            ((p["s0"][0], p["s0"][1]), (p["s1"][0], p["s1"][1]))
            for p in train_pairs_with_mu
        ]
        val_pairs = [
            ((p["s0"][0], p["s0"][1]), (p["s1"][0], p["s1"][1]))
            for p in val_pairs_with_mu
        ]
        test_pairs = [
            ((p["s0"][0], p["s0"][1]), (p["s1"][0], p["s1"][1]))
            for p in test_pairs_with_mu
        ]

        train_set = []
        val_set = []
        test_set = []

        for p in train_pairs:
            train_set.append(p[0])
            train_set.append(p[1])

        for p in val_pairs:
            val_set.append(p[0])
            val_set.append(p[1])

        for p in test_pairs:
            test_set.append(p[0])
            test_set.append(p[1])

        train_set = list(set(train_set))
        val_set = list(set(val_set))
        test_set = list(set(test_set))

    else:
        if (
            len(indices)
            < train_trajectories_cnt + val_trajectories_cnt + test_trajectories_cnt
        ):
            print("Not enough trajectories")
            return

        train_set = indices[:train_trajectories_cnt]
        val_set = indices[
            train_trajectories_cnt : train_trajectories_cnt + val_trajectories_cnt
        ]
        test_set = indices[
            train_trajectories_cnt
            + val_trajectories_cnt : train_trajectories_cnt
            + val_trajectories_cnt
            + test_trajectories_cnt
        ]

        train_pairs = generate_pairs_from_indices(
            train_set, train_pairs_cnt, trajectory_length
        )
        val_pairs = generate_pairs_from_indices(
            val_set, val_pairs_cnt, trajectory_length
        )
        test_pairs = generate_pairs_from_indices(
            test_set, test_pairs_cnt, trajectory_length
        )

    # raw
    save_raw_pairs(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pairs=train_pairs,
    )
    save_raw_pairs(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="val",
        pairs=val_pairs,
    )
    save_raw_pairs(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="test",
        pairs=test_pairs,
    )

    # full
    generate_and_save_full_pairs(
        dataset=dataset,
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pairs=train_pairs,
        mu_types=[
            "binary",
            "linear",
        ],
    )
    generate_and_save_full_pairs(
        dataset=dataset,
        env_name=env_name,
        exp_name=exp_name,
        pair_type="val",
        pairs=val_pairs,
        mu_types=[
            "binary",
            "linear",
        ],
    )

    # test
    generate_and_save_full_pairs(
        dataset=dataset,
        env_name=env_name,
        exp_name=exp_name,
        pair_type="test",
        pairs=test_pairs,
        mu_types=["binary"],
    )

    # list

    generate_and_save_list_pairs(
        dataset=dataset,
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pairs=train_pairs,
        num_groups=[2, 3, 5, 11],
    )

    generate_and_save_list_pairs(
        dataset=dataset,
        env_name=env_name,
        exp_name=exp_name,
        pair_type="val",
        pairs=val_pairs,
        num_groups=[2, 3, 5, 11],
    )

    if not include_score_pairs:
        return

    # rnn-cut-X

    for mu_scale in [0.75, 1.0]:
        # overided pairs are OK
        used_set_train = generate_and_save_cut_pairs(
            dataset=dataset,
            env_name=env_name,
            exp_name=exp_name,
            pair_type="train",
            pairs=train_pairs,
            cut_type="0.5",
            mu_scale=mu_scale,
        )

        used_set_val = generate_and_save_cut_pairs(
            dataset=dataset,
            env_name=env_name,
            exp_name=exp_name,
            pair_type="val",
            pairs=val_pairs,
            cut_type="0.5",
            mu_scale=mu_scale,
        )

        generate_score_pairs(
            dataset=dataset,
            env_name=env_name,
            exp_name=exp_name,
            num_epochs=100,
            pair_algo=f"cut-{mu_scale}",
            score_model="rnn",
        )

    new_train_pairs = generate_pairs_from_indices(
        used_set_train, train_pairs_cnt, trajectory_length
    )
    new_val_pairs = generate_pairs_from_indices(
        used_set_val, val_pairs_cnt, trajectory_length
    )

    generate_and_save_full_pairs(
        dataset=dataset,
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pairs=new_train_pairs,
        mu_types=["binary-with-0.5"],
    )

    generate_and_save_full_pairs(
        dataset=dataset,
        env_name=env_name,
        exp_name=exp_name,
        pair_type="val",
        pairs=new_val_pairs,
        mu_types=["binary-with-0.5"],
    )
