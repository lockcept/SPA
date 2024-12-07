import numpy as np

from data_generation.cut_pairs import generate_and_save_cut_pairs
from data_generation.full_pairs import generate_and_save_full_pairs
from data_generation.list_pairs import generate_and_save_list_pairs
from data_generation.scored_pairs import generate_score_pairs
from data_generation.utils import extract_trajectory_indices
from data_loading.load_data import load_dataset, load_pair


def choose_index_pairs_from_list(list_length, pair_count):
    """
    choose pairs from a list
    there is not unused element in the list
    some indices may be repeated
    """

    index_pairs = []
    for i in range(pair_count):
        if i * 2 < list_length:
            if i * 2 + 1 < list_length:
                index_pairs.append((i * 2, i * 2 + 1))
            else:
                index_pairs.append((i * 2, np.random.randint(0, list_length - 1)))
        else:
            i_0, i_1 = np.random.randint(0, list_length - 1, 2)
            index_pairs.append((i_0, i_1))
    return index_pairs


def cut_trajectories(indices, trajectory_length):
    """
    cut trajectories to have a fixed length
    """
    valid_trajectories = []

    for start, end in indices:
        if end - start >= trajectory_length:
            random_start = np.random.randint(start, end - trajectory_length)
            valid_trajectories.append((random_start, random_start + trajectory_length))

    return valid_trajectories


def generate_all_algo_pairs(env_name, exp_name, include_score_pairs=False):
    """
    generate all algo pairs with hard-coded values
    """
    trajectory_length = 50

    train_trajectories = 1000
    val_trajectories = 1000
    test_trajectories = 1000

    train_pairs = 500
    val_pairs = 500
    test_pairs = 500

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

    dataset = load_dataset(env_name=env_name)
    if is_already_exist:
        print("full-binary Pair already exists, use it for generating")

        train_pairs_with_mu = load_pair(
            env_name=env_name,
            exp_name=exp_name,
            pair_type="train",
            pair_algo="full-binary",
        )["data"]
        val_pairs_with_mu = load_pair(
            env_name=env_name,
            exp_name=exp_name,
            pair_type="val",
            pair_algo="full-binary",
        )["data"]
        test_pairs_with_mu = load_pair(
            env_name=env_name,
            exp_name=exp_name,
            pair_type="test",
            pair_algo="full-binary",
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
        indices = extract_trajectory_indices(dataset)
        indices = cut_trajectories(indices, trajectory_length)
        np.random.shuffle(indices)

        if len(indices) < train_trajectories + val_trajectories + test_trajectories:
            print("Not enough trajectories")
            return

        train_set = indices[:train_trajectories]
        val_set = indices[train_trajectories : train_trajectories + val_trajectories]
        test_set = indices[
            train_trajectories
            + val_trajectories : train_trajectories
            + val_trajectories
            + test_trajectories
        ]

        train_index_pairs = choose_index_pairs_from_list(
            train_trajectories, train_pairs
        )
        val_index_pairs = choose_index_pairs_from_list(val_trajectories, val_pairs)
        test_index_pairs = choose_index_pairs_from_list(test_trajectories, test_pairs)

        train_pairs = [(train_set[i0], train_set[i1]) for i0, i1 in train_index_pairs]
        val_pairs = [(val_set[i0], val_set[i1]) for i0, i1 in val_index_pairs]
        test_pairs = [(test_set[i0], test_set[i1]) for i0, i1 in test_index_pairs]

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
        all_indices=train_set,
        num_groups=[2, 3, 5, 11],
    )

    generate_and_save_list_pairs(
        dataset=dataset,
        env_name=env_name,
        exp_name=exp_name,
        pair_type="val",
        pairs=val_pairs,
        all_indices=val_set,
        num_groups=[2, 3, 5, 11],
    )

    if not include_score_pairs:
        return

    # score-rnn

    generate_score_pairs(
        dataset=dataset,
        env_name=env_name,
        exp_name=exp_name,
        num_epochs=2000,
        pair_algo="full-binary",
        score_model="rnn",
    )

    # cut-score-rnn

    for cut_type in ["0.5", "0.25", "half-random", "random"]:
        generate_and_save_cut_pairs(
            dataset=dataset,
            env_name=env_name,
            exp_name=exp_name,
            pair_type="train",
            pairs=train_pairs,
            cut_type=cut_type,
        )

        generate_and_save_cut_pairs(
            dataset=dataset,
            env_name=env_name,
            exp_name=exp_name,
            pair_type="val",
            pairs=val_pairs,
            cut_type=cut_type,
        )

        generate_score_pairs(
            dataset=dataset,
            env_name=env_name,
            exp_name=exp_name,
            num_epochs=2000,
            pair_algo=f"cut-{cut_type}",
            score_model="rnn",
        )
