import numpy as np

from data_generation.full_pairs import generate_and_save_full_pairs
from data_generation.raw_pairs import save_raw_pairs
from data_generation.scored_pairs import generate_score_pairs
from data_generation.utils import (
    extract_trajectory_indices,
    generate_pairs_from_indices,
    generate_pairs_from_using_all,
)
from data_loading.load_data import load_dataset, load_pair


def generate_all_algo_pairs(env_name, exp_name):
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
            pair_algo="raw",
        )["data"]
        is_already_exist = True
    except FileNotFoundError:
        is_already_exist = False

    if is_already_exist:
        print("Raw Pair already exists, use it for generating")

        train_pairs_with_mu = load_pair(
            env_name=env_name,
            exp_name=exp_name,
            pair_type="train",
            pair_algo="raw",
        )["data"]
        train_all_pairs_with_mu = load_pair(
            env_name=env_name,
            exp_name=exp_name,
            pair_type="train",
            pair_algo="raw_all",
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

        all_traj_set = []
        for p in train_all_pairs_with_mu:
            all_traj_set.append(p[0])
            all_traj_set.append(p[1])

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

        train_all_pairs = generate_pairs_from_using_all(train_set)

        all_traj_set = []
        for p in train_all_pairs:
            all_traj_set.append(p[0])
            all_traj_set.append(p[1])

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
        save_raw_pairs(
            env_name=env_name,
            exp_name=exp_name,
            pair_type="train_all",
            pairs=train_all_pairs,
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
            "binary-with-0.5",
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
            "binary-with-0.5",
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

    generate_score_pairs(
        dataset=dataset,
        env_name=env_name,
        exp_name=exp_name,
        num_epochs=100,
        pair_algo="full-binary-with-0.5",
        score_model="lstm",
        aug_list=["10000"],
        traj_set=all_traj_set,
    )
