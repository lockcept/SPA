import numpy as np
import torch

from data_generation.raw_pairs import save_raw_pairs
from data_generation.ternary_pairs import generate_and_save_ternary_pairs
from data_generation.picker.trajectory_pair_classifier_with_flip import (
    generate_classifier_flip_pairs,
)
from data_generation.utils import (
    generate_pairs_from_indices,
    generate_pairs_from_using_all,
)
from data_loading import load_dataset, load_pair, extract_trajectory_indices

device = "cuda" if torch.cuda.is_available() else "cpu"


def generate_all_algo_pairs(env_name, exp_name):
    """
    generate all algo pairs with hard-coded values
    """
    dataset = load_dataset(env_name=env_name)
    indices = extract_trajectory_indices(dataset)
    np.random.shuffle(indices)

    trajectory_length = 25

    # hard coded values
    if len(indices) >= 1800:
        train_trajectories_cnt = len(indices) - 800
        val_trajectories_cnt = 400
        test_trajectories_cnt = 400
    elif len(indices) >= 1200:
        train_trajectories_cnt = 1000
        val_trajectories_cnt = 100
        test_trajectories_cnt = 100
    elif len(indices) >= 600:
        train_trajectories_cnt = 500
        val_trajectories_cnt = 50
        test_trajectories_cnt = 50
    else:
        train_trajectories_cnt = 100
        val_trajectories_cnt = 100
        test_trajectories_cnt = 100

    print(
        f"train_trajectories_cnt: {train_trajectories_cnt}, val_trajectories_cnt: {val_trajectories_cnt}, test_trajectories_cnt: {test_trajectories_cnt}"
    )

    train_pairs_cnt = 100000
    val_pairs_cnt = 100000
    test_pairs_cnt = 500

    try:
        _ = load_pair(
            env_name=env_name,
            exp_name=exp_name,
            pair_type="train",
            pair_algo="raw",
        )
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
        )
        train_all_pairs_with_mu = load_pair(
            env_name=env_name,
            exp_name=exp_name,
            pair_type="train_all",
            pair_algo="raw",
        )
        val_pairs_with_mu = load_pair(
            env_name=env_name,
            exp_name=exp_name,
            pair_type="val",
            pair_algo="raw",
        )
        test_pairs_with_mu = load_pair(
            env_name=env_name,
            exp_name=exp_name,
            pair_type="test",
            pair_algo="raw",
        )

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

        print("Generating train pairs")
        train_pairs = generate_pairs_from_indices(
            train_set, train_pairs_cnt, trajectory_length
        )

        print("Generating train all pairs")
        train_all_pairs = generate_pairs_from_using_all(train_set)

        all_traj_set = []
        for p in train_all_pairs:
            all_traj_set.append(p[0])
            all_traj_set.append(p[1])

        print("Generating val pairs")
        val_pairs = generate_pairs_from_indices(
            val_set, val_pairs_cnt, trajectory_length
        )

        print("Generating test pairs")
        test_pairs = generate_pairs_from_indices(
            test_set, test_pairs_cnt, trajectory_length
        )

        # raw
        save_raw_pairs(
            env_name=env_name,
            exp_name=exp_name,
            pair_type="train",
            pairs=train_pairs,
            raw_name="raw",
        )
        save_raw_pairs(
            env_name=env_name,
            exp_name=exp_name,
            pair_type="val",
            pairs=val_pairs,
            raw_name="raw",
        )
        save_raw_pairs(
            env_name=env_name,
            exp_name=exp_name,
            pair_type="test",
            pairs=test_pairs,
            raw_name="raw",
        )
        save_raw_pairs(
            env_name=env_name,
            exp_name=exp_name,
            pair_type="train_all",
            pairs=train_all_pairs,
            raw_name="raw",
        )

    generate_and_save_ternary_pairs(
        dataset=dataset,
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pairs=train_pairs,
        mu_types=[
            "500",
        ],
    )
    generate_and_save_ternary_pairs(
        dataset=dataset,
        env_name=env_name,
        exp_name=exp_name,
        pair_type="val",
        pairs=val_pairs,
        mu_types=[
            "500",
        ],
    )
    generate_and_save_ternary_pairs(
        dataset=dataset,
        env_name=env_name,
        exp_name=exp_name,
        pair_type="test",
        pairs=val_pairs,
        mu_types=["500"],
    )

    generate_classifier_flip_pairs(
        env_name=env_name,
        exp_name=exp_name,
        traj_set=all_traj_set,
        val_pairs=val_pairs,
        device=device,
    )
