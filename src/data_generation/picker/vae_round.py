import os
import numpy as np
import torch
from auto_encoder.vae import evaluate_vae_error, train_vae
from data_generation.utils import generate_pairs_from_indices, save_feedbacks_npz
from utils.path import get_vae_model_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAJECTORY_LENGTH = 25


def fill_feedback_from_raw_dataset(average_reward, cumulative_rewards, pairs):
    """
    Fill feedback in dataset using cumulative rewards and calculate mu values.
    """

    feedbacks = []

    for s0, s1 in pairs:
        sum_of_rewards_0 = cumulative_rewards[s0[1] - 1] - (
            cumulative_rewards[s0[0] - 1] if s0[0] > 0 else 0
        )
        sum_of_rewards_1 = cumulative_rewards[s1[1] - 1] - (
            cumulative_rewards[s1[0] - 1] if s1[0] > 0 else 0
        )

        if (
            np.abs(sum_of_rewards_0 - sum_of_rewards_1)
            < average_reward * TRAJECTORY_LENGTH * 0.1
        ):
            mu = 0.5
        else:
            mu = 0 if sum_of_rewards_0 > sum_of_rewards_1 else 1

        feedbacks.append((s0, s1, mu))

    return feedbacks


def generate_vae_pairs(
    dataset,
    env_name,
    exp_name,
    traj_set,
    val_pairs,
    total_pairs_count=500,
    active_round=10,
    pairs_scale=1000,
    num_epoch=100,
):
    """
    Generate margin pairs using a trained classifier.
    """
    feedbacks = []
    pairs_count_per_round = total_pairs_count // active_round
    pair_algo = "vae"
    intermediate_pair_algo = f"{pair_algo}-intermediate"
    cumulative_rewards = np.cumsum(dataset["rewards"], dtype=np.float64)
    average_reward = np.mean(dataset["rewards"])

    vae_path = get_vae_model_path(env_name, exp_name, pair_algo=intermediate_pair_algo)
    if os.path.exists(vae_path):
        print(f"Removing existing vae model: {vae_path}")
        os.remove(vae_path)

    for round_num in range(active_round):
        if not feedbacks:
            new_pairs = generate_pairs_from_indices(
                dataset=dataset,
                trajectories=traj_set,
                pair_count=pairs_count_per_round,
                trajectory_length=TRAJECTORY_LENGTH,
            )
        else:
            candidate_pairs = generate_pairs_from_indices(
                dataset=dataset,
                trajectories=traj_set,
                pair_count=pairs_count_per_round * pairs_scale,
                trajectory_length=TRAJECTORY_LENGTH,
            )

            pairs, errors = evaluate_vae_error(
                env_name=env_name,
                exp_name=exp_name,
                pair_algo=pair_algo,
                pairs=candidate_pairs,
            )

            new_pairs = sorted(zip(pairs, errors), key=lambda x: x[1], reverse=True)[
                :pairs_count_per_round
            ]

        new_feedbacks = fill_feedback_from_raw_dataset(
            average_reward=average_reward,
            cumulative_rewards=cumulative_rewards,
            pairs=new_pairs,
        )

        feedbacks.extend(new_feedbacks)

        save_feedbacks_npz(
            env_name=env_name,
            exp_name=exp_name,
            pair_type="train",
            pair_name=f"{pair_algo}-round-{round_num}",
            feedbacks=feedbacks,
        )

        flipped_feedbacks = [(s1, s0, 1.0 - mu) for s0, s1, mu in feedbacks]

        save_feedbacks_npz(
            env_name=env_name,
            exp_name=exp_name,
            pair_type="train",
            pair_name=intermediate_pair_algo,
            feedbacks=feedbacks + flipped_feedbacks,
        )

        if round_num < active_round - 1:
            train_vae(
                env_name=env_name,
                exp_name=exp_name,
                pair_algo=intermediate_pair_algo,
                num_epochs=num_epoch,
            )

    save_feedbacks_npz(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_name=pair_algo,
        feedbacks=feedbacks,
    )

    val_feedbacks = fill_feedback_from_raw_dataset(
        average_reward=average_reward,
        cumulative_rewards=cumulative_rewards,
        pairs=val_pairs,
    )

    save_feedbacks_npz(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="val",
        pair_name=pair_algo,
        feedbacks=val_feedbacks,
    )
