import glob
import os
from typing import List

import numpy as np
import torch
from data_generation.utils import generate_pairs_from_indices, save_feedbacks_npz
from reward_learning import get_reward_model, train_reward_model, RewardModelBase
from utils.path import get_reward_model_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAJECTORY_LENGTH = 25


def fill_feedback_from_raw_dataset(cumulative_rewards, pairs):
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

        if np.abs(sum_of_rewards_0 - sum_of_rewards_1) < TRAJECTORY_LENGTH / 2:
            mu = 0.5
        else:
            mu = 0 if sum_of_rewards_0 > sum_of_rewards_1 else 1

        feedbacks.append((s0, s1, mu))

    return feedbacks


def fill_feedback_from_models(dataset, pairs, models: List[RewardModelBase]):
    """
    Fill feedback in dataset using multiple reward models and average their mu values.
    Also return the standard deviation of mu values.

    Args:
        dataset: dict
        pairs: list of tuples ((int, int), (int, int))
        models: list of torch.nn.Module
        linear_loss: bool, optional
            If True, use linear loss for mu calculation. Default is False.

    Returns:
        tuple:
            - np array of ((int, int), (int, int), float): mu values.
            - np array of float: standard deviation of mu values.
    """

    num_samples = len(dataset["observations"])
    batch_size = num_samples // 20
    model_outputs = []

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)

        obs_batch = torch.tensor(
            dataset["observations"][start_idx:end_idx], dtype=torch.float32
        ).to(device)
        act_batch = torch.tensor(
            dataset["actions"][start_idx:end_idx], dtype=torch.float32
        ).to(device)

        batch_model_outputs = []
        for model in models:
            rewards = model.batched_forward_trajectory(
                obs_batch=obs_batch, act_batch=act_batch
            )
            batch_model_outputs.append(rewards.detach().cpu().numpy())

        batch_predicted_rewards = np.mean(batch_model_outputs, axis=0)
        model_outputs.append(batch_predicted_rewards)

    predicted_rewards = np.concatenate(model_outputs, axis=0).squeeze()
    cumulative_rewards = np.cumsum(predicted_rewards, dtype=np.float64)

    feedbacks = []

    for i0, i1 in pairs:
        s0, e0 = i0
        s1, e1 = i1

        sum_of_rewards_0 = cumulative_rewards[e0 - 1] - (
            cumulative_rewards[s0 - 1] if s0 > 0 else 0
        )
        sum_of_rewards_1 = cumulative_rewards[e1 - 1] - (
            cumulative_rewards[s1 - 1] if s1 > 0 else 0
        )

        mu = sum_of_rewards_1 / (sum_of_rewards_0 + sum_of_rewards_1)

        feedbacks.append(((s0, e0), (s1, e1), mu))

    return feedbacks


def generate_active_margin_pairs(
    dataset,
    env_name,
    exp_name,
    traj_set,
    val_pairs,
    total_pairs_count=500,
    active_round=5,
    pairs_scale=100,
    num_epoch=100,
    reward_model_algo="MR-linear",
    reward_model_count=3,
):
    """
    Generate active margin pairs.
    """
    feedbacks = []
    pairs_count_per_round = total_pairs_count // active_round
    pair_algo = "active-margin"
    intermediate_pair_algo = f"{pair_algo}-intermediate"
    obs_dim, act_dim = dataset["observations"].shape[1], dataset["actions"].shape[1]

    cumulative_rewards = np.cumsum(dataset["rewards"], dtype=np.float64)

    # remove models if exists
    model_path_pattern = get_reward_model_path(
        env_name=env_name,
        exp_name=exp_name,
        pair_algo=intermediate_pair_algo,
        reward_model_algo=reward_model_algo,
        reward_model_tag="*",
    )
    model_files = glob.glob(model_path_pattern)

    for model_file in model_files:
        if model_file:
            print("Removing", model_file)
            os.remove(model_file)

    for round_num in range(active_round):
        if not feedbacks:
            new_pairs = generate_pairs_from_indices(
                dataset=dataset,
                trajectories=traj_set,
                pair_count=pairs_count_per_round,
                trajectory_length=TRAJECTORY_LENGTH,
            )
        else:
            for i in range(reward_model_count):
                train_reward_model(
                    env_name=env_name,
                    exp_name=exp_name,
                    pair_algo=intermediate_pair_algo,
                    reward_model_algo=reward_model_algo,
                    reward_model_tag=f"{i:02d}",
                    num_epoch=num_epoch,
                    train_from_existing=True,
                    no_val_data=True,
                )

            # get reward models
            model_path_pattern = get_reward_model_path(
                env_name=env_name,
                exp_name=exp_name,
                pair_algo=intermediate_pair_algo,
                reward_model_algo=reward_model_algo,
                reward_model_tag="*",
            )
            model_files = glob.glob(model_path_pattern)

            models = []

            for model_file in model_files:
                model, _ = get_reward_model(
                    reward_model_algo=reward_model_algo,
                    model_path=model_file,
                    allow_existing=True,
                    obs_dim=obs_dim,
                    act_dim=act_dim,
                )

                if model is not None:
                    model.eval()
                    models.append(model)

            # generate pair candidates
            candidate_pairs = generate_pairs_from_indices(
                dataset=dataset,
                trajectories=traj_set,
                pair_count=pairs_count_per_round * pairs_scale,
                trajectory_length=TRAJECTORY_LENGTH,
            )

            # calculate mu values
            candidate_feedbacks = fill_feedback_from_models(
                dataset=dataset,
                pairs=candidate_pairs,
                models=models,
            )

            # sort with margin
            sorted_feedbacks = sorted(
                candidate_feedbacks,
                key=lambda x: abs(x[2] - 0.5),
            )

            new_feedbacks = sorted_feedbacks[: (total_pairs_count // active_round)]
            new_pairs = [(f[0], f[1]) for f in new_feedbacks]

            feedbacks.extend(
                fill_feedback_from_raw_dataset(
                    cumulative_rewards=cumulative_rewards,
                    pairs=new_pairs,
                )
            )

        # save intermediate pairs
        save_feedbacks_npz(
            env_name=env_name,
            exp_name=exp_name,
            pair_type="train",
            pair_name=intermediate_pair_algo,
            feedbacks=feedbacks,
        )

        save_feedbacks_npz(
            env_name=env_name,
            exp_name=exp_name,
            pair_type="train",
            pair_name=f"{pair_algo}-round-{round_num}",
            feedbacks=feedbacks,
        )

    save_feedbacks_npz(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_name=pair_algo,
        feedbacks=feedbacks,
    )

    val_feedbacks = fill_feedback_from_raw_dataset(
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
