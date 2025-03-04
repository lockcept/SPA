import glob
import os
from typing import List

import numpy as np
import torch
from data_generation.raw_pairs import save_raw_pairs
from data_generation.utils import generate_pairs_from_indices, save_feedbacks_npz
from reward_learning import get_reward_model, train_reward_model, RewardModelBase
from utils.path import get_reward_model_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fill_feedback_from_pairs(dataset, pairs, models: List[RewardModelBase]):
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
        mu = 0 if sum_of_rewards_0 > sum_of_rewards_1 else 1

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
    pairs = []
    pairs_count_per_round = total_pairs_count // active_round
    pair_algo = "active-margin"
    intermediate_pair_algo = f"{pair_algo}-intermediate"
    obs_dim, act_dim = dataset["observations"].shape[1], dataset["actions"].shape[1]

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

    for _ in range(active_round):
        if not pairs:
            pairs = generate_pairs_from_indices(
                dataset=dataset,
                trajectories=traj_set,
                pair_count=pairs_count_per_round,
                trajectory_length=25,
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
                trajectory_length=25,
            )

            # calculate mu values
            candidate_feedbacks = fill_feedback_from_pairs(
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
            pairs.extend(new_pairs)

        # save intermediate pairs
        save_raw_pairs(
            env_name=env_name,
            exp_name=exp_name,
            pair_type="train",
            pairs=pairs,
            raw_name=intermediate_pair_algo,
        )

    # save full-binary pairs from selected pairs
    feedbacks = []

    for i0, i1 in pairs:
        s0, e0 = i0
        s1, e1 = i1

        sum_of_rewards_0 = np.sum(dataset["rewards"][s0:e0])
        sum_of_rewards_1 = np.sum(dataset["rewards"][s1:e1])
        mu = 0 if sum_of_rewards_0 > sum_of_rewards_1 else 1
        feedbacks.append(((s0, e0), (s1, e1), mu))

    save_feedbacks_npz(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_name=pair_algo,
        feedbacks=feedbacks,
    )

    val_feedbacks = fill_feedback_from_pairs(
        dataset=dataset,
        pairs=val_pairs,
        models=models,
    )

    save_feedbacks_npz(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="val",
        pair_name=pair_algo,
        feedbacks=val_feedbacks,
    )
