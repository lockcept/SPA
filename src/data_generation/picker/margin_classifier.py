import os
import numpy as np
import torch
from data_generation.classifier import (
    get_classifier_model,
    Classifier,
    train_classifier,
)
from data_generation.utils import generate_pairs_from_indices, save_feedbacks_npz
from data_loading.load_data import process_pairs
from data_loading.preference_dataloader import get_dataloader_from_processed_data
from utils.path import get_classifier_model_path

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


def fill_feedback_from_classifier(dataset, pairs, classifier: Classifier):
    """
    Fill feedback using a classifier's predicted probabilities.

    Args:
        dataset: dict
        pairs: list of tuples ((int, int), (int, int))
        classifier: trained classifier model

    Returns:
        list of tuples ((int, int), (int, int), float, List[float])
    """

    feedbacks = []
    classifier.eval()

    pairs_with_zero_mu = np.array(
        [(s0, s1, 0.0) for s0, s1 in pairs],
        dtype=[
            ("s0", "i4", (2,)),
            ("s1", "i4", (2,)),
            ("mu", "f"),
        ],
    )

    # Evaluate model with result data
    processed_data = process_pairs(dataset, pairs_with_zero_mu)
    dataloader = get_dataloader_from_processed_data(
        processed_data, shuffle=False, drop_last=False
    )

    with torch.no_grad():
        for batch in dataloader:
            (
                s0_obs_batch,
                s0_act_batch,
                s1_obs_batch,
                s1_act_batch,
                _,
                _,
                _,
            ) = [x.to(device) for x in batch]

            s0_input = torch.cat((s0_obs_batch, s0_act_batch), dim=-1).view(
                s0_obs_batch.shape[0], -1
            )
            s1_input = torch.cat((s1_obs_batch, s1_act_batch), dim=-1).view(
                s1_obs_batch.shape[0], -1
            )

            output = classifier(s0_input, s1_input)
            probabilities = (
                torch.softmax(output, dim=1).cpu().numpy()
            )  # (batch_size, 3)
            _, predicted = torch.max(output, 1)

            mu_values = np.where(
                predicted.cpu().numpy() == 2, 0.5, predicted.cpu().numpy()
            )

            for idx in range(len(s0_obs_batch)):
                feedbacks.append(
                    (pairs[idx][0], pairs[idx][1], mu_values[idx], probabilities[idx])
                )

    return feedbacks


def generate_classifier_margin_pairs(
    dataset,
    env_name,
    exp_name,
    traj_set,
    val_pairs,
    total_pairs_count=500,
    active_round=10,
    pairs_scale=100,
    num_epoch=100,
):
    """
    Generate margin pairs using a trained classifier.
    """
    feedbacks = []
    pairs_count_per_round = total_pairs_count // active_round
    pair_algo = "classifier-margin"
    intermediate_pair_algo = f"{pair_algo}-intermediate"
    obs_dim, act_dim = dataset["observations"].shape[1], dataset["actions"].shape[1]
    cumulative_rewards = np.cumsum(dataset["rewards"], dtype=np.float64)

    # remove classifier model if exists
    classifier_path = get_classifier_model_path(
        env_name, exp_name, pair_algo=intermediate_pair_algo
    )
    if os.path.exists(classifier_path):
        print(f"Removing existing classifier model: {classifier_path}")
        os.remove(classifier_path)

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

            classifier = get_classifier_model(
                env_name=env_name,
                exp_name=exp_name,
                pair_algo=intermediate_pair_algo,
            )

            candidate_feedback_with_prob = fill_feedback_from_classifier(
                dataset=dataset,
                pairs=candidate_pairs,
                classifier=classifier,
            )

            # select feedbacks with the largest margin
            sorted_feedbacks = sorted(
                candidate_feedback_with_prob,
                key=lambda x: np.sort(x[3])[0] - np.sort(x[3])[2],
            )
            filtered_feedbacks = sorted_feedbacks[: (total_pairs_count // active_round)]
            new_pairs = [(f[0], f[1]) for f in filtered_feedbacks]

        new_feedbacks = fill_feedback_from_raw_dataset(
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

        save_feedbacks_npz(
            env_name=env_name,
            exp_name=exp_name,
            pair_type="train",
            pair_name=intermediate_pair_algo,
            feedbacks=feedbacks,
        )

        if round_num < active_round - 1:
            train_classifier(
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
