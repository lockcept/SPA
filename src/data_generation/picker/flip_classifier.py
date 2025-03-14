import os
import numpy as np
import torch
from data_generation.classifier import (
    BinaryClassifier,
    get_binary_classifier_model,
    train_binary_classifier,
)
from data_generation.utils import generate_pairs_from_indices, save_feedbacks_npz
from data_loading.load_data import process_pairs
from data_loading.preference_dataloader import get_dataloader_from_processed_data
from utils.path import get_binary_classifier_model_path

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


def fill_feedback_from_classifier(dataset, pairs, classifier: BinaryClassifier):
    """
    Fill feedback using a classifier's predicted probabilities.

    Args:
        dataset: dict
        pairs: list of tuples ((int, int), (int, int))
        classifier: trained classifier model

    Returns:
        list of tuples ((int, int), (int, int), float, List[float])
    """

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

    predicted_list = []
    probabilities_list = []

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
            )  # (batch_size, 2)

            predicted = np.full((output.shape[0],), 0.5, dtype=np.float32)
            predicted[probabilities[:, 1] < 0.4] = 0
            predicted[probabilities[:, 1] > 0.6] = 1

            predicted_list.append(predicted)
            probabilities_list.append(probabilities)

    predicted_array = np.concatenate(predicted_list, axis=0)
    probabilities_array = np.concatenate(probabilities_list, axis=0)

    feedbacks = [
        (s0, s1, pred, prob)
        for (s0, s1), pred, prob in zip(pairs, predicted_array, probabilities_array)
    ]

    return feedbacks


def generate_classifier_flip_pairs(
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
    Generate flip pairs using a trained classifier.
    """
    feedbacks = []
    pairs_count_per_round = total_pairs_count // active_round
    pair_algo = "classifier-flip"
    intermediate_pair_algo = f"{pair_algo}-intermediate"
    cumulative_rewards = np.cumsum(dataset["rewards"], dtype=np.float64)
    average_reward = np.mean(dataset["rewards"])

    # remove classifier model if exists
    classifier_path = get_binary_classifier_model_path(
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

            classifier = get_binary_classifier_model(
                env_name=env_name,
                exp_name=exp_name,
                pair_algo=intermediate_pair_algo,
            )

            candidate_feedback_with_prob = fill_feedback_from_classifier(
                dataset=dataset,
                pairs=candidate_pairs,
                classifier=classifier,
            )

            flipped_candidate_pairs = [(s1, s0) for s0, s1 in candidate_pairs]

            flipped_candidate_feedback_with_prob = fill_feedback_from_classifier(
                dataset=dataset,
                pairs=flipped_candidate_pairs,
                classifier=classifier,
            )

            # select feedbacks with comparing the probabilities
            original_probs = [prob for _, _, _, prob in candidate_feedback_with_prob]
            flipped_probs = [
                prob for _, _, _, prob in flipped_candidate_feedback_with_prob
            ]
            mu_diffs = [
                (abs((prob_orig[0] + prob_flip[0]) - 1), idx)
                for idx, (prob_orig, prob_flip) in enumerate(
                    zip(original_probs, flipped_probs)
                )
            ]
            mu_diffs_sorted = sorted(mu_diffs, key=lambda x: x[0], reverse=True)
            filtered_feedbacks = [
                candidate_feedback_with_prob[idx]
                for _, idx in mu_diffs_sorted[:pairs_count_per_round]
            ]

            new_pairs = [(f[0], f[1]) for f in filtered_feedbacks]

            print(
                original_probs[mu_diffs_sorted[0][1]],
                flipped_probs[mu_diffs_sorted[0][1]],
            )
            print(
                original_probs[mu_diffs_sorted[pairs_count_per_round - 1][1]],
                flipped_probs[mu_diffs_sorted[pairs_count_per_round - 1][1]],
            )

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
            train_binary_classifier(
                env_name=env_name,
                exp_name=exp_name,
                pair_algo=intermediate_pair_algo,
                num_epochs=num_epoch,
                remove_if_exists=False,
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
