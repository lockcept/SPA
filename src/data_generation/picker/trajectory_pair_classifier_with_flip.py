import os
import numpy as np
import torch
from tqdm import tqdm

from data_generation.classifier.trajectory_pair_classifier import (
    TrajectoryPairClassifier,
    train_trajectory_pair_classifier,
)
from data_generation.utils import generate_pairs_from_indices, save_feedbacks_npz
from data_loading import get_dataloader, load_pair
from utils import get_trajectory_pair_classifier_path

TRAJECTORY_LENGTH = 25


def generate_classifier_flip_pairs(
    env_name,
    exp_name,
    traj_set,
    num_epochs=100,
    batch_size=32,
    top_k=100000,
    device="cpu",
):
    """
    Generate pairs using a trained classifier with flipped pairs.
    """

    train_pair_algo = "ternary-500"

    model_path = get_trajectory_pair_classifier_path(
        env_name=env_name, exp_name=exp_name, pair_algo=train_pair_algo
    )

    if os.path.exists(model_path):
        print(f"Removing existing classifier model: {model_path}")
        os.remove(model_path)

    train_trajectory_pair_classifier(
        env_name=env_name,
        exp_name=exp_name,
        pair_algo=train_pair_algo,
        num_epochs=num_epochs,
        batch_size=batch_size,
        device=device,
    )

    test_pairs = generate_pairs_from_indices(
        trajectories=traj_set,
        pair_count=500000,
        trajectory_length=TRAJECTORY_LENGTH,
    )

    test_feedbacks_with_zero = [(s0, s1, 0.0) for s0, s1 in test_pairs]

    test_feedback_name = "pairs_to_augment"

    save_feedbacks_npz(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="test",
        pair_name=test_feedback_name,
        feedbacks=test_feedbacks_with_zero,
    )

    test_loader = get_dataloader(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="test",
        pair_algo=test_feedback_name,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    classifier = TrajectoryPairClassifier(input_dim=43 * 25).to(device)
    classifier.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    classifier.eval()

    predicted_mu_list = []
    flipped_predicted_mu_list = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Model"):
            s0_obs_batch, s0_act_batch, s1_obs_batch, s1_act_batch, _, _, _ = [
                x.to(device) for x in batch
            ]
            batch_dim = s0_obs_batch.shape[0]

            s0_batch = torch.cat((s0_obs_batch, s0_act_batch), dim=-1).reshape(
                batch_dim, -1
            )
            s1_batch = torch.cat((s1_obs_batch, s1_act_batch), dim=-1).reshape(
                batch_dim, -1
            )
            batch = torch.cat((s0_batch, s1_batch), dim=-1)

            output = classifier(batch)
            predicted_mu = torch.sigmoid(output[:, 1])

            flipped_batch = torch.cat((s1_batch, s0_batch), dim=-1)
            flipped_output = classifier(flipped_batch)
            flipped_predicted_mu = torch.sigmoid(flipped_output[:, 1])

            predicted_mu_list.append(predicted_mu.cpu().numpy())
            flipped_predicted_mu_list.append(flipped_predicted_mu.cpu().numpy())

    predicted_mu_array = np.concatenate(predicted_mu_list, axis=0)
    flipped_predicted_mu_array = np.concatenate(flipped_predicted_mu_list, axis=0)

    certainty_scores = np.abs((predicted_mu_array + flipped_predicted_mu_array) - 1)
    top_indices = np.argsort(certainty_scores)[:top_k]

    def categorize_mu(mu_value):
        if mu_value < 1 / 3:
            return 0.0
        elif mu_value > 2 / 3:
            return 1.0
        else:
            return 0.5

    filtered_feedbacks = [
        (test_pairs[i][0], test_pairs[i][1], categorize_mu(predicted_mu_array[i]))
        for i in top_indices
    ]

    existed_feedbacks = load_pair(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_algo=train_pair_algo,
    )

    existed_feedbacks_list = existed_feedbacks.tolist()
    new_feedbacks = existed_feedbacks_list + filtered_feedbacks

    save_feedbacks_npz(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_name="flip_augmented",
        feedbacks=new_feedbacks,
    )

    # save val pairs with same pair_name
    val_feedbacks = load_pair(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="val",
        pair_algo=train_pair_algo,
    )

    save_feedbacks_npz(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="val",
        pair_name="flip_augmented",
        feedbacks=val_feedbacks,
    )
