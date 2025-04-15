import glob
import os
import shutil
import numpy as np
import torch
from tqdm import tqdm

from data_generation.classifier.trajectory_pair_classifier import (
    TrajectoryPairClassifier,
    train_trajectory_pair_classifier,
)
from data_generation.utils import generate_pairs_from_indices, save_feedbacks_npz
from data_loading import get_dataloader, load_pair
from data_loading.load_data import load_dataset
from reward_learning.get_model import get_reward_model
from reward_learning.train_model import train_reward_model
from utils import get_reward_model_path
from utils.path import get_trajectory_pair_classifier_path

TRAJECTORY_LENGTH = 25


def predict_rewards(
    env_name,
    exp_name,
    pair_algo,
    reward_model_algo,
    device,
):
    dataset = load_dataset(env_name)
    obs_dim = dataset["observations"].shape[1]
    act_dim = dataset["actions"].shape[1]

    print("obs_dim:", obs_dim, "act_dim:", act_dim)
    model_path_pattern = get_reward_model_path(
        env_name=env_name,
        exp_name=exp_name,
        pair_algo=pair_algo,
        reward_model_algo=reward_model_algo,
        reward_model_tag="*",
    )
    model_files = glob.glob(model_path_pattern)
    model_list = []

    for model_file in model_files:
        model, _ = get_reward_model(
            reward_model_algo=reward_model_algo,
            obs_dim=obs_dim,
            act_dim=act_dim,
            model_path=model_file,
            allow_existing=True,
        )
        model_list.append(model)

    dataset = load_dataset(env_name)

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
        for model in model_list:
            rewards = model.batched_forward_trajectory(
                obs_batch=obs_batch, act_batch=act_batch
            )
            batch_model_outputs.append(rewards.detach().cpu().numpy())

        batch_predicted_rewards = np.mean(batch_model_outputs, axis=0)
        model_outputs.append(batch_predicted_rewards)

    predicted_rewards = np.concatenate(model_outputs, axis=0).squeeze()

    return predicted_rewards


def generate_temp_pairs(
    env_name,
    exp_name,
    traj_set,
    device="cpu",
):
    train_pair_algo = "ternary-500"

    reward_model_algo = "MR-linear"

    # 1. learn reward
    for k in range(3):
        train_reward_model(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo=train_pair_algo,
            reward_model_algo=reward_model_algo,
            reward_model_tag=f"{k:02d}",
            num_epoch=100,
        )

    # 2. copy reward models
    for k in range(3):
        reward_model_path = get_reward_model_path(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo=train_pair_algo,
            reward_model_algo=reward_model_algo,
            reward_model_tag=f"{k:02d}",
        )

        dst_path1 = reward_model_path.replace(train_pair_algo, "ternary-500-aug-mr")
        dst_path2 = reward_model_path.replace(
            train_pair_algo, "ternary-500-aug-classifier"
        )

        os.makedirs(os.path.dirname(dst_path1), exist_ok=True)
        os.makedirs(os.path.dirname(dst_path2), exist_ok=True)

        shutil.copy(reward_model_path, dst_path1)
        shutil.copy(reward_model_path, dst_path2)

    # 3. learn classifier
    feedbacks = load_pair(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_algo=train_pair_algo,
    )

    feedbacks_flipped = []
    for s0, s1, mu in feedbacks:
        feedbacks_flipped.append((s0, s1, mu))
        feedbacks_flipped.append((s1, s0, 1 - mu))

    save_feedbacks_npz(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_name=f"{train_pair_algo}-flipped",
        feedbacks=feedbacks_flipped,
    )

    train_trajectory_pair_classifier(
        env_name=env_name,
        exp_name=exp_name,
        pair_algo=f"{train_pair_algo}-flipped",
        num_epochs=100,
        device=device,
    )

    # 4-1. generate augmentation pairs (MR)
    train_all_pairs_with_mu = load_pair(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train_all",
        pair_algo="raw",
    )

    all_traj_set = []

    for p in train_all_pairs_with_mu:
        all_traj_set.append(p[0])
        all_traj_set.append(p[1])

    pair_candidates = generate_pairs_from_indices(
        trajectories=traj_set,
        pair_count=100000,
        trajectory_length=TRAJECTORY_LENGTH,
    )

    predicted_rewards = predict_rewards(
        env_name=env_name,
        exp_name=exp_name,
        pair_algo=f"{train_pair_algo}-aug-mr",
        reward_model_algo="MR-linear",
        device=device,
    )

    feedbacks = feedbacks.tolist()
    feedbacks_mr = []

    predicted_rewards = np.array(predicted_rewards)
    predicted_cumsum = np.cumsum(predicted_rewards, dtype=np.float64)

    for i0, i1 in pair_candidates:
        s0, e0 = i0
        s1, e1 = i1

        sum_of_rewards_0 = predicted_cumsum[e0 - 1] - (
            predicted_cumsum[s0 - 1] if s0 > 0 else 0
        )
        sum_of_rewards_1 = predicted_cumsum[e1 - 1] - (
            predicted_cumsum[s1 - 1] if s1 > 0 else 0
        )

        mu = (sum_of_rewards_1 + 1e-6) / (sum_of_rewards_0 + sum_of_rewards_1 + 2e-6)

        feedbacks_mr.append(((s0, e0), (s1, e1), mu))

    anchors = [0.0, 1.0]

    def closest_anchor(mu):
        scores = [(a, abs(mu - a)) for a in anchors]
        return min(scores, key=lambda x: x[1])  # (anchor, score)

    anchor_score_list = [
        (f[0], f[1], *closest_anchor(f[2]))  # (t0, t1, anchor, score)
        for f in feedbacks_mr
    ]

    anchor_score_list.sort(key=lambda x: x[3])

    feedbacks_mr_final = [
        (t0, t1, anchor) for (t0, t1, anchor, _) in anchor_score_list[:10000]
    ]

    save_feedbacks_npz(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_name=f"{train_pair_algo}-aug-mr",
        feedbacks=feedbacks + feedbacks_mr_final,
    )

    save_feedbacks_npz(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_name=f"{train_pair_algo}-aug-mr-init",
        feedbacks=feedbacks + feedbacks_mr_final,
    )

    # 4-2. generate augmentation pairs (Classifier)
    classifier_model_path = get_trajectory_pair_classifier_path(
        env_name=env_name,
        exp_name=exp_name,
        pair_algo=f"{train_pair_algo}-flipped",
    )

    classifier = TrajectoryPairClassifier(input_dim=43 * 25).to(device)
    classifier.load_state_dict(torch.load(classifier_model_path))
    classifier.eval()

    pair_candidates_with_zero_mu = [(t0, t1, 0.0) for (t0, t1) in pair_candidates]
    save_feedbacks_npz(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_name=f"pair-candidates",
        feedbacks=pair_candidates_with_zero_mu,
    )
    dataloader = get_dataloader(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_algo="pair-candidates",
        batch_size=32,
        shuffle=False,
        drop_last=False,
    )

    predicted_mu_list = []
    flipped_predicted_mu_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Model"):
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

    predicted_mu_list = np.concatenate(predicted_mu_list, axis=0)
    flipped_predicted_mu_list = np.concatenate(flipped_predicted_mu_list, axis=0)

    def closest_anchor(pred_mu, flipped_mu):
        scores = [(a, abs(pred_mu - a) + abs(flipped_mu - (1 - a))) for a in anchors]
        return min(scores, key=lambda x: x[1])  # (anchor, score)

    anchor_score_list = [
        (
            i,
            *closest_anchor(predicted_mu_list[i], flipped_predicted_mu_list[i]),
        )  # (index, anchor, score)
        for i in range(len(pair_candidates))
    ]

    anchor_score_list.sort(key=lambda x: x[2])

    final_feedbacks_classifier = [
        (pair_candidates[i][0], pair_candidates[i][1], anchor)
        for i, anchor, _ in anchor_score_list[:10000]
    ]

    save_feedbacks_npz(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_name=f"{train_pair_algo}-aug-classifier",
        feedbacks=feedbacks_flipped + final_feedbacks_classifier,
    )

    save_feedbacks_npz(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_name=f"{train_pair_algo}-aug-classifier-init",
        feedbacks=feedbacks_flipped + final_feedbacks_classifier,
    )

    # 5. re-learn reward
    for k in range(3):
        train_reward_model(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo="ternary-500-aug-mr",
            reward_model_algo="MR-linear",
            reward_model_tag=f"{k:02d}",
            train_from_existing=True,
            num_epoch=100,
        )

        train_reward_model(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo="ternary-500-aug-classifier",
            reward_model_algo="MR-linear",
            reward_model_tag=f"{k:02d}",
            train_from_existing=True,
            num_epoch=100,
        )

        train_reward_model(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo="ternary-500-aug-mr-init",
            reward_model_algo="MR-linear",
            reward_model_tag=f"{k:02d}",
            num_epoch=100,
        )

        train_reward_model(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo="ternary-500-aug-classifier-init",
            reward_model_algo="MR-linear",
            reward_model_tag=f"{k:02d}",
            num_epoch=100,
        )
