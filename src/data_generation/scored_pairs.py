from itertools import combinations
from random import sample
import numpy as np
import torch

from data_generation.raw_pairs import save_raw_pairs
from data_generation.score.score_encoder import EncoderModel
from data_generation.score.score_rnn import RNNModel
from data_generation.score.score_lstm import LSTMModel
from data_generation.utils import generate_pairs_from_indices, save_feedbacks_npz
from data_loading import (
    get_dataloader,
    load_pair,
    load_dataset,
    process_pairs,
    get_dataloader_from_processed_data,
)
from utils import get_score_model_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fill_score_from_pairs(dataset, pairs, models):
    """
    Fill scores in dataset using multiple models and average their mu values.

    Args:
        dataset: dict
        pairs: list of tuples ((int, int), (int, int))
        models: list of torch.nn.Module
        linear_loss: bool, optional
            If True, use linear loss for mu calculation. Default is False.

    Returns:
        np array of ((int, int), float): scores.
    """

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

    score_results = []

    with torch.no_grad():
        for batch in dataloader:
            (
                s0_obs_batch,
                s0_act_batch,
                s1_obs_batch,
                s1_act_batch,
                _,
                mask0_batch,
                mask1_batch,
            ) = [x.to(device) for x in batch]

            s0_batch = torch.cat((s0_obs_batch, s0_act_batch), dim=-1)
            s1_batch = torch.cat((s1_obs_batch, s1_act_batch), dim=-1)

            lengths_s0 = (1 - mask0_batch.squeeze(dim=-1)).sum(dim=1)
            lengths_s1 = (1 - mask1_batch.squeeze(dim=-1)).sum(dim=1)

            scores_0_batch = []
            scores_1_batch = []

            for model in models:
                model.eval()

                # Calculate scores
                scores_0 = model(s0_batch, lengths_s0).cpu().numpy()
                scores_1 = model(s1_batch, lengths_s1).cpu().numpy()

                scores_0_batch.append(scores_0)
                scores_1_batch.append(scores_1)

            scores_0_batch = np.array(scores_0_batch)
            scores_1_batch = np.array(scores_1_batch)

            scores_0_batch = np.mean(scores_0_batch, axis=0)
            scores_1_batch = np.mean(scores_1_batch, axis=0)

            score_results.extend(
                [
                    score
                    for pair in zip(scores_0_batch, scores_1_batch)
                    for score in pair
                ]
            )

    all_trajs = [traj for pair in pairs for traj in pair]

    return zip(all_trajs, score_results)


def fill_feedback_from_pairs(dataset, pairs, models, linear_loss=False):
    """
    Fill feedback in dataset using multiple models and average their mu values.
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

    mu_results = []
    std_dev_results = []

    with torch.no_grad():
        for batch in dataloader:
            (
                s0_obs_batch,
                s0_act_batch,
                s1_obs_batch,
                s1_act_batch,
                _,
                mask0_batch,
                mask1_batch,
            ) = [x.to(device) for x in batch]

            s0_batch = torch.cat((s0_obs_batch, s0_act_batch), dim=-1)
            s1_batch = torch.cat((s1_obs_batch, s1_act_batch), dim=-1)

            lengths_s0 = (1 - mask0_batch.squeeze(dim=-1)).sum(dim=1)
            lengths_s1 = (1 - mask1_batch.squeeze(dim=-1)).sum(dim=1)

            batch_mu_results = []  # Collect mu values for the batch from all models

            for model in models:
                model.eval()

                # Calculate scores
                scores_0 = model(s0_batch, lengths_s0).cpu().numpy()
                scores_1 = model(s1_batch, lengths_s1).cpu().numpy()

                # Calculate mu for this model
                if linear_loss:
                    mu_batch = scores_1 / (scores_0 + scores_1 + 1e-6)
                else:
                    mu_batch = 1 / (1 + np.exp(scores_0 - scores_1))

                mu_batch = np.squeeze(mu_batch, axis=-1)
                batch_mu_results.append(mu_batch)

            batch_mu_results = np.array(batch_mu_results)

            avg_mu_batch = np.mean(batch_mu_results, axis=0)
            std_mu_batch = np.std(batch_mu_results, axis=0, ddof=1)

            mu_results = np.concatenate((mu_results, avg_mu_batch))
            std_dev_results = np.concatenate((std_dev_results, std_mu_batch))

    mu_array = np.array(
        [(s0, s1, mu) for (s0, s1), mu in zip(pairs, mu_results)],
        dtype=[
            ("s0", "i4", (2,)),
            ("s1", "i4", (2,)),
            ("mu", "f"),
        ],
    )

    return mu_array, std_dev_results


def train_model(
    env_name,
    exp_name,
    num_epochs,
    pair_algo,
    score_model,
    ensemble_num,
    max_ensemble_num,
):
    if max_ensemble_num > 1:
        # split train and val pairs
        dataset = load_dataset(env_name)
        pairs = load_pair(
            env_name=env_name, exp_name=exp_name, pair_type="train", pair_algo=pair_algo
        )

        chunk_size = len(pairs) // max_ensemble_num
        remainder = len(pairs) % max_ensemble_num

        start_idx = sum(
            chunk_size + (1 if i < remainder else 0) for i in range(ensemble_num)
        )
        end_idx = start_idx + chunk_size + (1 if ensemble_num < remainder else 0)

        train_pairs = np.concatenate((pairs[:start_idx], pairs[end_idx:]), axis=0)
        val_pairs = pairs[start_idx:end_idx]

        train_data_loader = get_dataloader_from_processed_data(
            process_pairs(dataset=dataset, pair=train_pairs)
        )

        val_data_loader = get_dataloader_from_processed_data(
            process_pairs(dataset=dataset, pair=val_pairs)
        )

    else:
        train_data_loader = get_dataloader(
            env_name=env_name, exp_name=exp_name, pair_type="train", pair_algo=pair_algo
        )

        val_data_loader = get_dataloader(
            env_name=env_name, exp_name=exp_name, pair_type="val", pair_algo=pair_algo
        )

    obs_dim, act_dim = train_data_loader.dataset.get_dimensions()

    model_path = get_score_model_path(
        env_name, exp_name, pair_algo, score_model, ensemble_num
    )

    if score_model == "rnn":
        model, optimizer = RNNModel.initialize(
            config={"obs_dim": obs_dim, "act_dim": act_dim}, path=model_path
        )
    elif score_model == "lstm.exp":
        model, optimizer = LSTMModel.initialize(
            config={"obs_dim": obs_dim, "act_dim": act_dim}, path=model_path
        )
    elif score_model == "lstm.linear":
        model, optimizer = LSTMModel.initialize(
            config={"obs_dim": obs_dim, "act_dim": act_dim},
            path=model_path,
            linear_loss=True,
        )
    elif score_model == "encoder":
        model, optimizer = EncoderModel.initialize(
            config={"obs_dim": obs_dim, "act_dim": act_dim},
            path=model_path,
        )

    else:
        model = None
        optimizer = None

    if model is None:
        print(f"Model {pair_algo} is not supported")
        return

    model.train_model(
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        optimizer=optimizer,
        num_epochs=num_epochs,
    )


def generate_score_pairs(
    dataset,
    env_name,
    exp_name,
    num_epochs,
    pair_algo,
    score_model,
    aug_list,
    traj_set,
    ensemble_size=1,
):
    """
    learn score model and save score pairs
    """

    for ensemble_num in range(ensemble_size):
        train_model(
            env_name=env_name,
            exp_name=exp_name,
            num_epochs=num_epochs,
            pair_algo=pair_algo,
            score_model=score_model,
            ensemble_num=ensemble_num,
            max_ensemble_num=ensemble_size,
        )

    obs_dim, act_dim = dataset["observations"].shape[1], dataset["actions"].shape[1]

    best_models = []

    for ensemble_num in range(ensemble_size):
        # generate pairs
        model_path = get_score_model_path(
            env_name, exp_name, pair_algo, score_model, ensemble_num
        )

        linear_loss = False

        if score_model == "rnn":
            best_model, _ = RNNModel.initialize(
                config={"obs_dim": obs_dim, "act_dim": act_dim},
                path=model_path,
                skip_if_exists=False,
            )
        elif score_model == "lstm.exp":
            best_model, _ = LSTMModel.initialize(
                config={"obs_dim": obs_dim, "act_dim": act_dim},
                path=model_path,
                skip_if_exists=False,
            )
        elif score_model == "lstm.linear":
            best_model, _ = LSTMModel.initialize(
                config={"obs_dim": obs_dim, "act_dim": act_dim},
                path=model_path,
                skip_if_exists=False,
                linear_loss=True,
            )
            linear_loss = True
        elif score_model == "encoder":
            best_model, _ = EncoderModel.initialize(
                config={"obs_dim": obs_dim, "act_dim": act_dim},
                path=model_path,
                skip_if_exists=False,
            )
        else:
            best_model = None

        best_models.append(best_model)

    train_pairs_with_mu = load_pair(
        env_name=env_name, exp_name=exp_name, pair_type="train", pair_algo=pair_algo
    )
    val_pairs_with_mu = load_pair(
        env_name=env_name, exp_name=exp_name, pair_type="val", pair_algo=pair_algo
    )

    train_pairs = [(s0, s1) for s0, s1, _ in train_pairs_with_mu]
    val_pairs = [(s0, s1) for s0, s1, _ in val_pairs_with_mu]

    # fill feedback in pairs
    train_feedback_pairs, _ = fill_feedback_from_pairs(
        dataset, train_pairs, best_models, linear_loss
    )
    val_feedback_pairs, _ = fill_feedback_from_pairs(
        dataset, val_pairs, best_models, linear_loss
    )

    save_feedbacks_npz(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_name=f"{score_model}-{pair_algo}",
        feedbacks=train_feedback_pairs,
    )
    save_feedbacks_npz(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="val",
        pair_name=f"{score_model}-{pair_algo}",
        feedbacks=val_feedback_pairs,
    )

    for aug in aug_list:
        if aug == "10000":
            try:
                loaded_pairs = load_pair(
                    env_name=env_name,
                    exp_name=exp_name,
                    pair_type="train",
                    pair_algo="raw_10000",
                )
                aug_train_pairs = [(pair["s0"], pair["s1"]) for pair in loaded_pairs]
            except FileNotFoundError:
                aug_train_pairs = generate_pairs_from_indices(
                    dataset, traj_set, 10000, 25
                )
                save_raw_pairs(
                    env_name=env_name,
                    exp_name=exp_name,
                    pair_type="train",
                    pairs=aug_train_pairs,
                    raw_name="raw_10000",
                )

            aug_train_feedback_pairs, _ = fill_feedback_from_pairs(
                dataset, aug_train_pairs, best_models, linear_loss
            )

            new_train_feedback_pairs = np.concatenate(
                [train_feedback_pairs, aug_train_feedback_pairs],
                axis=0,
            )
        elif aug == "50000":
            try:
                loaded_pairs = load_pair(
                    env_name=env_name,
                    exp_name=exp_name,
                    pair_type="train",
                    pair_algo="raw_50000",
                )
                aug_train_pairs = [(pair["s0"], pair["s1"]) for pair in loaded_pairs]
            except FileNotFoundError:
                aug_train_pairs = generate_pairs_from_indices(
                    dataset, traj_set, 50000, 25
                )
                save_raw_pairs(
                    env_name=env_name,
                    exp_name=exp_name,
                    pair_type="train",
                    pairs=aug_train_pairs,
                    raw_name="raw_50000",
                )

            aug_train_feedback_pairs, _ = fill_feedback_from_pairs(
                dataset, aug_train_pairs, best_models, linear_loss
            )

            new_train_feedback_pairs = np.concatenate(
                [train_feedback_pairs, aug_train_feedback_pairs],
                axis=0,
            )
        elif aug == "200000.5000":
            try:
                loaded_pairs = load_pair(
                    env_name=env_name,
                    exp_name=exp_name,
                    pair_type="train",
                    pair_algo="raw_200000",
                )
                aug_train_pairs = [(pair["s0"], pair["s1"]) for pair in loaded_pairs]
            except FileNotFoundError:
                aug_train_pairs = generate_pairs_from_indices(
                    dataset, traj_set, 200000, 25
                )

                save_raw_pairs(
                    env_name=env_name,
                    exp_name=exp_name,
                    pair_type="train",
                    pairs=aug_train_pairs,
                    raw_name="raw_200000",
                )

            aug_train_feedback_pairs, _ = fill_feedback_from_pairs(
                dataset, aug_train_pairs, best_models, linear_loss
            )

            distances = np.abs(aug_train_feedback_pairs["mu"] - 0.5)
            sorted_indices = np.argsort(-distances)
            top_feedback_pairs = aug_train_feedback_pairs[sorted_indices[:5000]]

            # save pairs for other experiments
            top_pairs = [(s0, s1) for s0, s1, _ in top_feedback_pairs]
            save_raw_pairs(
                env_name=env_name,
                exp_name=exp_name,
                pair_type="train",
                pairs=top_pairs,
                raw_name="raw_5000",
            )

            new_train_feedback_pairs = np.concatenate(
                [train_feedback_pairs, top_feedback_pairs],
                axis=0,
            )
        elif aug == "nC2.smoothing":
            train_pairs_with_score = fill_score_from_pairs(
                dataset, train_pairs, best_models
            )
            aug_feedback_pairs = []

            sample_size = 50000
            all_combinations = list(combinations(train_pairs_with_score, 2))
            random_combinations = sample(all_combinations, sample_size)

            for s0, s1 in random_combinations:
                mu = 1 / (1 + np.exp((s0[1] - s1[1]) / 5))
                aug_feedback_pairs.append((s0[0], s1[0], mu))

            aug_feedback_pairs = np.array(
                aug_feedback_pairs,
                dtype=[
                    ("s0", "i4", (2,)),
                    ("s1", "i4", (2,)),
                    ("mu", "f"),
                ],
            )

            new_train_feedback_pairs = np.concatenate(
                [train_feedback_pairs, aug_feedback_pairs],
                axis=0,
            )
        elif aug == "nC2":
            train_pairs_with_score = fill_score_from_pairs(
                dataset, train_pairs, best_models
            )
            aug_feedback_pairs = []

            for s0, s1 in combinations(train_pairs_with_score, 2):
                if np.abs(s0[1] - s1[1]) > 5:
                    aug_feedback_pairs.append(
                        (s0[0], s1[0], 1.0 if s1[1] > s0[1] else 0.0)
                    )

            aug_feedback_pairs = np.array(
                aug_feedback_pairs,
                dtype=[
                    ("s0", "i4", (2,)),
                    ("s1", "i4", (2,)),
                    ("mu", "f"),
                ],
            )

            new_train_feedback_pairs = np.concatenate(
                [train_feedback_pairs, aug_feedback_pairs],
                axis=0,
            )
        elif aug == "certain":
            try:
                loaded_pairs = load_pair(
                    env_name=env_name,
                    exp_name=exp_name,
                    pair_type="train",
                    pair_algo="raw_200000",
                )
                aug_train_pairs = [(pair["s0"], pair["s1"]) for pair in loaded_pairs]
            except FileNotFoundError:
                aug_train_pairs = generate_pairs_from_indices(
                    dataset, traj_set, 200000, 25
                )

                save_raw_pairs(
                    env_name=env_name,
                    exp_name=exp_name,
                    pair_type="train",
                    pairs=aug_train_pairs,
                    raw_name="raw_200000",
                )

            _, std_dev = fill_feedback_from_pairs(
                dataset, train_pairs, best_models, linear_loss
            )

            std_dev_criteria = np.mean(std_dev)

            aug_train_feedback_pairs, aug_std_dev = fill_feedback_from_pairs(
                dataset, aug_train_pairs, best_models, linear_loss
            )

            filtered_pairs = [
                pair
                for pair, mu, std in zip(
                    aug_train_feedback_pairs,
                    aug_train_feedback_pairs["mu"],
                    aug_std_dev,
                )
                if std < std_dev_criteria * 0.5
            ]

            print(std_dev_criteria, len(filtered_pairs))

            aug_train_feedback_pairs = np.array(
                filtered_pairs, dtype=aug_train_feedback_pairs.dtype
            )

            new_train_feedback_pairs = np.concatenate(
                [train_feedback_pairs, aug_train_feedback_pairs],
                axis=0,
            )
        elif aug == "cutting":
            aug_train_pairs = []

            for s0, s1 in train_pairs:
                split_points_0 = []
                split_points_1 = []

                for i in range(0, 5 + 1):
                    # 내분점
                    split_points_0.append((s0[0] * (5 - i) + s0[1] * i) // 5)
                    split_points_1.append((s1[0] * (5 - i) + s1[1] * i) // 5)

                for i in range(5):
                    for j in range(5 - i):
                        start_index = j
                        end_index = j + i + 1

                        aug_train_pairs.append(
                            (
                                (
                                    split_points_0[start_index],
                                    split_points_0[end_index],
                                ),
                                (
                                    split_points_1[start_index],
                                    split_points_1[end_index],
                                ),
                            )
                        )

            new_train_feedback_pairs, _ = fill_feedback_from_pairs(
                dataset, aug_train_pairs, best_models, linear_loss
            )
        elif aug == "uncertain":
            try:
                loaded_pairs = load_pair(
                    env_name=env_name,
                    exp_name=exp_name,
                    pair_type="train",
                    pair_algo="raw_200000",
                )
                aug_train_pairs = [(pair["s0"], pair["s1"]) for pair in loaded_pairs]
            except FileNotFoundError:
                aug_train_pairs = generate_pairs_from_indices(
                    dataset, traj_set, 200000, 25
                )

                save_raw_pairs(
                    env_name=env_name,
                    exp_name=exp_name,
                    pair_type="train",
                    pairs=aug_train_pairs,
                    raw_name="raw_200000",
                )

            _, std_dev = fill_feedback_from_pairs(
                dataset, train_pairs, best_models, linear_loss
            )

            std_dev_criteria = np.mean(std_dev)

            aug_train_feedback_pairs, aug_std_dev = fill_feedback_from_pairs(
                dataset, aug_train_pairs, best_models, linear_loss
            )

            filtered_pairs = [
                pair
                for pair, mu, std in zip(
                    aug_train_feedback_pairs,
                    aug_train_feedback_pairs["mu"],
                    aug_std_dev,
                )
                if std > std_dev_criteria and (mu < 0.1 or mu > 0.9)
            ]

            print(std_dev_criteria, len(filtered_pairs))

            aug_train_feedback_pairs = np.array(
                filtered_pairs, dtype=aug_train_feedback_pairs.dtype
            )

            new_train_feedback_pairs = np.concatenate(
                [train_feedback_pairs, aug_train_feedback_pairs],
                axis=0,
            )

        elif aug == "test":
            aug_train_pairs = generate_pairs_from_indices(dataset, traj_set, 1000, 25)
            new_train_feedback_pairs, _ = fill_feedback_from_pairs(
                dataset, aug_train_pairs, best_models, linear_loss
            )
        else:
            new_train_feedback_pairs = train_feedback_pairs

        save_feedbacks_npz(
            env_name=env_name,
            exp_name=exp_name,
            pair_type="train",
            pair_name=f"{score_model}-aug-{aug}-{pair_algo}",
            feedbacks=new_train_feedback_pairs,
        )

        # val feedback pairs are same as before
        save_feedbacks_npz(
            env_name=env_name,
            exp_name=exp_name,
            pair_type="val",
            pair_name=f"{score_model}-aug-{aug}-{pair_algo}",
            feedbacks=val_feedback_pairs,
        )

    return
