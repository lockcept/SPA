import numpy as np
import torch

from data_generation.score_rnn import RNNModel
from data_generation.score_lstm import LSTMModel
from data_generation.utils import generate_pairs_from_indices
from data_loading import get_dataloader, load_pair
from data_loading.load_data import process_pairs
from data_loading.preference_dataloader import get_dataloader_from_processed_data
from utils import get_score_model_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fill_feedback_from_pairs(dataset, pairs, model, linear_loss=False):
    """
    fill feedback in dataset with model

    Args:
        dataset: dict
        pairs: list of tuples ((int, int), (int, int))
        model: torch.nn.Module

    Returns:
        np array of ((int, int), (int, int), float)
    """

    pairs_with_zero_mu = np.array(
        [(s0, s1, 0.0) for s0, s1 in pairs],
        dtype=[
            ("s0", "i4", (2,)),
            ("s1", "i4", (2,)),
            ("mu", "f"),
        ],
    )

    # evaluate model with result data
    processed_data = process_pairs(dataset, pairs_with_zero_mu)
    dataloader = get_dataloader_from_processed_data(
        processed_data, shuffle=False, drop_last=False
    )

    mu_results = []
    model.eval()

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

            lengths_s0 = (1 - mask0_batch.squeeze()).sum(dim=1)
            lengths_s1 = (1 - mask1_batch.squeeze()).sum(dim=1)

            scores_0 = model(s0_batch, lengths_s0).cpu().numpy()
            scores_1 = model(s1_batch, lengths_s1).cpu().numpy()

            if linear_loss:
                mu_batch = (scores_1 / (scores_0 + scores_1)).squeeze()
            else:
                mu_batch = 1 / (1 + np.exp(scores_0 - scores_1)).squeeze()

            mu_results = np.concatenate((mu_results, mu_batch))

    return np.array(
        [(s0, s1, mu) for (s0, s1), mu in zip(pairs, mu_results)],
        dtype=[
            ("s0", "i4", (2,)),
            ("s1", "i4", (2,)),
            ("mu", "f"),
        ],
    )


def train_model(
    env_name,
    exp_name,
    num_epochs,
    pair_algo,
    score_model,
):
    train_data_loader = get_dataloader(
        env_name=env_name, exp_name=exp_name, pair_type="train", pair_algo=pair_algo
    )

    obs_dim, act_dim = train_data_loader.dataset.get_dimensions()

    val_data_loader = get_dataloader(
        env_name=env_name, exp_name=exp_name, pair_type="val", pair_algo=pair_algo
    )
    model_path = get_score_model_path(env_name, exp_name, pair_algo, score_model)

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
):
    """
    learn score model and save score pairs
    """
    train_model(
        env_name=env_name,
        exp_name=exp_name,
        num_epochs=num_epochs,
        pair_algo=pair_algo,
        score_model=score_model,
    )

    obs_dim, act_dim = dataset["observations"].shape[1], dataset["actions"].shape[1]

    # generate pairs
    model_path = get_score_model_path(env_name, exp_name, pair_algo, score_model)

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

    else:
        best_model = None

    train_pairs_with_mu = load_pair(
        env_name=env_name, exp_name=exp_name, pair_type="train", pair_algo=pair_algo
    )["data"]
    val_pairs_with_mu = load_pair(
        env_name=env_name, exp_name=exp_name, pair_type="val", pair_algo=pair_algo
    )["data"]

    train_pairs = [(s0, s1) for s0, s1, _ in train_pairs_with_mu]
    val_pairs = [(s0, s1) for s0, s1, _ in val_pairs_with_mu]

    # fill feedback in pairs
    train_feedback_pairs = fill_feedback_from_pairs(
        dataset, train_pairs, best_model, linear_loss
    )
    val_feedback_pairs = fill_feedback_from_pairs(
        dataset, val_pairs, best_model, linear_loss
    )
    np.savez(
        f"pair/{env_name}/{exp_name}/train/{score_model}-{pair_algo}.npz",
        data=train_feedback_pairs,
    )
    np.savez(
        f"pair/{env_name}/{exp_name}/val/{score_model}-{pair_algo}.npz",
        data=val_feedback_pairs,
    )

    for aug in aug_list:
        if aug == "10000":
            aug_train_pairs = generate_pairs_from_indices(traj_set, 10000, 25)
            aug_train_feedback_pairs = fill_feedback_from_pairs(
                dataset, aug_train_pairs, best_model, linear_loss
            )
            new_train_feedback_pairs = np.concatenate(
                [train_feedback_pairs, aug_train_feedback_pairs],
                axis=0,
            )
        elif aug == "10000-0.5":
            aug_train_pairs = generate_pairs_from_indices(traj_set, 10000, 25)
            aug_train_feedback_pairs = fill_feedback_from_pairs(
                dataset, aug_train_pairs, best_model, linear_loss
            )
            new_train_feedback_pairs = np.concatenate(
                [train_feedback_pairs, aug_train_feedback_pairs],
                axis=0,
            )
            mu_values = new_train_feedback_pairs["mu"]
            mu_values = np.where(mu_values < 0.4, 0, mu_values)
            mu_values = np.where(mu_values > 0.6, 1, mu_values)
            mu_values = np.where((mu_values != 0) & (mu_values != 1), 0.5, mu_values)
            new_train_feedback_pairs["mu"] = mu_values
        elif aug == "10000-50":
            total_cnt = 200000
            aug_train_pairs = generate_pairs_from_indices(traj_set, total_cnt, 50)
            aug_train_pairs_head = []
            aug_train_pairs_tail = []
            for s0, s1 in aug_train_pairs:
                i0, e0 = s0
                i1, e1 = s1
                m0 = (i0 + e0) // 2
                m1 = (i1 + e1) // 2
                aug_train_pairs_head.append(((i0, m0), (i1, m1)))
                aug_train_pairs_tail.append(((m0, e0), (m1, e1)))

            aug_train_feedback_pairs = fill_feedback_from_pairs(
                dataset, aug_train_pairs, best_model, linear_loss
            )
            aug_train_feedback_pairs_head = fill_feedback_from_pairs(
                dataset, aug_train_pairs_head, best_model, linear_loss
            )
            aug_train_feedback_pairs_tail = fill_feedback_from_pairs(
                dataset, aug_train_pairs_tail, best_model, linear_loss
            )

            aug_valid_feedback_pairs = []

            mu_diffs = []

            for i in range(total_cnt):
                is_s1_head_better = aug_train_feedback_pairs_head[i]["mu"] > 0.5
                is_s1_tail_better = aug_train_feedback_pairs_tail[i]["mu"] > 0.5

                if is_s1_head_better != is_s1_tail_better:
                    mu_diff = np.abs(aug_train_feedback_pairs[i]["mu"] - 0.5)
                    mu_diffs.append((i, mu_diff))

            mu_diffs = np.array(mu_diffs, dtype=[("index", "i4"), ("mu_diff", "f4")])

            top_indices = mu_diffs[np.argsort(-mu_diffs["mu_diff"])]["index"][:10000]

            for idx in top_indices:
                aug_valid_feedback_pairs.append(aug_train_feedback_pairs[idx])
                aug_valid_feedback_pairs.append(aug_train_feedback_pairs_head[idx])
                aug_valid_feedback_pairs.append(aug_train_feedback_pairs_tail[idx])

            print(f"Augmented {len(aug_valid_feedback_pairs)} pairs")

            new_train_feedback_pairs = np.concatenate(
                [train_feedback_pairs, aug_valid_feedback_pairs],
                axis=0,
            )
        elif aug == "200000":
            aug_train_pairs = generate_pairs_from_indices(traj_set, 200000, 25)
            aug_train_feedback_pairs = fill_feedback_from_pairs(
                dataset, aug_train_pairs, best_model, linear_loss
            )

            distances = np.abs(aug_train_feedback_pairs["mu"] - 0.5)
            sorted_indices = np.argsort(-distances)
            top_10000_pairs = aug_train_feedback_pairs[sorted_indices[:10000]]

            new_train_feedback_pairs = np.concatenate(
                [train_feedback_pairs, top_10000_pairs],
                axis=0,
            )

        else:
            new_train_feedback_pairs = train_feedback_pairs

        np.savez(
            f"pair/{env_name}/{exp_name}/train/{score_model}-aug-{aug}-{pair_algo}.npz",
            data=new_train_feedback_pairs,
        )

        # val feedback pairs are same as before
        np.savez(
            f"pair/{env_name}/{exp_name}/val/{score_model}-aug-{aug}-{pair_algo}.npz",
            data=val_feedback_pairs,
        )

    return
