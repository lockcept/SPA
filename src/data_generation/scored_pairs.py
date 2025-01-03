import numpy as np
import torch

from data_generation.score_rnn import RNNModel
from data_generation.score_lstm import LSTMModel
from data_generation.utils import generate_pairs_from_indices
from data_loading import get_dataloader, load_pair
from utils import get_score_model_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fill_feedback_from_pairs(dataset, pairs, model):
    """
    fill feedback in dataset with model

    Args:
        dataset: dict
        pairs: list of tuples ((int, int), (int, int))
        model: torch.nn.Module

    Returns:
        np array of ((int, int), (int, int), float)
    """

    # evaluate model with result data
    observations = dataset["observations"]
    actions = dataset["actions"]

    results = []
    model.eval()

    with torch.no_grad():
        for s0, s1 in pairs:
            s0_obs = observations[s0[0] : s0[1]]
            s0_act = actions[s0[0] : s0[1]]
            s1_obs = observations[s1[0] : s1[1]]
            s1_act = actions[s1[0] : s1[1]]

            s0_state = np.concatenate([s0_obs, s0_act], axis=1)
            s1_state = np.concatenate([s1_obs, s1_act], axis=1)

            s0_tensor = torch.tensor(s0_state, dtype=torch.float32).to(device)
            s1_tensor = torch.tensor(s1_state, dtype=torch.float32).to(device)

            score_0 = model(s0_tensor).item()
            score_1 = model(s1_tensor).item()

            mu = 1 / (1 + np.exp(score_0 - score_1))
            results.append((s0, s1, mu))

    return np.array(
        results,
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
        # train rnn with train data
        model, optimizer = RNNModel.initialize(
            config={"obs_dim": obs_dim, "act_dim": act_dim}, path=model_path
        )
    elif score_model == "lstm":
        # train lstm with train data
        model, optimizer = LSTMModel.initialize(
            config={"obs_dim": obs_dim, "act_dim": act_dim}, path=model_path
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

    if score_model == "rnn":
        best_model, _ = RNNModel.initialize(
            config={"obs_dim": obs_dim, "act_dim": act_dim},
            path=model_path,
            skip_if_exists=False,
        )
    elif score_model == "lstm":
        best_model, _ = LSTMModel.initialize(
            config={"obs_dim": obs_dim, "act_dim": act_dim},
            path=model_path,
            skip_if_exists=False,
        )
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
    train_feedback_pairs = fill_feedback_from_pairs(dataset, train_pairs, best_model)
    val_feedback_pairs = fill_feedback_from_pairs(dataset, val_pairs, best_model)
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
                dataset, aug_train_pairs, best_model
            )
            new_train_feedback_pairs = np.concatenate(
                [train_feedback_pairs, aug_train_feedback_pairs],
                axis=0,
            )
        elif aug == "10000-0.5":
            aug_train_pairs = generate_pairs_from_indices(traj_set, 10000, 25)
            aug_train_feedback_pairs = fill_feedback_from_pairs(
                dataset, aug_train_pairs, best_model
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
        elif aug == "200000":
            aug_train_pairs = generate_pairs_from_indices(traj_set, 200000, 25)
            aug_train_feedback_pairs = fill_feedback_from_pairs(
                dataset, aug_train_pairs, best_model
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
