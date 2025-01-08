import numpy as np
import torch

from data_generation.raw_pairs import save_raw_pairs
from data_generation.score_encoder import EncoderModel
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
                mu_batch = (scores_1 / (scores_0 + scores_1 + 1e-6)).squeeze()
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
    ensemble_num,
):
    train_data_loader = get_dataloader(
        env_name=env_name, exp_name=exp_name, pair_type="train", pair_algo=pair_algo
    )

    obs_dim, act_dim = train_data_loader.dataset.get_dimensions()

    val_data_loader = get_dataloader(
        env_name=env_name, exp_name=exp_name, pair_type="val", pair_algo=pair_algo
    )
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

    # Todo: use ensemble
    best_model = best_models[0]

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
        if aug == "5000-soft":
            aug_train_pairs = generate_pairs_from_indices(traj_set, 200000, 25)
            aug_train_feedback_pairs = fill_feedback_from_pairs(
                dataset, aug_train_pairs, best_model, linear_loss
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
        elif aug == "5000-hard":
            # Must be run after 5000-soft
            top_pairs = load_pair(
                env_name=env_name,
                exp_name=exp_name,
                pair_type="train",
                pair_algo="raw_5000",
            )

            aug_train_feedback_pairs = fill_feedback_from_pairs(
                dataset, aug_train_pairs, best_model, linear_loss
            )

            hard_pairs = []
            for s0, s1, mu in aug_train_feedback_pairs:
                if mu < 0.5:
                    hard_pairs.append((s0, s1, 0.0))
                else:
                    hard_pairs.append((s0, s1, 1.0))

            hard_pairs = np.array(
                hard_pairs,
                dtype=[
                    ("s0", "i4", (2,)),
                    ("s1", "i4", (2,)),
                    ("mu", "f"),
                ],
            )
            new_train_feedback_pairs = np.concatenate(
                [train_feedback_pairs, hard_pairs],
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
