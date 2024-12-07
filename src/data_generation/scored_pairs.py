import numpy as np
import torch

from data_generation.score_rnn import RNN
from data_loading import get_dataloader, load_pair

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fill_feedback_from_pairs(dataset, pairs, model):
    """
    fill feedback in dataset with model

    Args:
        dataset: dict
        pairs: list of tuples ((int, int), (int, int), float)
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
        for s0, s1, _ in pairs:
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


def generate_score_pairs(
    dataset,
    env_name,
    exp_name,
    num_epochs,
    pair_algo,
    score_model,
):
    """
    learn score model and save score pairs
    """

    train_data_loader = get_dataloader(
        env_name=env_name, exp_name=exp_name, pair_type="train", pair_algo=pair_algo
    )

    obs_dim, act_dim = train_data_loader.dataset.get_dimensions()

    val_data_loader = get_dataloader(
        env_name=env_name, exp_name=exp_name, pair_type="val", pair_algo=pair_algo
    )

    if score_model == "rnn":
        model_path = f"model/{env_name}/{exp_name}/score/rnn-{pair_algo}.pth"
        # train rnn with train data
        model, optimizer = RNN.initialize(
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

    best_model, _ = RNN.initialize(
        config={"obs_dim": obs_dim, "act_dim": act_dim},
        path=model_path,
        skip_if_exists=False,
    )

    # fill feedback in pairs
    train_pairs = load_pair(
        env_name=env_name, exp_name=exp_name, pair_type="train", pair_algo=pair_algo
    )["data"]
    val_pairs = load_pair(
        env_name=env_name, exp_name=exp_name, pair_type="val", pair_algo=pair_algo
    )["data"]
    train_pairs = fill_feedback_from_pairs(dataset, train_pairs, best_model)
    val_pairs = fill_feedback_from_pairs(dataset, val_pairs, best_model)

    # save pairs
    np.savez(
        f"pair/{env_name}/{exp_name}/train/{score_model}-{pair_algo}.npz",
        data=train_pairs,
    )
    np.savez(
        f"pair/{env_name}/{exp_name}/val/{score_model}-{pair_algo}.npz",
        data=val_pairs,
    )

    return
