import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data_generation.full_scripted_teacher import get_pairs_by_mu_type
from data_loading.preference_dataloader import PreferenceDataset, get_dataloader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from data_generation.score_rnn import RNN
from data_generation.utils import extract_trajectory_indices
from data_loading.load_data import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_pairs(env_name, pair, pair_algo, pair_data):
    save_path = f"pair/{env_name}/{pair}_score-{pair_algo}.npz"
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.savez(save_path, data=pair_data)
    print(f"Preference pairs saved at {save_path}")


def generate_score_pairs(env_name, pair_name_base, num_pairs, pair_algos=["rnn"]):
    for pair_algo in pair_algos:
        save_path = f"pair/{env_name}/{pair_name_base}-score-{pair_algo}.npz"
        is_already_exist = os.path.exists(save_path)
        if is_already_exist:
            print(f"Pair already exists at {save_path}, cancel generating")
            return

    train_val_ratio = 0.8

    # get all trajectories
    dataset = load_dataset(env_name=env_name)
    indices = extract_trajectory_indices(dataset)
    np.random.shuffle(indices)

    selected = indices[: num_pairs * 2]
    remaining = indices[num_pairs * 2 :]

    split_point = int(len(remaining) * train_val_ratio)
    train_set = remaining[:split_point]
    val_set = remaining[split_point:]

    def pair_elements(arr):
        if len(arr) % 2 != 0:
            arr = arr[:-1]
        return [(arr[i], arr[i + 1]) for i in range(0, len(arr), 2)]

    result_pairs = pair_elements(selected)
    train_pairs = pair_elements(train_set)
    val_pairs = pair_elements(val_set)

    def get_dataloader_from_pairs(pairs):
        preference_pairs = []

        for s0, s1 in pairs:
            r0 = np.sum(dataset["rewards"][s0[0] : s0[1]])
            r1 = np.sum(dataset["rewards"][s1[0] : s1[1]])
            preference_pairs.append((s0, s1, r0, r1))

        preference_pairs_np = np.array(
            preference_pairs,
            dtype=[
                ("s0", "i4", (2,)),
                ("s1", "i4", (2,)),
                ("rewards_0", "O"),
                ("rewards_1", "O"),
            ],
        )

        pair_data = get_pairs_by_mu_type(
            mu_type="binary",
            pair_data=preference_pairs_np,
        )

        save_pair_name = f"{pair_name_base}_score-temp"
        save_path = f"pair/{env_name}/{save_pair_name}.npz"
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.savez(save_path, data=pair_data)
        print(f"Preference pairs saved at {save_path}")

        return get_dataloader(env_name=env_name, pair_name=save_pair_name)

    train_data_loader, obs_dim, act_dim = get_dataloader_from_pairs(train_pairs)
    val_data_loader, _, _ = get_dataloader_from_pairs(val_pairs)

    num_epochs = 500

    for pair_algo in pair_algos:
        if pair_algo == "rnn":
            model_path = f"model/{env_name}/score/{pair_name_base}_{pair_algo}.pth"
            # train rnn with train data
            model, optimizer = RNN.initialize(
                config={"obs_dim": obs_dim, "act_dim": act_dim}, path=model_path
            )

        if model is None:
            print(f"Model {pair_algo} is not supported")
            continue

        model.train_model(
            train_data_loader=train_data_loader,
            val_data_loader=val_data_loader,
            optimizer=optimizer,
            num_epochs=num_epochs,
        )

        pairs = []

        # evaluate model with result data
        observations = dataset["observations"]
        actions = dataset["actions"]

        for s0, s1 in result_pairs:
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
            pairs.append((s0, s1, mu))

        pairs_np = np.array(
            pairs,
            dtype=[
                ("s0", "i4", (2,)),
                ("s1", "i4", (2,)),
                ("mu", "f"),
            ],
        )

        # save pairs
        np.savez(
            f"pair/{env_name}/{pair_name_base}_score-{pair_algo}.npz", data=pairs_np
        )

    return
