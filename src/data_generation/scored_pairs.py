import os
import sys

import numpy as np
from torch.utils.data import DataLoader, Dataset

from data_loading.preference_dataloader import PreferenceDataset, get_dataloader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from data_generation.score_rnn import RNN
from data_generation.utils import extract_trajectory_indices
from data_loading.load_data import (
    get_processed_data_from_dataset_and_pair,
    load_dataset,
)


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

    obs_dim, act_dim = None, None

    def get_data_loader_from_pairs(pairs):
        processed_data = get_processed_data_from_dataset_and_pair(dataset, pairs)
        dataset = PreferenceDataset(processed_data)
        if (obs_dim, act_dim) == (None, None):
            obs_dim, act_dim = dataset.get_dimensions()
        return DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            drop_last=True,
        )

    train_data_loader = get_data_loader_from_pairs(train_pairs)
    val_data_loader = get_data_loader_from_pairs(val_pairs)
    result_data_loader = get_data_loader_from_pairs(result_pairs)

    val_data_loader = None
    num_epochs = 500

    for pair_algo in pair_algos:
        if pair_algo == "rnn":
            model_path = f"model/{env_name}/score/{pair_name_base}_{pair_algo}.pth"
            # train rnn with train data
            model, optimizer = RNN.initialize(
                config={"obs_dim": obs_dim, "act_dim": act_dim}, path=model_path
            )

        if model is not None:
            model.train_model(
                train_loader=train_data_loader,
                val_loader=val_data_loader,
                optimizer=optimizer,
                num_epochs=num_epochs,
            )

        # evaluate model with result data
        # save pairs

    return
