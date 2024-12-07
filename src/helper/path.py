import os
from typing import Literal


def make_dir_from_path(path):
    """
    Create directory from path
    """
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_pair_path(
    env_name, exp_name, pair_type: Literal["train", "val", "test"], pair_algo
):
    """
    Return path of pair file
    """
    path = f"pair/{env_name}/{exp_name}/{pair_type}/{pair_algo}.npz"
    make_dir_from_path(path)
    return path


def get_reward_model_path(
    env_name,
    exp_name,
    pair_algo,
    reward_model_algo: Literal["MR", "MR-linear"],
    reward_model_tag,
):
    """
    Return path of reward model file
    """
    path = f"model/{env_name}/{exp_name}/reward/{pair_algo}/{reward_model_algo}-{reward_model_tag}.pth"
    make_dir_from_path(path)
    return path


def get_new_dataset_path(env_name, exp_name, pair_algo, reward_model_algo):
    """
    Return path of new dataset file
    """
    path = f"dataset/{env_name}/{exp_name}/{pair_algo}/{reward_model_algo}.npz"
    make_dir_from_path(path)
    return path
