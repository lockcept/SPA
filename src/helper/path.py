import os
from typing import Literal


def make_dir_from_path(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


def get_pair_path(
    env_name, exp_name, pair_type: Literal["train", "val", "test"], pair_algo
):
    path = f"pair/{env_name}/{exp_name}/{pair_type}/{pair_algo}.npz"
    make_dir_from_path(path)
    return path
