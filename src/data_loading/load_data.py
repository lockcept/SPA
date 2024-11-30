import os
import random
import numpy as np


metaworld_ids = {
    "box-close-v2": "1yva0VXvnnyMOCLfWstj5q0TK-oi3Rt65",
    "button-press-topdown-v2": "1McVLA6KWi6KWJOI0lpIQ3dm60dxF_SpL",
    "button-press-topdown-wall-v2": "1dxdzUom2NsKFKkv0nrD2590UojoMYFmf",
    "dial-turn-v2": "11lC_Ihn55Lruv-GDPe7lrkLNd0pxSCy4",
    "drawer-open-v2": "1ixXsiscRFFypnQBLRkb2WGYSTOwzU7XS",
    "hammer-v2": "1QLQUTxlt9kFig6kzcAA6tm8oByLAe07v",
    "handle-pull-side-v2": "16wrAL6708u8aODyuqAHgEjVmrHDuIUzY",
    "lever-pull-v2": "17kLkqfAX3OPefb1bsuVXSMdglDmtYwuJ",
    "peg-insert-side-v2": "1Edy5_RPsoKW3gIKH4D7tNjMgDCWN-iqI",
    "peg-unplug-side-v2": "1Elc7IU-J8D2IxTc8GnussLjFLIUW85SC",
    "sweep-into-v2": "1G3VghYKH5Mm2XHM69-oMl6uDNx7fdCxC",
    "sweep-v2": "1u7f5WZYQlqXSxyJGI56kWlafYluFrgJb",
}


class MetaworldEnvWrapper:
    def __init__(self, env_gen):
        self.env = None
        self.env_gen = env_gen

    def reset(self, seed=None):
        seed = seed if seed is not None else random.randint(0, 1000)
        self.env = self.env_gen(seed=seed)
        obs, _ = self.env.reset()
        return obs

    def step(self, action):
        next_obs, reward, terminal, truncated, _ = self.env.step(action)
        return (next_obs, reward, terminal | truncated, {})

    def __getattr__(self, name):
        return getattr(self.env, name)

    def get_normalized_score(self, reward):
        return reward


def save_d4rl_dataset(env_name, save_dir):
    import gym
    import d4rl  # import 해야 gym.make()에서 d4rl 환경을 불러올 수 있음

    file_path = os.path.join(save_dir, "raw_dataset.npz")

    is_already_exist = os.path.exists(file_path)
    if is_already_exist:
        print(f"Dataset already exists at {file_path}")
        return

    env = gym.make(env_name)
    dataset = env.get_dataset()

    save_data = {
        "observations": dataset["observations"],
        "actions": dataset["actions"],
        "rewards": dataset["rewards"],
        "terminals": dataset["terminals"],
        "timeouts": dataset["timeouts"],
    }

    for key in dataset.keys():
        if key.startswith("infos"):
            save_data[key] = dataset[key]

    np.savez(file_path, **save_data)

    print(f"Dataset saved with keys: {save_data.keys()}")


def save_metaworld_dataset(env_name, save_dir):
    from zipfile import ZipFile
    import gdown
    import pickle

    file_id = metaworld_ids[env_name]
    npz_path = os.path.join(save_dir, "raw_dataset.npz")
    output_path = os.path.join(save_dir, f"{env_name}.zip")

    if not os.path.exists(npz_path):
        print("Generating Metaworld raw dataset")

        temp_dir = os.path.join(save_dir, "temp_unzip")
        os.makedirs(temp_dir, exist_ok=True)

        if not os.path.exists(output_path):
            # download file
            print(f"Downloading {env_name} dataset from Google Drive...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output_path, quiet=False)

        # unzip file
        with ZipFile(output_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        for dirpath, _, files in os.walk(temp_dir):
            for file in files:
                src_path = os.path.join(dirpath, file)
                dst_path = os.path.join(save_dir, file)
                os.replace(src_path, dst_path)

        # remove temp directory
        for root, dirs, files in os.walk(temp_dir, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(temp_dir)

    else:
        print(f"Dataset already exists at {npz_path}")

    pkl_files = [f for f in os.listdir(save_dir) if f.endswith(".pkl")]

    if not pkl_files:
        raise FileNotFoundError("No .pkl files found in the specified directory.")

    first_pkl_file = sorted(pkl_files)[0]
    file_path = os.path.join(save_dir, first_pkl_file)

    with open(file_path, "rb") as file:
        dataset = pickle.load(file)

    print("keys:", dataset.keys())

    save_data = {
        "observations": dataset["observations"],
        "actions": dataset["actions"],
        "rewards": dataset["rewards"],
        "terminals": dataset["dones"].astype(bool),
        "timeouts": np.zeros_like(dataset["dones"], dtype=bool),
        "success": dataset["success"].astype(bool),
    }

    np.savez(npz_path, **save_data)

    print(f"Dataset saved with keys: {save_data.keys()}")


def get_env(env_name, is_hidden=False):
    if env_name in metaworld_ids.keys():
        from metaworld.envs import (
            ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN,
        )

        if not is_hidden:
            env_gen = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[f"{env_name}-goal-observable"]
            env = MetaworldEnvWrapper(env_gen=env_gen)
            env.reset()
            return env
        else:
            env_gen = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[f"{env_name}-goal-hidden"]
            env = MetaworldEnvWrapper(env_gen=env_gen)
            env.reset()
    else:
        import gym
        import d4rl

        env = gym.make(env_name)
    return env


def save_dataset(env_name):
    save_dir = f"dataset/{env_name}"
    os.makedirs(save_dir, exist_ok=True)

    if env_name in metaworld_ids.keys():
        save_metaworld_dataset(env_name=env_name, save_dir=save_dir)
    else:
        save_d4rl_dataset(env_name=env_name, save_dir=save_dir)


def load_dataset(env_name):
    dir_path = f"dataset/{env_name}"
    dataset_name = "raw_dataset.npz"
    dataset = np.load(os.path.join(dir_path, dataset_name))

    return dataset


def load_pair(env_name, pair_name):
    dir_path = f"pair/{env_name}"
    pair = np.load(os.path.join(dir_path, f"{pair_name}.npz"), allow_pickle=True)

    return pair


def get_processed_data(env_name, pair_name):
    """
    return structured array of (s0, s1, mu) pairs
    s0, s1 is a structured array of (observations, actions)
    mu is a float
    """
    dataset = load_dataset(env_name)

    observations = dataset["observations"]
    actions = dataset["actions"]
    pair = load_pair(env_name, pair_name)
    processed_data = []

    for entry in pair["data"]:
        s0_idx, s1_idx, mu = (
            entry["s0"],
            entry["s1"],
            entry["mu"],
        )

        s0_obs = observations[s0_idx[0] : s0_idx[1]]
        s0_act = actions[s0_idx[0] : s0_idx[1]]
        s1_obs = observations[s1_idx[0] : s1_idx[1]]
        s1_act = actions[s1_idx[0] : s1_idx[1]]
        mu = mu

        dtype_list_s0 = [
            ("observations", "f4", (s0_obs.shape[1],)),
            ("actions", "f4", (s0_act.shape[1],)),
        ]
        s0 = np.array(list(zip(s0_obs, s0_act)), dtype=dtype_list_s0)

        dtype_list_s1 = [
            ("observations", "f4", (s1_obs.shape[1],)),
            ("actions", "f4", (s1_act.shape[1],)),
        ]
        s1 = np.array(list(zip(s1_obs, s1_act)), dtype=dtype_list_s1)

        processed_data.append(
            (
                s0,
                s1,
                mu,
            )
        )

    return np.array(processed_data, dtype=[("s0", "O"), ("s1", "O"), ("mu", "f4")])
