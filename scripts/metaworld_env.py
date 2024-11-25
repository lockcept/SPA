import os
import sys
import metaworld
import random
from metaworld.envs import (
    ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
    ALL_V2_ENVIRONMENTS_GOAL_HIDDEN,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/")))

from data_loading.load_data import load_dataset


if __name__ == "__main__":
    env_name_list = ["box-close-v2"]

    for env_name in env_name_list:
        raw_data = load_dataset(env_name)

        print(raw_data["observations"][0])

        env_gen = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[f"{env_name}-goal-observable"]
        env = env_gen(seed=0)
        obs, _ = env.reset(seed=0)
        print(obs)

        env_gen = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[f"{env_name}-goal-hidden"]
        env = env_gen(seed=0)
        obs, _ = env.reset(seed=0)
        print(obs)

        ml1 = metaworld.ML1(env_name)

        env = ml1.train_classes[env_name]()
        task = random.choice(ml1.train_tasks)
        env.set_task(task)
        obs, _ = env.reset()  # Reset environment
        print(obs)
