import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/")))

from data_loading.load_data import get_processed_data, load_d4rl_dataset


if __name__ == "__main__":
    env_name_list = ["walker2d-medium-v2"]
    pair_name_list = ["full_minus-binary", "full_constant"]

    for env_name in env_name_list:
        for pair_name in pair_name_list:
            raw_data = load_d4rl_dataset(env_name)

            dir_path = f"dataset/{env_name}"
            dataset_name = f"{pair_name}_MR_dataset.npz"
            data = np.load(os.path.join(dir_path, dataset_name))

            print(raw_data["rewards"].shape)
            print(data["rewards"].shape)
            print(raw_data["rewards"][0:5])
            print(data["rewards"][0:5])

            plt.figure(figsize=(8, 6))
            plt.scatter(
                raw_data["rewards"], data["rewards"], alpha=0.5, label=pair_name
            )
            plt.xlabel("Actual Rewards")
            plt.ylabel("Predicted Rewards")
            plt.title(f"Actual vs. Predicted Rewards")
            plt.legend()
            plt.grid(True)
            plt.show()
