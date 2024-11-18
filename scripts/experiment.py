import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/")))

from data_loading.load_data import get_processed_data


if __name__ == "__main__":
    env_name_list = ["box-close-v2"]
    pair_name_list = [
        "val_binary",
        "val_sigmoid",
        "val_linear",
    ]

    for env_name in env_name_list:
        for pair_name in pair_name_list:
            data = get_processed_data(env_name, pair_name)

            # histogram of mu values
            mu_values = [item["mu"] for item in data]
            plt.figure(figsize=(10, 6))
            plt.hist(mu_values, bins=50, alpha=0.75)
            plt.xlabel("Mu Values")
            plt.ylabel("Frequency")
            plt.title(f"{env_name}_{pair_name}")
            plt.grid(True)
            plt.savefig(f"log/mu_histogram_{env_name}_{pair_name}.png")
