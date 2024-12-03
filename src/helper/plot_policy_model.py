import os

from matplotlib import pyplot as plt
import pandas as pd


def remove_max_min(series):
    # if len(series) <= 2:
    #     return series
    # return series.sort_values().iloc[1:-1]
    return series


def process_csv_files(csv_files):
    combined_df = pd.DataFrame()

    for file in csv_files:
        df = pd.read_csv(file)

        selected_columns = ["Timesteps", "Reward", "Success"]
        if all(col in df.columns for col in selected_columns):
            df = df[selected_columns]
            combined_df = pd.concat([combined_df, df], axis=0)
        else:
            print(f"File {file} does not contain required columns: {selected_columns}")

    grouped = combined_df.groupby("Timesteps")
    mean_df = (
        grouped["Reward"]
        .apply(remove_max_min)
        .groupby("Timesteps")
        .mean()
        .reset_index()
        .rename(columns={"Reward": "Reward_mean"})
    )
    std_df = (
        grouped["Reward"]
        .apply(lambda x: ((x**2).mean() ** 0.5) / ((len(x)) ** 0.5))
        .reset_index()
        .rename(columns={"Reward": "Reward_std"})
    )
    success_rate_df = (
        grouped["Success"]
        .apply(lambda x: (x > 0).mean())
        .reset_index()
        .rename(columns={"Success": "Success_rate"})
    )

    final_df = mean_df.merge(std_df, on="Timesteps").merge(
        success_rate_df, on="Timesteps"
    )

    final_df["Reward_mean"] = final_df["Reward_mean"].ewm(alpha=0.5).mean()
    final_df["Reward_std"] = final_df["Reward_std"].ewm(alpha=0.9).mean()

    return final_df


def plot_and_save(df_list=[], output_name="name"):
    plt.figure(figsize=(10, 12))

    # Reward
    plt.subplot(2, 1, 1)
    for mu_algo_name, df in df_list:
        mean_values = df["Reward_mean"].values
        std_values = df["Reward_std"].values

        plt.plot(df["Timesteps"].values, mean_values, label=mu_algo_name)
        plt.fill_between(
            df["Timesteps"],
            mean_values - std_values,
            mean_values + std_values,
            alpha=0.4,
        )
    plt.xlabel("Timestep")
    plt.ylabel("Reward per timesteps")
    plt.title(f"{output_name} - Reward")
    plt.legend()
    plt.grid(True)

    # Success
    plt.subplot(2, 1, 2)
    for mu_algo_name, df in df_list:
        plt.plot(df["Timesteps"].values, df["Success_rate"].values, label=mu_algo_name)
    plt.xlabel("Timestep")
    plt.ylabel("Success rate")
    plt.title(f"{output_name} - Success")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"log/{output_name}.png")
    plt.close()


def plot(env_name="", pair_list=[], postfix_list=[], output_name="name"):
    csv_files = []
    df_list = []

    for postfix in postfix_list:
        csv_files = []
        for pair_name in pair_list:
            file_path = f"model/{env_name}/policy/{pair_name}_{postfix}/train_log.csv"
            if os.path.exists(file_path):
                print(file_path)
                csv_files.append(file_path)

        if not csv_files:
            continue

        df = process_csv_files(csv_files=csv_files)
        df_list.append((postfix, df))

    if df_list:
        plot_and_save(df_list=df_list, output_name=output_name)
    else:
        print("nothing to plot")


def plot_policy_models(pair_name_base):
    """
    plot the policy models from hard-coded lists
    """
    env_list = ["box-close-v2"]
    pair_list = [
        pair_name_base,
        f"{pair_name_base}-00",
        f"{pair_name_base}-01",
        f"{pair_name_base}-02",
        f"{pair_name_base}-03",
        f"{pair_name_base}-04",
    ]
    postfix_list = [
        "full-binary_MR",
        "full-binary_MR-linear",
        "full-linear_MR-linear",
        "score-rnn_MR",
        "score-rnn_MR-linear",
        "list-2_MR-linear",
        "list-3_MR-linear",
        "list-5_MR-linear",
        "list-11_MR-linear",
    ]

    for env_name in env_list:
        plot(
            env_name=env_name,
            pair_list=pair_list,
            postfix_list=postfix_list,
            output_name=f"policy_{env_name}_{pair_name_base}",
        )
