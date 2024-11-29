import os

from matplotlib import pyplot as plt
import pandas as pd


def remove_max_min(series):
    # if len(series) <= 2:
    #     return series
    # return series.sort_values().iloc[1:-1]
    return series


def process_csv_files(csv_files, label):
    combined_df = pd.DataFrame()

    for file in csv_files:
        df = pd.read_csv(file)

        selected_columns = ["timestep", label, f"{label}_std"]
        if all(col in df.columns for col in selected_columns):
            df = df[selected_columns]
            combined_df = pd.concat([combined_df, df], axis=0)
        else:
            print(f"File {file} does not contain required columns: {selected_columns}")

    grouped = combined_df.groupby("timestep")
    mean_df = (
        grouped[label]
        .apply(remove_max_min)
        .groupby("timestep")
        .mean()
        .reset_index()
        .rename(columns={label: "mean"})
    )
    std_df = (
        grouped[f"{label}_std"]
        .apply(lambda x: ((x**2).mean() ** 0.5) / ((10 * len(x)) ** 0.5))
        .reset_index()
        .rename(columns={f"{label}_std": "std"})
    )

    result = pd.merge(mean_df, std_df, on="timestep")
    result["smoothed_mean"] = result["mean"].ewm(alpha=0.5).mean()
    result["smoothed_std"] = result[f"std"].ewm(alpha=0.9).mean()

    return result


def plot_and_save(df_list=[], output_name="name"):
    plt.figure(figsize=(10, 6))
    for mu_algo_name, df in df_list:
        mean_values = df["smoothed_mean"].values
        std_values = df["smoothed_std"].values

        plt.plot(
            df["timestep"].values, mean_values, label=mu_algo_name
        )  # Plotting the mean column
        plt.fill_between(
            df["timestep"],
            mean_values - std_values,
            mean_values + std_values,
            alpha=0.4,
        )
    label = df_list[0][1].columns[1]

    plt.xlabel("Timestep")
    plt.ylabel("Mean and Standard Deviation")
    plt.title(f"{output_name} - {label}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"log/{output_name}.png")
    plt.close()


def plot(env_name="", pair_list=[], postfix_list=[], output_name="name"):
    label = "eval/normalized_episode_reward"

    csv_files = []
    df_list = []

    for postfix in postfix_list:
        csv_files = []
        for pair_name in pair_list:
            file_path = f"model/{env_name}/policy/{pair_name}_{postfix}/record/policy_training_progress.csv"
            print(file_path)
            if os.path.exists(file_path):
                csv_files.append(file_path)

        if not csv_files:
            continue

        df = process_csv_files(
            csv_files=csv_files,
            label=label,
        )
        df_list.append((postfix, df))

    if df_list:
        plot_and_save(df_list=df_list, output_name=output_name)
    else:
        print("nothing to plot")


def plot_policy_models():
    """
    plot the policy models from hard-coded lists
    """
    env_list = ["box-close-v2", "lever-pull-v2", "dial-turn-v2"]
    pair_list = ["train-00", "train-01", "train-02", "train-03", "train-04"]
    postfix_list = [
        "full-binary_MR",
        "full-sigmoid_MR",
        "full-linear_MR",
        "full-linear_MR-linear",
    ]

    for env_name in env_list:
        plot(
            env_name=env_name,
            pair_list=pair_list,
            postfix_list=postfix_list,
            output_name=f"policy_full_{env_name}",
        )
