"""
Offline PbRL Scripts
"""

import argparse
import glob
import os

import numpy as np
import matplotlib.pyplot as plt

from src.helper import (
    analyze_env_dataset,
    save_reward_graph,
    evaluate_reward_model,
    evaluate_best_and_last_policy,
    plot_policy_models,
)
from src.data_loading import (
    get_processed_data,
    load_pair,
    load_dataset,
    save_dataset,
    get_dataloader,
)
from src.data_generation.full_scripted_teacher import generate_full_pairs
from src.data_generation.list_scripted_teacher import generate_list_pairs
from src.data_generation.scored_pairs import generate_score_pairs
from src.reward_learning import MR, train_reward_model
from src.policy_learning import train, change_reward_from_all_datasets


DEFAULT_ENV = "box-close-v2"
DEFAULT_PAIR = "train"
DEFAULT_PAIR_VAL = "val"
DEFAULT_PAIR_ALGO = "full-binary"
DEFAULT_REWARD_MODEL_ALGO = "MR"
DEFAULT_REWARD_MODEL_TAG = "-"

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(
        description="Script for different functionalities",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "-e",
        "--env",
        type=str,
        default=DEFAULT_ENV,
        help="Name of the environment (maze2d-medium-dense-v1, etc.)",
    )

    parser.add_argument(
        "-eh",
        "--env_hidden",
        action="store_true",
        default=False,
        help="is_hidden for environment (True, False)",
    )

    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=1000,
        help="Number of pairs",
    )

    parser.add_argument(
        "-p",
        "--pair",
        type=str,
        default=DEFAULT_PAIR,
        help="Name of Experiment pair",
    )

    parser.add_argument(
        "-pv",
        "--pair_val",
        type=str,
        default=DEFAULT_PAIR_VAL,
        help="Name of the trajectory pair file to use for validate(val-full, etc.)",
    )

    parser.add_argument(
        "-pa",
        "--pair_algo",
        type=str,
        default=DEFAULT_PAIR_ALGO,
        help="Algorithm of generating pair (full-sigmoid, list-2, etc.)",
    )

    parser.add_argument(
        "-ra",
        "--reward_model_algo",
        type=str,
        default=DEFAULT_REWARD_MODEL_ALGO,
        help="Algorithm of reward model (MR, etc.)",
    )
    parser.add_argument(
        "-rt",
        "--reward_model_tag",
        type=str,
        default=DEFAULT_REWARD_MODEL_TAG,
        help="Tag of reward model",
    )

    parser.add_argument(
        "-f",
        "--function_number",
        type=float,
        default=0,
        help=(
            "0: Do nothing\n"
            "-1: Analyze dataset\n"
            "-2: Analyze Pairset, plot mu\n"
            "-2.1: Analyze Pairset, mu accuracy\n"
            "-3: Evaluate reward model\n"
            "-4: Analyze changed dataset\n"
            "-5: Plot policy evaluation (Full methods)\n"
            "-5.1: Plot policy evaluation (List methods)\n"
            "-5.2: Evaluate policy\n"
            "1: Load and save dataset\n"
            "2: Generate preference pairs\n"
            "3: Train reward model\n"
            "4: Change reward and save dataset\n"
            "5: Train policy\n"
            "Provide the number corresponding to the function you want to execute."
        ),
    )

    # Parse arguments
    args = parser.parse_args()
    env_name = args.env
    env_hidden = args.env_hidden
    num = args.num
    pair_name_base = args.pair
    pair_val_name_base = args.pair_val
    reward_model_algo = args.reward_model_algo
    reward_model_tag = args.reward_model_tag
    function_number = args.function_number
    pair_algo = args.pair_algo

    # Derived variables
    pair_name = f"{pair_name_base}_{pair_algo}"
    pair_val_name = f"{pair_val_name_base}_{pair_algo}"
    new_dataset_name = f"{pair_name}_{reward_model_algo}"
    reward_model_name = f"{new_dataset_name}_{reward_model_tag}"

    # Paths
    reward_model_path = f"model/{env_name}/reward/{reward_model_name}.pth"
    new_dataset_path = f"dataset/{env_name}/{new_dataset_name}_dataset.npz"
    policy_model_dir = f"model/{env_name}/policy/{new_dataset_name}"
    if env_hidden:
        policy_model_dir = policy_model_dir + "_hidden"

    print("main function started with args", args)

    # Execute function
    if function_number == 0:
        # Do nothing
        print("Pass")
    elif function_number == -1:
        # Analyze dataset
        analyze_env_dataset(env_name)
    elif function_number == -2:
        # Analyze Pairset

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

    elif function_number == -2.1:
        # Analyze Pairset

        dataset = load_dataset(env_name)
        data = load_pair(env_name, pair_name)

        answer_count = 0
        for s0, s1, mu in data["data"]:
            rewards_sum_0 = np.sum(dataset["rewards"][s0[0] : s0[1]])
            rewards_sum_1 = np.sum(dataset["rewards"][s1[0] : s1[1]])

            if rewards_sum_0 < rewards_sum_1 and mu > 0.5:
                answer_count += 1
            elif rewards_sum_0 > rewards_sum_1 and mu < 0.5:
                answer_count += 1
        print(answer_count / len(data["data"]))

    elif function_number == -3:
        # Evaluate reward model

        print("Evaluating reward model", env_name, new_dataset_name)

        log_path = "log/main_evaluate_reward.log"
        log_dir = os.path.dirname(log_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        model_path_pattern = f"model/{env_name}/reward/{new_dataset_name}_*.pth"
        model_files = glob.glob(model_path_pattern)
        accuarcy, mse, pcc = None, None, None

        data_loader, obs_dim, act_dim = get_dataloader(
            env_name=env_name,
            pair_name="test-reward_full-sigmoid",
            drop_last=False,
        )

        models = []

        for model_file in model_files:
            if reward_model_algo == "MR":
                model, _ = MR.initialize(
                    config={"obs_dim": obs_dim, "act_dim": act_dim}, path=model_file
                )
            elif reward_model_algo == "MR-linear":
                model, _ = MR.initialize(
                    config={"obs_dim": obs_dim, "act_dim": act_dim},
                    path=model_file,
                    linear_loss=True,
                )
            else:
                model = None

            if model is not None:
                model.eval()
                models.append(model)

        accuracy, mse, pcc = evaluate_reward_model(
            env_name=env_name,
            models=models,
            data_loader=data_loader,
            output_name=f"{env_name}_{new_dataset_name}",
        )

        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(
                f"{env_name}, {pair_name_base}, {pair_algo},{reward_model_algo},{reward_model_tag}, {accuracy:.4f}, {mse:.6f}, {pcc:.4f}\n"
            )
    elif function_number == -4:
        # Analyze changed dataset

        dataset_path = new_dataset_path
        dataset_npz = np.load(dataset_path)
        dataset = {key: dataset_npz[key] for key in dataset_npz}

        log_path = f"log/dataset_reward_distribution_{new_dataset_name}.png"

        print("Analyzing changed dataset")

        save_reward_graph(dataset, new_dataset_name, log_path)
    elif function_number == -5:
        # Plot policy evaluation

        print("Plotting policy evaluation")
        plot_policy_models()

    elif function_number == -5.2:
        # Evaluate policy
        evaluate_best_and_last_policy(env_name, policy_model_dir)

    elif function_number == 1:
        # Load and save dataset

        print("Loading and saving dataset", env_name)

        save_dataset(env_name)
    elif function_number == 2:
        # Generate preference pairs

        print("Generating preference pairs", env_name, pair_name_base, num)

        pair_algo_category = pair_algo.split("-")[0]

        if pair_algo_category == "full":
            generate_full_pairs(
                env_name=env_name,
                pair_name_base=pair_name_base,
                num_pairs=num,
                mu_types=[
                    "binary",
                    "sigmoid",
                    "linear",
                ],
            )
        elif pair_algo_category == "list":
            generate_list_pairs(
                env_name=env_name,
                pair_name_base=pair_name_base,
                num_trajectories=num,
                pair_algos=["list-2", "list-3", "list-5", "list-11"],
            )
        elif pair_algo_category == "score":
            generate_score_pairs(
                env_name=env_name,
                pair_name_base=pair_name_base,
                num_pairs=num,
                pair_algos=["rnn"],
            )
    elif function_number == 3:
        # Train reward model

        print(
            "Training reward model",
            env_name,
            pair_name,
            pair_val_name,
            reward_model_algo,
        )

        train_reward_model(
            env_name=env_name,
            pair_name=pair_name,
            pair_val_name=pair_val_name,
            reward_model_algo=reward_model_algo,
            reward_model_path=reward_model_path,
            num=num,
        )

    elif function_number == 4:
        # Change reward and save dataset
        print("Changing reward", env_name, new_dataset_name)

        change_reward_from_all_datasets(
            env_name=env_name,
            pair_name=pair_name,
            reward_model_algo=reward_model_algo,
            dataset_name=new_dataset_name,
            new_dataset_path=new_dataset_path,
        )

    elif function_number == 5:
        # Train policy
        print("Training policy", env_name, new_dataset_path)

        train(
            env_name=env_name,
            dataset_path=new_dataset_path,
            log_dir=policy_model_dir,
            num_epochs=num,
            is_goal_hidden=env_hidden,
        )
