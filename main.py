"""
Offline PbRL Scripts
"""

import argparse
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# pylint: disable=C0413


from src.helper import (
    analyze_env_dataset,
    save_reward_graph,
    evaluate_score_model,
    evaluate_and_log_reward_models,
    plot_pair,
    evaluate_pair,
    plot_policy_models,
    analyze_pair,
    evaluate_reward_by_state,
    evaluate_existing_reward_dataset,
)
from src.data_loading import (
    save_dataset,
)
from src.data_generation import generate_all_algo_pairs
from src.data_generation.data_research import data_research
from src.reward_learning import train_reward_model
from src.policy_learning import (
    train_iql_policy,
    train_ipl_policy,
    train_dppo_policy,
    change_reward_from_all_datasets,
    change_reward_and_save_pt,
)


DEFAULT_ENV = "box-close-v2"
DEFAULT_EXP_NAME = "exp00"
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
        "-n",
        "--num",
        type=int,
        default=1000,
        help="Number of epoch",
    )

    parser.add_argument(
        "-exp",
        "--exp",
        type=str,
        default=DEFAULT_EXP_NAME,
        help="Name of Experiment",
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
            "-1: Analyze and plot dataset\n"
            "-2: Plot mu histogram\n"
            "-2.1: Analyze Pairset, mu accuracy\n"
            "-2.2: Evaluate score model\n"
            "-2.3: Analyze Good Pairset\n",
            "-3: Evaluate reward model\n"
            "-4: Analyze changed dataset\n"
            "-5: Plot policy evaluation\n"
            "-5.1: Evaluate policy\n"
            "1: Load and save dataset\n"
            "2: Generate preference pairs\n"
            "3: Train reward model\n"
            "4: Change reward and save dataset\n"
            "5: Train policy\n"
            "Provide the number corresponding to the function you want to execute.",
        ),
    )

    # Parse arguments
    args = parser.parse_args()
    env_name = args.env
    num = args.num
    exp_name = args.exp
    reward_model_algo = args.reward_model_algo
    reward_model_tag = args.reward_model_tag
    function_number = args.function_number
    pair_algo = args.pair_algo

    print("main function started with args", args)

    # Execute function
    if function_number == 0:
        # Do nothing
        print("Pass")
    elif function_number == -1:
        # Analyze dataset
        analyze_env_dataset(env_name)
    elif function_number == -2:
        # Plot mu histogram from pairset

        plot_pair(
            env_name_list=[env_name],
            exp_name=exp_name,
            pair_algo_list=[pair_algo],
        )

    elif function_number == -2.1:
        # Analyze Pairset (PCC, ACC, etc)
        evaluate_pair(
            env_name=env_name, exp_name=exp_name, pair_type="train", pair_algo=pair_algo
        )

    elif function_number == -2.2:
        # Evaluate score model
        score_algo = pair_algo.split("-")[1]
        evaluate_score_model(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo=pair_algo,
            test_pair_type="train",
            test_pair_algo=pair_algo,
        )

        evaluate_score_model(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo=pair_algo,
            test_pair_type="test",
            test_pair_algo="full-binary",
        )

    elif function_number == -2.3:
        # Analyze good pairset
        analyze_pair(env_name, exp_name, "train", pair_algo)

    elif function_number == -3:
        # Evaluate reward model

        print(
            "Evaluating reward model", env_name, exp_name, pair_algo, reward_model_algo
        )
        evaluate_and_log_reward_models(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo=pair_algo,
            reward_model_algo=reward_model_algo,
        )
        evaluate_reward_by_state(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo=pair_algo,
            reward_model_algo=reward_model_algo,
        )

    elif function_number == -4:
        # Analyze changed dataset
        print("Analyzing changed dataset")

        save_reward_graph(env_name, exp_name, pair_algo, reward_model_algo)
    elif function_number == -4.1:
        # Evaluate reward model
        print(
            "Evaluating reward model", env_name, exp_name, pair_algo, reward_model_algo
        )

        evaluate_existing_reward_dataset(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo=pair_algo,
            reward_model_algo=reward_model_algo,
        )
    elif function_number == -5:
        # Plot policy evaluation

        print("Plotting policy evaluation")
        plot_policy_models(exp_name=exp_name)

    elif function_number == 1:
        # Load and save dataset
        print("Loading and saving dataset", env_name)

        save_dataset(env_name)
    elif function_number == 2:
        # Generate preference pairs
        print("Generating preference pairs", env_name, exp_name)

        generate_all_algo_pairs(env_name=env_name, exp_name=exp_name)
    elif function_number == 2.5:
        # Generate preference pairs
        print("Function for Research", env_name, exp_name)

        data_research(env_name=env_name, exp_name=exp_name)
    elif function_number == 3:
        # Train reward model
        print("Training reward model")

        train_reward_model(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo=pair_algo,
            reward_model_algo=reward_model_algo,
            reward_model_tag=reward_model_tag,
            num_epoch=num,
        )

    elif function_number == 4:
        # Change reward and save dataset
        print("Changing reward", env_name, exp_name, pair_algo, reward_model_algo)

        if reward_model_algo == "PT-linear":
            is_linear = True
            change_reward_and_save_pt(
                env_name=env_name,
                exp_name=exp_name,
                pair_algo=pair_algo,
                is_linear=is_linear,
                seq_len=25,
            )
        elif reward_model_algo == "PT-exp":
            is_linear = False
            change_reward_and_save_pt(
                env_name=env_name,
                exp_name=exp_name,
                pair_algo=pair_algo,
                is_linear=is_linear,
                seq_len=25,
            )
        else:
            change_reward_from_all_datasets(
                env_name=env_name,
                exp_name=exp_name,
                pair_algo=pair_algo,
                reward_model_algo=reward_model_algo,
            )
    elif function_number == 5:
        # Train policy
        print("Training policy", env_name, exp_name, pair_algo, reward_model_algo)

        train_iql_policy(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo=pair_algo,
            reward_model_algo=reward_model_algo,
        )
    elif function_number == 5.1:
        # Train IPL
        print("Train IPL", env_name, exp_name, pair_algo, reward_model_algo)

        train_ipl_policy(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo=pair_algo,
            reward_model_algo=reward_model_algo,
        )
    elif function_number == 5.2:
        # Train DPPO
        print("Train DPPO", env_name, exp_name, pair_algo, reward_model_algo)

        train_dppo_policy(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo=pair_algo,
            reward_model_algo=reward_model_algo,
        )
