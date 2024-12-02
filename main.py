"""
Offline PbRL Scripts
"""

import argparse
import glob
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# pylint: disable=C0413


from src.helper import (
    analyze_env_dataset,
    save_reward_graph,
    evaluate_score_model,
    evaluate_reward_model,
    evaluate_best_and_last_policy,
    plot_pair,
    evaluate_pair,
    plot_policy_models,
)
from src.data_loading import (
    save_dataset,
    get_dataloader,
)
from src.data_generation import generate_all_algo_pairs
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
    reward_model_algo = args.reward_model_algo
    reward_model_tag = args.reward_model_tag
    function_number = args.function_number
    pair_algo = args.pair_algo

    # Derived variables
    train_pair_name_base = f"{pair_name_base}-train"
    val_pair_name_base = f"{pair_name_base}-val"
    test_pair_name_base = f"{pair_name_base}-test"

    train_pair_name = f"{train_pair_name_base}_{pair_algo}"
    val_pair_name = f"{val_pair_name_base}_{pair_algo}"
    test_pair_name = f"{test_pair_name_base}_full-binary"

    pair_name = f"{pair_name_base}_{pair_algo}"
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

        plot_pair(
            env_name_list=["box-close-v2"],
            pair_algo_list=[
                "full-binary",
                "full-linear",
                "list-2",
                "list-3",
                "list-5",
                "list-11",
                "score-rnn",
            ],
            pair_name_base=train_pair_name_base,
        )

    elif function_number == -2.1:
        # Analyze Pairset
        evaluate_pair(env_name, train_pair_name)

    elif function_number == -2.2:
        # Analyze Pairset
        evaluate_score_model(
            env_name=env_name,
            model_path=f"model/{env_name}/score/{train_pair_name}.pth",
            pair_name=test_pair_name,
        )

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
            pair_name=test_pair_name,
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
                model = None  # pylint: disable=C0103

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
        print("Analyzing changed dataset")

        save_reward_graph(new_dataset_path, new_dataset_name)
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
        print("Generating preference pairs", env_name, pair_name_base)

        generate_all_algo_pairs(env_name, pair_name_base, include_score_pairs=True)
    elif function_number == 3:
        # Train reward model
        print("Training reward model")

        train_reward_model(
            env_name=env_name,
            pair_name=train_pair_name,
            pair_val_name=val_pair_name,
            reward_model_algo=reward_model_algo,
            reward_model_path=reward_model_path,
            num=num,
        )

    elif function_number == 4:
        # Change reward and save dataset
        print("Changing reward", env_name, new_dataset_name)

        change_reward_from_all_datasets(
            env_name=env_name,
            pair_name=train_pair_name,
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
