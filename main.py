import argparse
import glob
import os

import numpy as np


DEFAULT_ENV = "box-close-v2"
DEFAULT_PAIR = "train"
DEFAULT_PAIR_VAL = "val"
DEFAULT_PAIR_ALGO = "full-binary"
DEFAULT_REWARD_MODEL_ALGO = "MR"
DEFAULT_REWARD_MODEL_TAG = "-"


if __name__ == "__main__":
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
            "-2: Analyze Pairset\n"
            "-3: Evaluate reward model\n"
            "-4: Analyze changed dataset\n"
            "-5: Plot policy evaluation (Full methods)\n"
            "-5.1: Plot policy evaluation (List methods)\n"
            "-5.2: Evaluate policy\n"
            "1: Load and save dataset\n"
            "2: Generate preference pairs\n"
            "2.1: Generate preference pairs for test reward model\n"
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
        pass
    elif function_number == -1:
        # Analyze dataset
        from src.helper.analyze_dataset import analyze

        print("Analyzing dataset")

        analyze(env_name)
    elif function_number == -2:
        # Analyze Pairset
        from data_loading.load_data import get_processed_data
        import matplotlib.pyplot as plt

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

    elif function_number == -3:
        # Evaluate reward model
        from src.data_loading.preference_dataloader import get_dataloader
        from src.helper.evaluate_reward_model import evaluate_reward_model
        from src.reward_learning.MR import MR

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
            pair_name="test_reward_sigmoid",
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
            model.eval()
            models.append(model)

        accuracy, mse, pcc = evaluate_reward_model(
            env_name=env_name,
            models=models,
            data_loader=data_loader,
            output_name=f"{env_name}_{new_dataset_name}",
        )

        with open(log_path, "a") as log_file:
            log_file.write(
                f"{env_name}, {pair_name_base}, {mu_algo},{reward_model_algo},{reward_model_tag}, {accuracy:.4f}, {mse:.6f}, {pcc:.4f}\n"
            )
    elif function_number == -4:
        from src.helper.analyze_dataset import save_reward_graph

        # Analyze changed dataset

        dataset_path = new_dataset_path
        dataset_npz = np.load(dataset_path)
        dataset = {key: dataset_npz[key] for key in dataset_npz}

        log_path = f"log/dataset_reward_distribution_{new_dataset_name}.png"

        print("Analyzing changed dataset")

        save_reward_graph(dataset, new_dataset_name, log_path)
    elif function_number == -5:
        # Plot policy evaluation
        from src.helper.plot_policy_model import plot

        print("Plotting policy evaluation")

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
    elif function_number == -5.1:
        # Plot policy evaluation
        from src.helper.plot_policy_model import plot

        print("Plotting policy evaluation")

        env_list = ["lever-pull-v2"]
        pair_list = ["train-00", "train-01", "train-02", "train-03", "train-04"]
        postfix_list = [
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
                output_name=f"policy_train_{env_name}",
            )
    elif function_number == -5.2:
        # Evaluate policy
        from src.helper.evaluate_policy_model import evaluate_policy

        evaluate_policy(
            env_name=env_name,
            model_path=f"{policy_model_dir}/model/best_policy.pth",
        )

        evaluate_policy(
            env_name=env_name,
            model_path=f"{policy_model_dir}/model/last_policy.pth",
        )

    elif function_number == 1:
        # Load and save dataset
        from src.data_loading.load_data import save_dataset

        print("Loading and saving dataset", env_name)

        save_dataset(env_name)
    elif function_number == 2:
        # Generate preference pairs
        from src.data_generation.full_scripted_teacher import generate_full_pairs
        from src.data_generation.list_scripted_teacher import generate_list_pairs

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
    elif function_number == 2.1:
        # Generate preference pairs for test reward model
        from src.data_generation.full_scripted_teacher import generate_full_pairs

        print("Generating preference pairs for test_full_sigmoid", env_name, num)

        generate_full_pairs(
            env_name=env_name,
            pair_name_base="test_reward",
            num_pairs=num,
            mu_types=["sigmoid"],
        )
    elif function_number == 3:
        # Train reward model
        from src.data_loading.preference_dataloader import get_dataloader
        from src.reward_learning.reward_model_base import RewardModelBase
        from src.reward_learning.MR import MR

        print(
            "Training reward model",
            env_name,
            pair_name,
            pair_val_name,
            reward_model_algo,
        )

        data_loader, obs_dim, act_dim = get_dataloader(
            env_name=env_name,
            pair_name=pair_name,
        )

        val_data_loader, _, _ = get_dataloader(
            env_name=env_name,
            pair_name=pair_val_name,
        )

        print("obs_dim:", obs_dim, "act_dim:", act_dim)

        reward_model: RewardModelBase
        optimizer = None

        if reward_model_algo == "MR":
            model, optimizer = MR.initialize(
                config={"obs_dim": obs_dim, "act_dim": act_dim},
                path=reward_model_path,
                skip_if_exists=True,
            )
        elif reward_model_algo == "MR-linear":
            model, optimizer = MR.initialize(
                config={"obs_dim": obs_dim, "act_dim": act_dim},
                path=reward_model_path,
                linear_loss=True,
                skip_if_exists=True,
            )

        if model is not None:
            model.train_model(
                train_loader=data_loader,
                val_loader=val_data_loader,
                optimizer=optimizer,
                num_epochs=num,
            )

    elif function_number == 4:
        # Change reward and save dataset
        from src.data_loading.preference_dataloader import get_dataloader
        from src.reward_learning.reward_model_base import RewardModelBase
        from src.reward_learning.MR import MR
        from src.policy_learning.change_reward import change_reward

        print("Changing reward", env_name, new_dataset_name)

        _, obs_dim, act_dim = get_dataloader(env_name=env_name, pair_name=pair_name)

        print("obs_dim:", obs_dim, "act_dim:", act_dim)
        model_path_pattern = f"model/{env_name}/reward/{new_dataset_name}_*.pth"
        model_files = glob.glob(model_path_pattern)
        model_list = []

        if reward_model_algo == "MR":
            for model_file in model_files:
                model, _ = MR.initialize(
                    config={"obs_dim": obs_dim, "act_dim": act_dim}, path=model_file
                )
                model_list.append(model)
        elif reward_model_algo == "MR-linear":
            for model_file in model_files:
                model, _ = MR.initialize(
                    config={"obs_dim": obs_dim, "act_dim": act_dim},
                    path=model_file,
                    linear_loss=True,
                )
                model_list.append(model)

        change_reward(
            env_name=env_name, model_list=model_list, dataset_path=new_dataset_path
        )
    elif function_number == 5:
        # Train policy
        from src.policy_learning.iql import train

        print("Training policy", env_name, new_dataset_path)

        train(
            env_name=env_name,
            dataset_path=new_dataset_path,
            log_dir=policy_model_dir,
            num_epochs=num,
            is_goal_hidden=env_hidden,
        )
