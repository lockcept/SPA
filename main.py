import argparse
import glob
import os


DEFAULT_ENV = "maze2d-medium-dense-v1"
DEFAULT_PAIR = "full"
DEFAULT_VAL_PAIR = "val_full"
DEFAULT_MU_ALGO = "binary"
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
        help="Name of the trajectory pair generation algorithm (full, etc.)",
    )

    parser.add_argument(
        "-vp",
        "--val_pair",
        type=str,
        default=DEFAULT_VAL_PAIR,
        help="Name of the trajectory pair file to use for validate(val_full, etc.)",
    )

    parser.add_argument(
        "-ra",
        "--reward_model_algo",
        type=str,
        default=DEFAULT_REWARD_MODEL_ALGO,
        help="Algorithm of reward model (MLP, etc.)",
    )
    parser.add_argument(
        "-rt",
        "--reward_model_tag",
        type=str,
        default=DEFAULT_REWARD_MODEL_TAG,
        help="Tag of reward model",
    )

    parser.add_argument(
        "-ma",
        "--mu_algo",
        type=str,
        default=DEFAULT_MU_ALGO,
        help="Algorithm of Mu(binary, continuous)",
    )

    parser.add_argument(
        "-f",
        "--function_number",
        type=float,
        default=0,
        help=(
            "-3: helper policy evalutaion\n"
            "-2: helper evaluate reward model\n"
            "-1: helper analyze d4rl\n"
            " 0: do nothing\n"
            " 1: load and save d4rl\n"
            " 2: load and save preference pairs\n"
            " 3: train reward model\n"
            " 4: change reward\n"
            " 5: train policy\n"
            "Provide the number corresponding to the function you want to execute."
        ),
    )

    args = parser.parse_args()
    env_name = args.env
    num = args.num
    pair_name_base = args.pair
    val_pair_name_base = args.val_pair
    reward_model_algo = args.reward_model_algo
    reward_model_tag = args.reward_model_tag
    function_number = args.function_number
    mu_algo = args.mu_algo

    pair_name = f"{pair_name_base}_{mu_algo}"
    val_pair_name = f"{val_pair_name_base}_{mu_algo}"
    new_dataset_name = f"{pair_name}_{reward_model_algo}"
    reward_model_name = f"{new_dataset_name}_{reward_model_tag}"

    pair_path = f"model/{env_name}/{pair_name}.npz"
    val_pair_path = f"model/{env_name}/{val_pair_name}.npz"
    reward_model_path = f"model/{env_name}/reward/{reward_model_name}.pth"
    new_dataset_path = f"dataset/{env_name}/{new_dataset_name}_dataset.npz"
    policy_model_dir_path = f"model/{env_name}/policy/{new_dataset_name}"

    print("main function started with args", args)

    if function_number == 0:
        print("Pass")
        pass
    elif function_number == -1:
        from src.helper.analyze_d4rl import analyze

        print("Analyzing d4rl dataset")

        analyze(env_name)
    elif function_number == -2:
        from src.data_loading.preference_dataloader import get_dataloader
        from src.helper.evaluate_reward_model import evaluate_reward_model
        from src.reward_learning.MLP import MLP
        from src.reward_learning.MR import MR

        print("Evaluating reward model", env_name, new_dataset_name)

        log_path = "log/main_evaluate_reward.log"

        model_path_pattern = f"model/{env_name}/reward/{new_dataset_name}_*.pth"
        model_files = glob.glob(model_path_pattern)
        accuarcy, mse, pcc = None, None, None

        data_loader, obs_dim, act_dim = get_dataloader(
            env_name=env_name,
            pair_name="test_full_sigmoid",
            drop_last=False,
        )

        models = []

        for model_file in model_files:
            if reward_model_algo == "MLP":
                model, _ = MLP.initialize(
                    config={"obs_dim": obs_dim, "act_dim": act_dim}, path=model_file
                )
            elif reward_model_algo == "MR":
                model, _ = MR.initialize(
                    config={"obs_dim": obs_dim, "act_dim": act_dim}, path=model_file
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
    elif function_number == -3:
        from src.helper.plot_policy_model import plot

        print("Plotting policy evaluation")

        env_list = ["halfcheetah-random", "hopper-medium-v2", "walker2d-medium-v2"]
        pair_list = ["full_00", "full_01", "full_02", "full_03", "full_04"]
        mu_algo_list = ["binary", "sigmoid", "sigmoid-0.25", "sigmoid-0.5"]

        for env_name in env_list:
            plot(
                env_name=env_name,
                pair_list=pair_list,
                mu_algo_list=mu_algo_list,
                output_name=f"policy_{env_name}_full_MR",
            )

    elif function_number == 1:
        from src.data_loading.load_d4rl import save_d4rl_dataset

        print("Loading and saving d4rl dataset", env_name)

        save_d4rl_dataset(env_name)
    elif function_number == 2:
        from src.data_generation.full_scripted_teacher import generate_pairs

        print("Generating preference pairs", env_name, pair_name_base, num)

        generate_pairs(
            env_name=env_name,
            pair_name_base=pair_name_base,
            num_pairs=num,
            mu_types=[
                "binary",
                "sigmoid",
                "linear",
            ],
        )
    elif function_number == 2.1:
        from src.data_generation.full_scripted_teacher import generate_pairs

        print("Generating preference pairs for test_full_sigmoid", env_name, num)

        generate_pairs(
            env=env_name,
            pair_name_base="test_full",
            num_pairs=num,
            mu_types=["sigmoid"],
        )
    elif function_number == 3:
        from src.data_loading.preference_dataloader import get_dataloader
        from src.reward_learning.reward_model_base import RewardModelBase
        from src.reward_learning.MLP import MLP
        from src.reward_learning.MR import MR

        print(
            "Training reward model",
            env_name,
            pair_name,
            val_pair_name,
            reward_model_algo,
        )

        data_loader, obs_dim, act_dim = get_dataloader(
            env_name=env_name,
            pair_name=pair_name,
        )

        val_data_loader, _, _ = get_dataloader(
            env_name=env_name,
            pair_name=val_pair_name,
        )

        print("obs_dim:", obs_dim, "act_dim:", act_dim)

        reward_model: RewardModelBase
        optimizer = None

        if reward_model_algo == "MLP":
            model, optimizer = MLP.initialize(
                config={"obs_dim": obs_dim, "act_dim": act_dim},
                path=reward_model_path,
                skip_if_exists=True,
            )
        elif reward_model_algo == "MR":
            model, optimizer = MR.initialize(
                config={"obs_dim": obs_dim, "act_dim": act_dim},
                path=reward_model_path,
                skip_if_exists=True,
            )

        if model is not None:
            model.train_model(
                train_loader=data_loader,
                val_loader=val_data_loader,
                optimizer=optimizer,
            )

    elif function_number == 4:
        from src.data_loading.preference_dataloader import get_dataloader
        from src.reward_learning.reward_model_base import RewardModelBase
        from src.reward_learning.MLP import MLP
        from src.reward_learning.MR import MR
        from src.policy_learning.change_reward import change_reward

        print("Changing reward", env_name, new_dataset_name)

        _, obs_dim, act_dim = get_dataloader(env_name=env_name, pair_name=pair_name)

        print("obs_dim:", obs_dim, "act_dim:", act_dim)
        model_path_pattern = f"model/{env_name}/reward/{new_dataset_name}_*.pth"
        model_files = glob.glob(model_path_pattern)
        model_list = []

        if reward_model_algo == "MLP":
            for model_file in model_files:
                model, _ = MLP.initialize(
                    config={"obs_dim": obs_dim, "act_dim": act_dim}, path=model_file
                )
                model_list.append(model)
        elif reward_model_algo == "MR":
            for model_file in model_files:
                model, _ = MR.initialize(
                    config={"obs_dim": obs_dim, "act_dim": act_dim}, path=model_file
                )
                model_list.append(model)

        change_reward(
            env_name=env_name, model_list=model_list, dataset_path=new_dataset_path
        )
    elif function_number == 5:
        from src.policy_learning.iql import train

        print("Training policy", env_name, new_dataset_path)

        train(
            env_name=env_name,
            dataset_path=new_dataset_path,
            log_dir=policy_model_dir_path,
        )
