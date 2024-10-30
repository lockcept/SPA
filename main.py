import argparse
import glob


DEFAULT_ENV = "maze2d-medium-dense-v1"
DEFAULT_PAIR = "full"
DEFAULT_EVAL_PAIR = "eval_full"
DEFAULT_MU_ALGO = "binary"
DEFAULT_REWARD_MODEL_ALGO = "MLP"
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
        "-ep",
        "--eval_pair",
        type=str,
        default=DEFAULT_EVAL_PAIR,
        help="Name of the trajectory pair file to use for eval(eval_full, etc.)",
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
    eval_pair_name_base = args.eval_pair
    reward_model_algo = args.reward_model_algo
    reward_model_tag = args.reward_model_tag
    function_number = args.function_number
    mu_algo = args.mu_algo

    pair_name = f"{pair_name_base}_{mu_algo}"
    eval_pair_name = f"{eval_pair_name_base}_{mu_algo}"
    new_dataset_name = f"{pair_name}_{reward_model_algo}"
    reward_model_name = f"{new_dataset_name}_{reward_model_tag}"

    pair_path = f"model/{env_name}/{pair_name}.npz"
    eval_pair_path = f"model/{env_name}/{eval_pair_name}.npz"
    reward_model_path = f"model/{env_name}/reward/{reward_model_name}.pth"
    new_dataset_path = f"dataset/{env_name}/{new_dataset_name}_dataset.npz"
    policy_model_dir_path = f"model/{env_name}/policy/{new_dataset_name}"

    if function_number == 0:
        print("Pass")
        pass
    elif function_number == -1:
        from src.helper.analyze_d4rl import analyze

        analyze(env_name)
    elif function_number == -2:
        from src.helper.evaluate_reward_model import evaluate_reward_model_MLP

        if reward_model_algo == "MLP":
            evaluate_reward_model_MLP(
                env_name, reward_model_path, test_pair_name="test_full_sigmoid"
            )
    elif function_number == -3:
        from src.helper.plotter import plot

        plot(progress_path_list=["model/hopper-medium-v2/policy/full_binary_MLP",
                                 "model/hopper-medium-v2/policy/full_sigmoid_MLP",
                                 "model/maze2d-medium-dense-v1/policy/full_1000_binary_MLP",
                                 "model/maze2d-medium-dense-v1/policy/full_1000_sigmoid_MLP"])


    elif function_number == 1:
        from src.data_loading.load_d4rl import save_d4rl_dataset

        save_d4rl_dataset(env_name)
    elif function_number == 2:
        from src.data_generation.full_scripted_teacher import generate_pairs

        generate_pairs(
            env=env_name,
            pair_name_base=pair_name_base,
            num_pairs=num,
            mu_types=["binary", "continuous", "sigmoid", "sigmoid_0.1", "sigmoid_0.25","sigmoid_0.5"],
        )
    elif function_number == 2.1:
        from src.data_generation.full_scripted_teacher import generate_pairs

        print("Generating preference pairs for test_full_sigmoid")
        generate_pairs(env=env_name, pair_name_base= "test_full", num_pairs=num, mu_types=["sigmoid"])
    elif function_number == 3:
        from src.data_loading.preference_dataloader import get_dataloader
        from src.reward_learning.multilayer_perceptron import (
            train,
        )

        data_loader, obs_dim, act_dim = get_dataloader(
            env_name=env_name,
            pair_name=pair_name,
        )

        eval_data_loader, _, _ = get_dataloader(
            env_name=env_name,
            pair_name=eval_pair_name,
        )

        print("obs_dim:", obs_dim, "act_dim:", act_dim)

        if reward_model_algo == "MLP":
            train(
                data_loader=data_loader,
                eval_data_loader=eval_data_loader,
                reward_model_path=reward_model_path,
                obs_dim=obs_dim,
                act_dim=act_dim,
            )

    elif function_number == 4:
        from src.data_loading.preference_dataloader import get_dataloader
        from src.reward_learning.multilayer_perceptron import initialize_network
        from src.policy_learning.change_reward import change_reward

        data_loader, obs_dim, act_dim = get_dataloader(
            env_name=env_name, pair_name=pair_name
        )

        print("obs_dim:", obs_dim, "act_dim:", act_dim)

        if reward_model_algo == "MLP":
            model_path_pattern = f"model/{env_name}/reward/{new_dataset_name}_*.pth"
            model_files = glob.glob(model_path_pattern)

            model_list = []
            for model_file in model_files:
                model, _ = initialize_network(obs_dim=obs_dim, act_dim=act_dim, path=model_file)
                model_list.append(model)

            change_reward(env_name=env_name, model_list=model_list, dataset_path=new_dataset_path)
    elif function_number == 5:
        from src.policy_learning.iql import train

        train(env_name=env_name, dataset_path=new_dataset_path, log_dir = policy_model_dir_path)
