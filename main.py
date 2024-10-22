import argparse
import os


DEFAULT_ENV = "maze2d-medium-dense-v1"
DEFAULT_PAIR = "full"
DEFAULT_TEST_PAIR = "test_full"
DEFAULT_MU_TYPE = "binary"
DEFAULT_REWARD_MODEL = "MLP"


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
        "-p",
        "--pair",
        type=str,
        default=DEFAULT_PAIR,
        help="Name of the trajectory pair generation algorithm (full, etc.)",
    )

    parser.add_argument(
        "-tp",
        "--test_pair",
        type=str,
        default=DEFAULT_TEST_PAIR,
        help="Name of the trajectory pair file to use for test or eval(test_full, etc.)",
    )

    parser.add_argument(
        "-r",
        "--reward_model",
        type=str,
        default=DEFAULT_REWARD_MODEL,
        help="Name of reward model (MLP, etc.)",
    )

    parser.add_argument(
        "-m",
        "--mu",
        type=str,
        default=DEFAULT_MU_TYPE,
        help="Type of Mu(binary, continuous)",
    )

    parser.add_argument(
        "-f",
        "--function_number",
        type=float,
        default=0,
        help=(
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
    pair = args.pair
    test_pair = args.test_pair
    reward_model = args.reward_model
    function_number = args.function_number
    mu_type = args.mu

    pair_name = f"{pair}_{mu_type}"
    pair_path = f"model/{env_name}/{pair_name}.npz"
    test_pair_name = f"{test_pair}_{mu_type}"
    test_pair_path = f"model/{env_name}/{test_pair_name}.npz"
    reward_model_name = f"{pair_name}_{reward_model}"
    reward_model_path = f"model/{env_name}/{reward_model_name}.pth"

    if function_number == 0:
        print("Pass")
        pass
    elif function_number == -1:
        from src.helper.analyze_d4rl import analyze

        analyze(env_name)
    elif function_number == -2:
        from src.helper.evaluate_reward_model import evaluate_reward_model_MLP

        if reward_model == "MLP":
            evaluate_reward_model_MLP(
                env_name, reward_model_path, eval_pair_name="eval_full_sigmoid"
            )

    elif function_number == 1:
        from src.data_loading.load_d4rl import save_d4rl_dataset

        save_d4rl_dataset(env_name)
    elif function_number == 2:
        from src.data_generation.full_scripted_teacher import generate_pairs

        generate_pairs(env_name, pair, 30)
    elif function_number == 2.1:
        from src.data_generation.full_scripted_teacher import generate_pairs

        generate_pairs(env_name, "eval_full", 5000, ["sigmoid"])
    elif function_number == 3:
        from src.data_loading.preference_dataloader import get_dataloader
        from src.reward_learning.multilayer_perceptron import (
            BradleyTerryLoss,
            initialize_network,
            learn,
        )

        data_loader, obs_dim, act_dim = get_dataloader(
            env_name=env_name,
            pair_name=pair_name,
        )

        test_data_loader, _, _ = get_dataloader(
            env_name=env_name,
            pair_name=test_pair_name,
        )

        print("obs_dim:", obs_dim, "act_dim:", act_dim)

        save_dir = os.path.dirname(reward_model_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if reward_model == "MLP":
            model, optimizer = initialize_network(
                obs_dim, act_dim, path=reward_model_path
            )
            loss_fn = BradleyTerryLoss()

            num_epochs = 100
            loss_history = learn(
                model,
                optimizer,
                data_loader,
                test_data_loader,
                loss_fn,
                model_path=reward_model_path,
                num_epochs=num_epochs,
            )

            print("Training completed. Loss history:", loss_history)
    elif function_number == 4:
        from src.data_loading.preference_dataloader import get_dataloader
        from src.reward_learning.multilayer_perceptron import initialize_network
        from src.policy_learning.change_reward import change_reward

        save_path = f"model/{env_name}/{pair_name}_{reward_model_name}.pth"
        dataset_path = f"dataset/{env_name}/{reward_model_name}_dataset.npz"
        data_loader, obs_dim, act_dim = get_dataloader(
            env_name=env_name, pair_name=pair_name
        )

        print("obs_dim:", obs_dim, "act_dim:", act_dim)

        if reward_model_name == "MLP":
            model, _ = initialize_network(obs_dim, act_dim, path=save_path)
            change_reward(env_name, model, dataset_path)
    elif function_number == 5:
        from src.policy_learning.iql import train

        dataset_path = f"dataset/{env_name}/{reward_model_name}_dataset.npz"

        train(env_name=env_name, dataset_path=dataset_path)
