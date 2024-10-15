import argparse
import os

import torch


DEFAULT_ENV = "hopper-medium-v2"
DEFAULT_PAIRS = "full_preference_pairs"
DEFAULT_REWARD_MODEL = "MLP"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env",
        type=str,
        default=DEFAULT_ENV,
        help="Name of the environment to load the dataset for",
    )

    parser.add_argument(
        "--pairs",
        type=str,
        default=DEFAULT_PAIRS,
        help="Name of the file to load the preference pairs to",
    )

    parser.add_argument(
        "--reward_model",
        type=str,
        default=DEFAULT_REWARD_MODEL,
        help="Name of the file to load the preference pairs to",
    )
    # -2: helper evaluate reward model
    # -1: helper analyze d4rl
    # 0: do nothing
    # 1: load and save d4rl
    # 2: load and save preference pairs from full_scripted_teacher
    # 3: train reward model
    # 4: change reward
    parser.add_argument(
        "--function_number",
        type=int,
        default=0,
        help="Number of the function to execute",
    )

    args = parser.parse_args()
    env_name = args.env
    pair_name = args.pairs
    reward_model_name = args.reward_model
    function_number = args.function_number

    if function_number == 0:
        print("Pass")
        pass
    elif function_number == -1:
        from src.helper.analyze_d4rl import analyze

        analyze(env_name)
    elif function_number == -2:
        from src.helper.evaluate_reward_model import evaluate_reward_model_MLP

        if reward_model_name == "MLP":
            evaluate_reward_model_MLP(env_name, pair_name)

    elif function_number == 1:
        from src.data_loading.load_d4rl import load

        load(env_name)
    elif function_number == 2:
        from src.data_generation.full_scripted_teacher import generate_and_save

        generate_and_save(env_name, pair_name, 1000)
    elif function_number == 3:
        from src.data_loading.preference_dataloader import get_dataloader
        from src.reward_learning.multilayer_perceptron import (
            BradleyTerryLoss,
            initialize_network,
            learn,
        )

        save_path = f"model/{env_name}/{pair_name}_{reward_model_name}.pth"

        data_loader, obs_dim, act_dim = get_dataloader(
            env_name=env_name, pair_name=pair_name
        )

        print("obs_dim:", obs_dim, "act_dim:", act_dim)

        if reward_model_name == "MLP":
            model, optimizer = initialize_network(obs_dim, act_dim, path=save_path)
            loss_fn = BradleyTerryLoss()

            num_epochs = 30
            loss_history = learn(
                model,
                optimizer,
                data_loader,
                loss_fn,
                num_epochs=num_epochs,
            )

            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(model.state_dict(), save_path)

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
