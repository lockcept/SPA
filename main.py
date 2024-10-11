import argparse
import os

import torch


DEFAULT_ENV_NAME = "hopper-medium-v2"
DEFAULT_PAIR_NAME = "full_preference_pairs"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env_name",
        type=str,
        default=DEFAULT_ENV_NAME,
        help="Name of the environment to load the dataset for",
    )

    parser.add_argument(
        "--pair_name",
        type=str,
        default=DEFAULT_PAIR_NAME,
        help="Name of the file to load the preference pairs to",
    )
    # -2: heler evaluate reward model MLP
    # -1: helper analyze d4rl
    # 0: do nothing
    # 1: load and save d4rl
    # 2: load and save preference pairs from full_scripted_teacher
    # 3: train MLP model
    # 4: change MLP reward
    parser.add_argument(
        "--function_number",
        type=int,
        default=0,
        help="Number of the function to execute",
    )

    args = parser.parse_args()
    env_name = args.env_name
    pair_name = args.pair_name
    function_number = args.function_number

    if function_number == 0:
        print("Pass")
        pass
    elif function_number == -1:
        from src.helper.analyze_d4rl import analyze

        analyze(env_name)
    elif function_number == -2:
        from src.helper.evaluate_reward_model import evaluate_reward_model_MLP

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

        save_path = f"model/{env_name}/{pair_name}_MLP.pth"

        data_loader, obs_dim, act_dim = get_dataloader(
            env_name=env_name, pair_name=pair_name
        )

        print("obs_dim:", obs_dim, "act_dim:", act_dim)

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

        save_path = f"model/{env_name}/{pair_name}_MLP.pth"
        dataset_path = f"dataset/{env_name}/MLP_dataset.npz"
        data_loader, obs_dim, act_dim = get_dataloader(
            env_name=env_name, pair_name=pair_name
        )

        print("obs_dim:", obs_dim, "act_dim:", act_dim)

        model, _ = initialize_network(obs_dim, act_dim, path=save_path)
        change_reward(env_name, model, dataset_path)
