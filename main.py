import argparse


DEFAULT_ENV_NAME = "maze2d-medium-dense-v1"
DEFAULT_PAIR_NAME = "full_prefernce_pairs.npz"


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
    # -1: helper analyze d4rl
    # 0: do nothing
    # 1: load and save d4rl
    # 2: load and save preference pairs from full_scripted_teacher
    # 3: MLP
    parser.add_argument(
        "--function_number",
        type=int,
        default=0,
        help="Number of the function to execute",
    )

    args = parser.parse_args()

    if args.function_number == 0:
        print("Pass")
        pass
    elif args.function_number == -1:
        from src.helper.analyze_d4rl import analyze

        analyze(args.env_name)
    elif args.function_number == 1:
        from src.data_loading.load_d4rl import load

        load(args.env_name)
    elif args.function_number == 2:
        from src.data_generation.full_scripted_teacher import generate_and_save

        generate_and_save(args.env_name)
    elif args.function_number == 3:

        from src.data_loading.preference_dataloader import get_dataloader
        from src.reward_learning.multilayer_perceptron import (
            BradleyTerryLoss,
            initialize_network,
            learn,
        )

        data_loader, obs_dim, act_dim = get_dataloader(
            env_name="maze2d-medium-dense-v1", pair_name="full_preference_pairs.npz"
        )

        print("Observation Dimension:", obs_dim)
        print("Action Dimension:", act_dim)
        model, optimizer = initialize_network(data_loader, obs_dim, act_dim)

        loss_fn = BradleyTerryLoss()

        num_epochs = 10
        loss_history = learn(
            model, optimizer, data_loader, loss_fn, num_epochs=num_epochs
        )

        # 최종 학습 결과 확인
        print("Training completed. Loss history:", loss_history)
