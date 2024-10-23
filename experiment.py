if __name__ == "__main__":
    from src.data_loading.load_d4rl import save_d4rl_dataset
    from src.data_generation.full_scripted_teacher import generate_pairs
    from src.data_loading.preference_dataloader import get_dataloader
    from src.reward_learning.multilayer_perceptron import (
        train,
    )
    from src.helper.evaluate_reward_model import evaluate_reward_model_MLP

    env_name = "maze2d-medium-dense-v1"
    reward_model_name_base = "MLP"

    save_d4rl_dataset(env_name)

    for pair_count in [100, 200, 500, 1000]:
        pair_name_base = f"full_{pair_count}"
        test_pair_name_base = f"test_full"
        generate_pairs(env_name, pair_name_base, pair_count, ["binary", "continuous"])

        for mu_type in ["binary", "continuous"]:
            pair_name = f"{pair_name_base}_{mu_type}"
            test_pair_name = f"{test_pair_name_base}_{mu_type}"
            reward_model_name = f"{pair_name}_{reward_model_name_base}"
            reward_model_path = f"model/{env_name}/{reward_model_name}.pth"

            data_loader, obs_dim, act_dim = get_dataloader(
                env_name=env_name,
                pair_name=pair_name,
            )

            test_data_loader, _, _ = get_dataloader(
                env_name=env_name,
                pair_name=test_pair_name,
            )

            train(
                data_loader=data_loader,
                test_data_loader=test_data_loader,
                reward_model_path=reward_model_path,
                obs_dim=obs_dim,
                act_dim=act_dim,
            )

            evaluate_reward_model_MLP(
                env_name, reward_model_path, eval_pair_name="eval_full_sigmoid"
            )
