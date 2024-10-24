import os


if __name__ == "__main__":
    from src.data_loading.load_d4rl import save_d4rl_dataset
    from src.data_generation.full_scripted_teacher import generate_pairs
    from src.data_loading.preference_dataloader import get_dataloader
    from src.reward_learning.multilayer_perceptron import (
        train,
    )
    from src.helper.evaluate_reward_model import (
        evaluate_reward_model_MLP,
    )

    env_name = "maze2d-medium-dense-v1"
    reward_model_name_base = "MLP"

    save_d4rl_dataset(env_name)

    log_path = "log/experiment.log"
    save_dir = os.path.dirname(log_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    mu_types = ["binary", "continuous", "sigmoid", "sigmoid_0.1", "sigmoid_0.25"]

    for pair_count in [100, 200, 500, 1000]:
        pair_name_base = f"full_{pair_count}"
        eval_pair_name_base = f"eval_full"
        generate_pairs(env_name, pair_name_base, pair_count, mu_types=mu_types)

        for mu_type in mu_types:
            pair_name = f"{pair_name_base}_{mu_type}"
            eval_pair_name = f"{eval_pair_name_base}_{mu_type}"
            reward_model_name = f"{pair_name}_{reward_model_name_base}"
            reward_model_path = f"model/{env_name}/{reward_model_name}.pth"

            data_loader, obs_dim, act_dim = get_dataloader(
                env_name=env_name,
                pair_name=pair_name,
            )

            eval_data_loader, _, _ = get_dataloader(
                env_name=env_name,
                pair_name=eval_pair_name,
            )

            train(
                data_loader=data_loader,
                eval_data_loader=eval_data_loader,
                reward_model_path=reward_model_path,
                obs_dim=obs_dim,
                act_dim=act_dim,
            )

            accuracy, mse, pcc = evaluate_reward_model_MLP(
                env_name, reward_model_path, test_pair_name="test_full_sigmoid"
            )

            with open(log_path, "a") as log_file:
                log_file.write(
                    f"{env_name}, {pair_name_base}, {mu_type}, {accuracy:.4f}, {mse:.6f}, {pcc:.4f}\n"
                )
