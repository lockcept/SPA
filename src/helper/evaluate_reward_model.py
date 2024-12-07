import csv
import glob
import os
from typing import List
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from data_loading import load_dataset, get_dataloader
from reward_learning import RewardModelBase, MR
from utils import get_reward_model_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_pearson_correlation_with_d4rl(
    env_name, models: List[RewardModelBase], output_name
):
    dataset = load_dataset(env_name)

    observations = dataset["observations"]
    actions = dataset["actions"]
    rewards = dataset["rewards"]
    termainals = dataset["terminals"] | dataset["timeouts"]

    obs_tensor = torch.tensor(observations, dtype=torch.float32).to(device)
    act_tensor = torch.tensor(actions, dtype=torch.float32).to(device)

    with torch.no_grad():
        predicted_rewards_list = []
        for model in models:
            predicted_rewards = model.batched_forward_trajectory(
                obs_batch=obs_tensor, act_batch=act_tensor
            )
            predicted_rewards_list.append(predicted_rewards)

        mean_predicted_rewards = torch.mean(torch.stack(predicted_rewards_list), dim=0)

    actual_rewards_np = rewards[~termainals]
    mean_predicted_rewards_np = mean_predicted_rewards.cpu().numpy()[~termainals]

    pearson_corr, _ = pearsonr(
        mean_predicted_rewards_np.flatten(), actual_rewards_np.flatten()
    )

    plt.figure(figsize=(8, 6))
    plt.scatter(
        actual_rewards_np, mean_predicted_rewards_np, alpha=0.005, label=output_name
    )
    plt.xlabel("Actual Rewards")
    plt.ylabel("Predicted Rewards")
    plt.title(f"Actual vs. Predicted Rewards\nPearson Correlation: {pearson_corr:.2f}")
    plt.legend(loc="upper right")
    plt.grid(True)

    output_path = f"log/reward_PCC_{output_name}.png"
    plt.savefig(output_path, format="png")
    plt.close()

    return pearson_corr


def evaluate_reward_model(
    env_name, models: List[RewardModelBase], data_loader, output_name
):

    correct_predictions = 0
    total_samples = 0
    mse_loss = torch.nn.MSELoss()
    cumulative_mse = 0

    with torch.no_grad():
        for batch in data_loader:
            (
                s0_obs_batch,
                s0_act_batch,
                s1_obs_batch,
                s1_act_batch,
                mu_batch,
                mask_batch,
            ) = [x.to(device) for x in batch]

            rewards_s0_list = []
            rewards_s1_list = []

            for model in models:

                rewards_s0 = model.batched_forward_trajectory(
                    s0_obs_batch, s0_act_batch
                )
                rewards_s1 = model.batched_forward_trajectory(
                    s1_obs_batch, s1_act_batch
                )

                rewards_s0_list.append(rewards_s0 * (1 - mask_batch))
                rewards_s1_list.append(rewards_s1 * (1 - mask_batch))

            mean_rewards_s0 = torch.mean(torch.stack(rewards_s0_list), dim=0)
            mean_rewards_s1 = torch.mean(torch.stack(rewards_s1_list), dim=0)

            sum_rewards_s0 = torch.sum(mean_rewards_s0, dim=1)
            sum_rewards_s1 = torch.sum(mean_rewards_s1, dim=1)

            pred_probs_s1 = 1 / (1 + torch.exp(sum_rewards_s0 - sum_rewards_s1))

            mu_batch = mu_batch.unsqueeze(1)

            correct_predictions += torch.sum(
                ((pred_probs_s1 < 0.5) & (mu_batch < 0.5))
                | ((pred_probs_s1 >= 0.5) & (mu_batch >= 0.5))
            ).item()

            mse_batch = mse_loss(pred_probs_s1, mu_batch)
            cumulative_mse += mse_batch.item() * mu_batch.size(0)

            total_samples += mu_batch.size(0)

    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    avg_mse = cumulative_mse / total_samples if total_samples > 0 else 0
    pearson_corr = calculate_pearson_correlation_with_d4rl(
        env_name, models, output_name=output_name
    )

    print(f"Correct predictions: {correct_predictions}")
    print(f"Total samples: {total_samples}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Mean Squared Error (MSE): {avg_mse:.6f}")
    print(
        f"Pearson correlation between predicted rewards and actual rewards: {pearson_corr:.4f}"
    )

    return accuracy, avg_mse, pearson_corr


def evaluate_and_log_reward_models(
    env_name,
    exp_name,
    pair_algo,
    reward_model_algo,
):
    log_path = "log/main_evaluate_reward.csv"
    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    model_path_pattern = get_reward_model_path(
        env_name=env_name,
        exp_name=exp_name,
        pair_algo=pair_algo,
        reward_model_algo=reward_model_algo,
        reward_model_tag="*",
    )
    model_files = glob.glob(model_path_pattern)

    data_loader = get_dataloader(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="test",
        pair_algo="full-binary",
        drop_last=False,
    )

    obs_dim, act_dim = data_loader.dataset.get_dimensions()

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
        output_name=f"{env_name}_{exp_name}_{pair_algo}_{reward_model_algo}",
    )

    with open(log_path, "a", encoding="utf-8", newline="") as log_file:
        writer = csv.writer(log_file)

        if log_file.tell() == 0:
            writer.writerow(
                [
                    "EnvName",
                    "ExpName",
                    "PairAlgo",
                    "RewardModelAlgo",
                    "Accuracy",
                    "MSE",
                    "PCC",
                ]
            )
        writer.writerow(
            f"{env_name}, {exp_name}, {pair_algo},{reward_model_algo}, {accuracy:.4f}, {mse:.6f}, {pcc:.4f}\n"
        )
