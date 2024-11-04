import torch
import os
import sys
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from data_loading.load_dataset import load_d4rl_dataset
from data_loading.preference_dataloader import get_dataloader
from reward_learning.MLP import MLP
from reward_learning.MR import MR


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_pearson_correlation_with_d4rl_MLP(env_name, models, output_name):
    dataset = load_d4rl_dataset(env_name)

    observations = dataset["observations"][:-1]
    actions = dataset["actions"][:-1]
    next_obs = dataset["observations"][1:]
    rewards = dataset["rewards"][:-1]

    obs_tensor = torch.tensor(observations, dtype=torch.float32).to(device)
    act_tensor = torch.tensor(actions, dtype=torch.float32).to(device)
    next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).to(device)

    with torch.no_grad():
        predicted_rewards_list = []
        for model in models:
            predicted_rewards = model(obs_tensor, act_tensor, next_obs_tensor)
            predicted_rewards_list.append(predicted_rewards)

        mean_predicted_rewards = torch.mean(torch.stack(predicted_rewards_list), dim=0)

    is_terminal = dataset["terminals"][:-1] | dataset["timeouts"][:-1]

    actual_rewards_np = rewards[~is_terminal]
    mean_predicted_rewards_np = mean_predicted_rewards.cpu().numpy()[~is_terminal]

    pearson_corr, _ = pearsonr(
        mean_predicted_rewards_np.flatten(), actual_rewards_np.flatten()
    )

    plt.figure(figsize=(8, 6))
    plt.scatter(
        actual_rewards_np, mean_predicted_rewards_np, alpha=0.5, label=output_name
    )
    plt.xlabel("Actual Rewards")
    plt.ylabel("Predicted Rewards")
    plt.title(f"Actual vs. Predicted Rewards\nPearson Correlation: {pearson_corr:.2f}")
    plt.legend()
    plt.grid(True)

    output_path = f"log/reward_PCC_{output_name}.png"
    plt.savefig(output_path, format="png")
    plt.close()

    return pearson_corr


def evaluate_reward_model_MLP(env_name, model_path_list, test_pair_name, output_name):
    data_loader, obs_dim, act_dim = get_dataloader(
        env_name=env_name,
        pair_name=test_pair_name,
        drop_last=False,
    )

    models = []
    for model_path in model_path_list:
        model, _ = MLP.initialize(
            config={"obs_dim": obs_dim, "act_dim": act_dim}, path=model_path
        )
        model.eval()
        models.append(model)

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
            ) = [x.to(device) for x in batch]

            rewards_s0_list = []
            rewards_s1_list = []

            for model in models:
                s0_obs_next_batch = s0_obs_batch[:, 1:, :]
                s1_obs_next_batch = s1_obs_batch[:, 1:, :]

                s0_obs_batch = s0_obs_batch[:, :-1, :]
                s0_act_batch = s0_act_batch[:, :-1, :]
                s1_obs_batch = s1_obs_batch[:, :-1, :]
                s1_act_batch = s1_act_batch[:, :-1, :]

                rewards_s0 = model(s0_obs_batch, s0_act_batch, s0_obs_next_batch)
                rewards_s1 = model(s1_obs_batch, s1_act_batch, s1_obs_next_batch)

                rewards_s0_list.append(rewards_s0)
                rewards_s1_list.append(rewards_s1)

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

            mse_batch = mse_loss(pred_probs_s1, mu_batch.float())
            cumulative_mse += mse_batch.item() * mu_batch.size(0)

            total_samples += mu_batch.size(0)

    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    avg_mse = cumulative_mse / total_samples if total_samples > 0 else 0
    pearson_corr = calculate_pearson_correlation_with_d4rl_MLP(
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


def calculate_pearson_correlation_with_d4rl_MR(env_name, models, output_name):
    dataset = load_d4rl_dataset(env_name)

    observations = dataset["observations"]
    actions = dataset["actions"]
    rewards = dataset["rewards"]

    obs_tensor = torch.tensor(observations, dtype=torch.float32).to(device)
    act_tensor = torch.tensor(actions, dtype=torch.float32).to(device)

    with torch.no_grad():
        predicted_rewards_list = []
        for model in models:
            predicted_rewards = model(obs_tensor, act_tensor)
            predicted_rewards_list.append(predicted_rewards)

        mean_predicted_rewards = torch.mean(torch.stack(predicted_rewards_list), dim=0)

    mean_predicted_rewards_np = mean_predicted_rewards.cpu().numpy()

    pearson_corr, _ = pearsonr(mean_predicted_rewards_np.flatten(), rewards.flatten())

    plt.figure(figsize=(8, 6))
    plt.scatter(rewards, mean_predicted_rewards_np, alpha=0.5, label=output_name)
    plt.xlabel("Actual Rewards")
    plt.ylabel("Predicted Rewards")
    plt.title(f"Actual vs. Predicted Rewards\nPearson Correlation: {pearson_corr:.2f}")
    plt.legend()
    plt.grid(True)

    output_path = f"log/reward_PCC_{output_name}.png"
    plt.savefig(output_path, format="png")
    plt.close()

    return pearson_corr


def evaluate_reward_model_MR(env_name, model_path_list, test_pair_name, output_name):
    data_loader, obs_dim, act_dim = get_dataloader(
        env_name=env_name,
        pair_name=test_pair_name,
        drop_last=False,
    )

    models = []
    for model_path in model_path_list:
        model, _ = MR.initialize(
            config={"obs_dim": obs_dim, "act_dim": act_dim}, path=model_path
        )
        model.eval()
        models.append(model)

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
            ) = [x.to(device) for x in batch]

            rewards_s0_list = []
            rewards_s1_list = []

            for model in models:
                rewards_s0 = model(s0_obs_batch, s0_act_batch)
                rewards_s1 = model(s1_obs_batch, s1_act_batch)

                rewards_s0_list.append(rewards_s0)
                rewards_s1_list.append(rewards_s1)

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

            mse_batch = mse_loss(pred_probs_s1, mu_batch.float())
            cumulative_mse += mse_batch.item() * mu_batch.size(0)

            total_samples += mu_batch.size(0)

    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    avg_mse = cumulative_mse / total_samples if total_samples > 0 else 0
    pearson_corr = calculate_pearson_correlation_with_d4rl_MR(
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
