import torch
import os
import sys
from scipy.stats import pearsonr


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.data_loading.load_dataset import load_d4rl_dataset
from data_loading.preference_dataloader import get_dataloader
from reward_learning.multilayer_perceptron import initialize_network


def calculate_pearson_correlation_with_d4rl(env_name, model):
    dataset = load_d4rl_dataset(env_name)

    observations = dataset["observations"][:-1]
    actions = dataset["actions"][:-1]
    next_obs = dataset["observations"][1:]
    rewards = dataset["rewards"][:-1]

    obs_tensor = torch.tensor(observations, dtype=torch.float32)
    act_tensor = torch.tensor(actions, dtype=torch.float32)
    next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32)

    with torch.no_grad():
        predicted_rewards = model(obs_tensor, act_tensor, next_obs_tensor)

    is_terminal = dataset["terminals"][:-1] | dataset["timeouts"][:-1]

    actual_rewards_np = rewards[~is_terminal]
    predicted_rewards_np = predicted_rewards.cpu().numpy()[~is_terminal]

    pearson_corr, _ = pearsonr(
        predicted_rewards_np.flatten(), actual_rewards_np.flatten()
    )

    print(
        f"Pearson correlation between predicted rewards and actual rewards: {pearson_corr:.4f}"
    )


def evaluate_reward_model_MLP(env_name, model_path, eval_pair_name):
    data_loader, obs_dim, act_dim = get_dataloader(
        env_name=env_name,
        pair_name=eval_pair_name,
        drop_last=False,
    )

    model, _ = initialize_network(obs_dim, act_dim, path=model_path)
    model.eval()

    correct_predictions = 0
    total_samples = 0
    mse_loss = torch.nn.MSELoss()
    cumulative_mse = 0

    with torch.no_grad():
        for batch in data_loader:
            (
                s0_obs_batch,
                s0_act_batch,
                s0_obs_next_batch,
                s1_obs_batch,
                s1_act_batch,
                s1_obs_next_batch,
                mu_batch,
            ) = batch

            rewards_s0 = model(s0_obs_batch, s0_act_batch, s0_obs_next_batch)
            rewards_s1 = model(s1_obs_batch, s1_act_batch, s1_obs_next_batch)

            sum_rewards_s0 = torch.sum(rewards_s0, dim=1)
            sum_rewards_s1 = torch.sum(rewards_s1, dim=1)

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

    print(f"Correct predictions: {correct_predictions}")
    print(f"Total samples: {total_samples}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Mean Squared Error (MSE): {avg_mse:.6f}")

    calculate_pearson_correlation_with_d4rl(env_name, model)
