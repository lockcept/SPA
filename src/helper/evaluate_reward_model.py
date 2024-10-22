import torch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from data_loading.preference_dataloader import get_dataloader
from reward_learning.multilayer_perceptron import initialize_network


def evaluate_reward_model_MLP(env_name, model_path, eval_pair_name):
    data_loader, obs_dim, act_dim = get_dataloader(
        env_name=env_name, pair_name=eval_pair_name, drop_last=False
    )

    model, _ = initialize_network(obs_dim, act_dim, path=model_path)
    model.eval()

    correct_predictions = 0
    total_samples = 0

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

            mu_batch = mu_batch.unsqueeze(1)

            correct_predictions += torch.sum(
                ((sum_rewards_s0 > sum_rewards_s1) & (mu_batch == 0))
                | ((sum_rewards_s0 < sum_rewards_s1) & (mu_batch == 1))
            ).item()

            total_samples += mu_batch.size(0)

    accuracy = correct_predictions / total_samples if total_samples > 0 else 0

    print(f"Correct predictions: {correct_predictions}")
    print(f"Total samples: {total_samples}")
    print(f"Accuracy: {accuracy:.4f}")
