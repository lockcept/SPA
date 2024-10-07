import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.data_loading.load_dataset import load_processed_data
from config import DEFAULT_ENV_NAME


class RewardNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(RewardNet, self).__init__()

        self.s_t_layer = nn.Linear(state_dim, hidden_dim)
        self.a_t_layer = nn.Linear(action_dim, hidden_dim)
        self.s_t1_layer = nn.Linear(state_dim, hidden_dim)

        self.reward_layer = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()

    def forward(self, s_t, a_t, s_t1):
        s_t_out = self.relu(self.s_t_layer(s_t))
        a_t_out = self.relu(self.a_t_layer(a_t))
        s_t1_out = self.relu(self.s_t1_layer(s_t1))

        combined = s_t_out + a_t_out + s_t1_out

        r_t = self.reward_layer(combined)
        return r_t


def bradley_terry(r_t, r_t_prime):
    return torch.sigmoid(r_t - r_t_prime)


def preference_loss(predicted_probs, actual_preferences):
    return nn.BCELoss()(predicted_probs, actual_preferences)


# Example: Data preparation
# Assuming s_t, a_t, s_t1, and the preference label (0 or 1) are available
# s_t, a_t, s_t1 are tensors of shape (batch_size, feature_dim), and preference_label is a tensor of shape (batch_size)
def prepare_data(s_t, a_t, s_t1, preference_label):
    s_t = torch.tensor(s_t).float()
    a_t = torch.tensor(a_t).float()
    s_t1 = torch.tensor(s_t1).float()
    preference_label = torch.tensor(preference_label).float()
    return s_t, a_t, s_t1, preference_label


def learn(env_name):
    data = load_processed_data(env_name)

    state_dim = len(data[0][0][0])
    action_dim = len(data[0])
    hidden_dim = 64
    learning_rate = 1e-3
    num_epochs = 100

    print(state_dim, action_dim, hidden_dim, learning_rate, num_epochs)

    # # Initialize model, optimizer
    # model = RewardNet(state_dim, action_dim, hidden_dim)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # # Assuming processed_data is a list of (s_t, a_t, s_t+1, preference_label) tuples
    # for epoch in range(num_epochs):
    #     total_loss = 0
    #     for s_t, a_t, s_t1, preference_label in data:
    #         s_t, a_t, s_t1, preference_label = prepare_data(
    #             s_t, a_t, s_t1, preference_label
    #         )

    #         r_t = model(s_t, a_t, s_t1)
    #         r_t_prime = model(s_t_prime, a_t_prime, s_t1_prime)  # Alternate trajectory

    #         # Calculate the probability using the Bradley-Terry model
    #         predicted_probs = bradley_terry(r_t, r_t_prime)

    #         # Calculate the preference loss
    #         loss = preference_loss(predicted_probs, preference_label)

    #         # Backpropagation and optimization
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         total_loss += loss.item()

    #     print(
    #         f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(processed_data):.4f}"
    #     )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load and save D4RL dataset for a given environment."
    )

    parser.add_argument(
        "--env_name",
        type=str,
        default=DEFAULT_ENV_NAME,
        help="Name of the environment to load the dataset for",
    )

    args = parser.parse_args()

    learn(env_name=args.env_name)
