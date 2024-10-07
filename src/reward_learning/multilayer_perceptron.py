import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class RewardNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super(RewardNetwork, self).__init__()

        self.obs_layer = nn.Linear(obs_dim, hidden_dim)
        self.act_layer = nn.Linear(act_dim, hidden_dim)
        self.obs_next_layer = nn.Linear(obs_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim * 3, 1)

    def forward(self, obs_t, act_t, obs_t_next):
        obs_t = F.relu(self.obs_layer(obs_t))
        act_t = F.relu(self.act_layer(act_t))
        obs_t_next = F.relu(self.obs_next_layer(obs_t_next))

        combined = torch.cat([obs_t, act_t, obs_t_next], dim=-1)

        reward_t = self.fc(combined)
        return reward_t


class BradleyTerryLoss(nn.Module):
    def __init__(self):
        super(BradleyTerryLoss, self).__init__()
        self.cross_entropy_loss = nn.BCELoss()

    def forward(self, reward_s0, reward_s1, mu):
        prob_s0_wins = torch.sigmoid(reward_s0 - reward_s1)

        loss = self.cross_entropy_loss(prob_s0_wins, mu)
        return loss


def learn(model, optimizer, data_loader, loss_fn, num_epochs=10):
    loss_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

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

            reward_s0 = model(s0_obs_batch, s0_act_batch, s0_obs_next_batch)
            reward_s1 = model(s1_obs_batch, s1_act_batch, s1_obs_next_batch)

            loss = loss_fn(reward_s0, reward_s1, mu_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_epoch_loss = epoch_loss / num_batches
        loss_history.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")

    return loss_history


def initialize_network(obs_dim, act_dim, hidden_size=64, lr=0.001):
    model = RewardNetwork(obs_dim, act_dim, hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    return model, optimizer
