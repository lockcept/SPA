import os
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

    def forward(self, rewards_s0, rewards_s1, mu):
        reward_s0_sum = torch.sum(rewards_s0, dim=1)
        reward_s1_sum = torch.sum(rewards_s1, dim=1)

        prob_s1_wins = torch.sigmoid(reward_s1_sum - reward_s0_sum)
        prob_s1_wins = prob_s1_wins.squeeze()

        loss = self.cross_entropy_loss(prob_s1_wins, mu)
        return loss


def evaluate(model, data_loader, loss_fn):
    model.eval()
    epoch_loss = 0.0
    num_batches = 0

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

            loss = loss_fn(rewards_s0, rewards_s1, mu_batch)

            epoch_loss += loss.item()
            num_batches += 1

    avg_epoch_loss = epoch_loss / num_batches
    return avg_epoch_loss


def learn(
    model,
    optimizer,
    train_data_loader,
    test_data_loader,
    loss_fn,
    model_path,
    num_epochs=10,
):
    best_loss = float("inf")
    loss_history = []
    test_loss_history = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_data_loader:
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

            loss = loss_fn(rewards_s0, rewards_s1, mu_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_epoch_loss = epoch_loss / num_batches
        loss_history.append(avg_epoch_loss)

        test_loss = evaluate(model, test_data_loader, loss_fn)
        test_loss_history.append(test_loss)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_epoch_loss:.4f}, Test Loss: {test_loss:.4f}"
        )

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved with test loss: {test_loss:.4f}")


def initialize_network(obs_dim, act_dim, hidden_size=64, lr=0.001, path=None):
    model = RewardNetwork(obs_dim, act_dim, hidden_size)
    if path is not None:
        if os.path.isfile(path):
            model.load_state_dict(torch.load(path, weights_only=True))
            print(f"Model loaded from {path}")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    return model, optimizer


def train(data_loader, eval_data_loader, reward_model_path, obs_dim, act_dim):
    save_dir = os.path.dirname(reward_model_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model, optimizer = initialize_network(obs_dim, act_dim, path=reward_model_path)
    loss_fn = BradleyTerryLoss()

    print("[Train started] reward_model_path:", reward_model_path)

    num_epochs = 100
    learn(
        model,
        optimizer,
        data_loader,
        eval_data_loader,
        loss_fn,
        model_path=reward_model_path,
        num_epochs=num_epochs,
    )

    print("Training completed")
