import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from reward_learning.reward_model_base import RewardModelBase

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(RewardModelBase):

    @staticmethod
    def initialize(config, path=None, skip_if_exists=False):
        obs_dim = config.get("obs_dim")
        act_dim = config.get("act_dim")
        hidden_size = config.get("hidden_size", 256)
        lr = config.get("lr", 0.003)

        model = MLP(
            config={"obs_dim": obs_dim, "act_dim": act_dim, "hidden_size": hidden_size},
            path=path,
        )

        if path is not None:
            if os.path.isfile(path):
                if skip_if_exists:
                    print("Skipping model initialization")
                    return None, None
                model.load_state_dict(torch.load(path, weights_only=True))
                print(f"Model loaded from {path}")
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        return model, optimizer

    def __init__(self, config, path):
        super(MLP, self).__init__(config, path)

        obs_dim = config.get("obs_dim")
        act_dim = config.get("act_dim")
        hidden_dim = config.get("hidden_size")

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

    def batched_forward_trajectory(self, obs_batch, act_batch):
        obs_next_batch = obs_batch[:, 1:, :]
        obs_batch = obs_batch[:, :-1, :]
        act_batch = act_batch[:, :-1, :]
        rewards_batch = self(obs_batch, act_batch, obs_next_batch)

        # add a dummy reward for the last step of tensor
        rewards_batch = torch.cat(
            [rewards_batch, torch.zeros(rewards_batch.shape[0], 1, 1)], dim=1
        )
        return rewards_batch

    def evaluate(self, data_loader, loss_fn):
        self.eval()
        epoch_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in data_loader:
                (
                    s0_obs_batch,
                    s0_act_batch,
                    s1_obs_batch,
                    s1_act_batch,
                    mu_batch,
                ) = [x.to(device) for x in batch]

                s0_obs_next_batch = s0_obs_batch[:, 1:, :]
                s1_obs_next_batch = s1_obs_batch[:, 1:, :]

                s0_obs_batch = s0_obs_batch[:, :-1, :]
                s0_act_batch = s0_act_batch[:, :-1, :]
                s1_obs_batch = s1_obs_batch[:, :-1, :]
                s1_act_batch = s1_act_batch[:, :-1, :]

                rewards_s0 = self(s0_obs_batch, s0_act_batch, s0_obs_next_batch)
                rewards_s1 = self(s1_obs_batch, s1_act_batch, s1_obs_next_batch)

                loss = loss_fn(rewards_s0, rewards_s1, mu_batch)

                epoch_loss += loss.item()
                num_batches += 1

        avg_epoch_loss = epoch_loss / num_batches
        return avg_epoch_loss

    def _learn(
        self,
        optimizer,
        train_data_loader,
        val_data_loader,
        loss_fn,
        num_epochs=10,
    ):
        best_loss = float("inf")
        loss_history = []
        val_loss_history = []

        for epoch in range(num_epochs):
            self.train()
            epoch_loss = 0.0

            for batch in train_data_loader:
                (
                    s0_obs_batch,
                    s0_act_batch,
                    s1_obs_batch,
                    s1_act_batch,
                    mu_batch,
                ) = [x.to(device) for x in batch]

                s0_obs_next_batch = s0_obs_batch[:, 1:, :]
                s1_obs_next_batch = s1_obs_batch[:, 1:, :]

                s0_obs_batch = s0_obs_batch[:, :-1, :]
                s0_act_batch = s0_act_batch[:, :-1, :]
                s1_obs_batch = s1_obs_batch[:, :-1, :]
                s1_act_batch = s1_act_batch[:, :-1, :]

                rewards_s0 = self(s0_obs_batch, s0_act_batch, s0_obs_next_batch)
                rewards_s1 = self(s1_obs_batch, s1_act_batch, s1_obs_next_batch)

                loss = loss_fn(rewards_s0, rewards_s1, mu_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(train_data_loader)
            loss_history.append(avg_epoch_loss)

            val_loss = self.evaluate(data_loader=val_data_loader, loss_fn=loss_fn)
            val_loss_history.append(val_loss)

            print(
                f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.state_dict(), self.path)
                print(f"New best model saved with Val loss: {val_loss:.4f}")

    def train_model(self, optimizer, train_loader, val_loader):
        loss_fn = BradleyTerryLoss()

        print("[Train started] reward_model_path:", self.path)

        self._learn(
            optimizer=optimizer,
            train_data_loader=train_loader,
            val_data_loader=val_loader,
            loss_fn=loss_fn,
            num_epochs=1000,
        )

        print("Training completed")


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
