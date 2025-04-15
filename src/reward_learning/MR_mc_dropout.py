import csv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from reward_learning.reward_model_base import RewardModelBase

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MRWithMCDropout(RewardModelBase):
    @staticmethod
    def initialize(config, path=None, allow_existing=True):
        obs_dim = config.get("obs_dim")
        act_dim = config.get("act_dim")
        hidden_size = config.get("hidden_size", 256)
        lr = config.get("lr", 0.001)
        dropout = config.get("dropout", 0.1)

        model = MRWithMCDropout(
            config={
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "hidden_size": hidden_size,
                "dropout": dropout,
            },
            path=path,
        )

        if path is not None:
            if os.path.isfile(path):
                if not allow_existing:
                    print("Skipping model initialization because already exists")
                    return None, None
                model.load_state_dict(
                    torch.load(path, weights_only=True, map_location=device)
                )
                print(f"Model loaded from {path}")
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        return model, optimizer

    def __init__(self, config, path):
        super(MRWithMCDropout, self).__init__(config, path)
        obs_dim = config.get("obs_dim")
        act_dim = config.get("act_dim")
        hidden_dim = config.get("hidden_size")
        dropout = config.get("dropout", 0.1)

        self.hidden_layer_1 = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.hidden_layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, obs_t, act_t):
        combined = torch.cat([obs_t, act_t], dim=-1)
        x = F.relu(self.hidden_layer_1(combined))
        x = self.dropout1(x)
        x = F.relu(self.hidden_layer_2(x))
        x = self.dropout2(x)
        reward_t = self.fc(x)
        reward_t = 1 + torch.tanh(reward_t)

        return reward_t

    def forward_mc(self, obs_t, act_t, mc_passes=10):
        self.train()  # dropout 작동을 위해 train 유지
        preds = []
        for _ in range(mc_passes):
            with torch.no_grad():
                preds.append(self(obs_t, act_t).unsqueeze(0))
        preds = torch.cat(preds, dim=0)
        return preds.mean(dim=0), preds.std(dim=0)

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
                    mask0_batch,
                    mask1_batch,
                ) = [x.to(device) for x in batch]

                rewards_s0 = self(s0_obs_batch, s0_act_batch)
                rewards_s1 = self(s1_obs_batch, s1_act_batch)

                loss = loss_fn(
                    rewards_s0, rewards_s1, mu_batch, mask0_batch, mask1_batch
                )
                epoch_loss += loss.item()
                num_batches += 1

        return epoch_loss / num_batches

    def _learn(
        self, optimizer, loss_fn, train_data_loader, val_data_loader=None, num_epochs=10
    ):
        with open(self.log_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Train Loss", "Validation Loss"])

        loss_history = []
        val_loss_history = []
        best_train_loss = float("inf")

        for epoch in tqdm(range(num_epochs), desc="learning MR with MC Dropout"):
            self.train()
            epoch_loss = 0.0

            for batch in train_data_loader:
                (
                    s0_obs_batch,
                    s0_act_batch,
                    s1_obs_batch,
                    s1_act_batch,
                    mu_batch,
                    mask0_batch,
                    mask1_batch,
                ) = [x.to(device) for x in batch]

                rewards_s0 = self(s0_obs_batch, s0_act_batch)
                rewards_s1 = self(s1_obs_batch, s1_act_batch)

                loss = loss_fn(
                    rewards_s0, rewards_s1, mu_batch, mask0_batch, mask1_batch
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(train_data_loader)
            loss_history.append(avg_epoch_loss)

            if val_data_loader is not None:
                val_loss = self.evaluate(val_data_loader, loss_fn)
                val_loss_history.append(val_loss)
            else:
                val_loss = 0.0

            with open(self.log_path, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([epoch + 1, avg_epoch_loss, val_loss])

            if avg_epoch_loss < best_train_loss:
                best_train_loss = avg_epoch_loss
                torch.save(self.state_dict(), self.path)
                print(f"New best model saved (Train loss: {avg_epoch_loss:.4f})")

    def train_model(self, optimizer, train_loader, val_loader, num_epochs):
        loss_fn = LinearLoss()

        print("[Train started] reward_model_path:", self.path)
        self._learn(optimizer, loss_fn, train_loader, val_loader, num_epochs)
        print("Training completed")


class LinearLoss(nn.Module):
    def __init__(self):
        super(LinearLoss, self).__init__()
        self.cross_entropy_loss = nn.BCELoss()

    def forward(self, rewards_s0, rewards_s1, mu, mask0, mask1):
        reward_s0_sum = torch.sum(rewards_s0 * (1 - mask0), dim=1)
        reward_s1_sum = torch.sum(rewards_s1 * (1 - mask1), dim=1)
        linear_ratio = reward_s1_sum / (reward_s0_sum + reward_s1_sum + 1e-6)
        return self.cross_entropy_loss(linear_ratio.squeeze(), mu)
