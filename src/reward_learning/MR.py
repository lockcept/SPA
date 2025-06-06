import csv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from reward_learning.reward_model_base import RewardModelBase

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MR(RewardModelBase):

    @staticmethod
    def initialize(config, path=None, allow_existing=True, linear_loss=False):
        obs_dim = config.get("obs_dim")
        act_dim = config.get("act_dim")
        hidden_size = config.get("hidden_size", 256)
        lr = config.get("lr", 0.001)

        model = MR(
            config={"obs_dim": obs_dim, "act_dim": act_dim, "hidden_size": hidden_size},
            path=path,
            linear_loss=linear_loss,
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

    def __init__(self, config, path, linear_loss=False):
        super(MR, self).__init__(config, path)
        self.linear_loss = linear_loss

        obs_dim = config.get("obs_dim")
        act_dim = config.get("act_dim")
        hidden_dim = config.get("hidden_size")

        self.hidden_layer_1 = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.hidden_layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, obs_t, act_t, return_logit=False):
        combined = torch.cat([obs_t, act_t], dim=-1)

        combined = F.relu(self.hidden_layer_1(combined))
        combined = F.relu(self.hidden_layer_2(combined))

        reward_t_logit = self.fc(combined)
        if self.linear_loss:
            reward_t = 1 + torch.tanh(reward_t_logit)
        else:
            reward_t = torch.tanh(reward_t_logit)
        
        if return_logit:
            return reward_t, reward_t_logit
        return reward_t

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

        avg_epoch_loss = epoch_loss / num_batches
        return avg_epoch_loss

    def batched_forward_trajectory(self, obs_batch, act_batch):
        rewards_batch = self(obs_batch, act_batch)
        return rewards_batch

    def _learn(
        self,
        optimizer,
        loss_fn,
        train_data_loader,
        val_data_loader=None,
        num_epochs=10,
    ):
        with open(self.log_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Train Loss", "Validation Loss"])

        loss_history = []
        val_loss_history = []

        best_train_loss = float("inf") 

        for epoch in tqdm(range(num_epochs), desc="learning MR reward"):
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
                val_loss = self.evaluate(data_loader=val_data_loader, loss_fn=loss_fn)
                val_loss_history.append(val_loss)
            else:
                val_loss = 0.0

            with open(self.log_path, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([epoch + 1, avg_epoch_loss, val_loss])

            # train loss가 최소일 때만 저장
            if avg_epoch_loss < best_train_loss:
                best_train_loss = avg_epoch_loss
                torch.save(self.state_dict(), self.path)
                # print(f"New best model saved (Train loss: {avg_epoch_loss:.4f})")

    def train_model(self, optimizer, train_loader, val_loader, num_epochs):
        loss_fn = None

        if self.linear_loss:
            loss_fn = LinearLoss()
        else:
            loss_fn = BradleyTerryLoss()

        print("[Train started] reward_model_path:", self.path)

        self._learn(
            optimizer=optimizer,
            train_data_loader=train_loader,
            val_data_loader=val_loader,
            loss_fn=loss_fn,
            num_epochs=num_epochs,
        )

        print("Training completed")


class BradleyTerryLoss(nn.Module):
    def __init__(self):
        super(BradleyTerryLoss, self).__init__()
        self.cross_entropy_loss = nn.BCELoss()

    def forward(self, rewards_s0, rewards_s1, mu, mask0, mask1):
        reward_s0_sum = torch.sum(rewards_s0 * (1 - mask0), dim=1)
        reward_s1_sum = torch.sum(rewards_s1 * (1 - mask1), dim=1)

        prob_s1_wins = torch.sigmoid(reward_s1_sum - reward_s0_sum)
        prob_s1_wins = prob_s1_wins.squeeze()

        loss = self.cross_entropy_loss(prob_s1_wins, mu)
        return loss


class LinearLoss(nn.Module):
    def __init__(self):
        super(LinearLoss, self).__init__()
        self.cross_entropy_loss = nn.BCELoss()

    def forward(self, rewards_s0, rewards_s1, mu, mask0, mask1):
        # Apply mask0 and mask1 to compute masked reward sums
        reward_s0_sum = torch.sum(rewards_s0 * (1 - mask0), dim=1)
        reward_s1_sum = torch.sum(rewards_s1 * (1 - mask1), dim=1)

        linear_ratio = (reward_s1_sum) / (reward_s1_sum + reward_s0_sum + 1e-6)
        linear_ratio = linear_ratio.squeeze()

        loss = self.cross_entropy_loss(linear_ratio, mu)
        return loss
