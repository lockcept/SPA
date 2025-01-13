import csv
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMModel(nn.Module):
    @staticmethod
    def initialize(config, path=None, skip_if_exists=True, linear_loss=False):
        obs_dim = config.get("obs_dim")
        act_dim = config.get("act_dim")
        hidden_size = config.get("hidden_size", 256)
        lr = config.get("lr", 0.001)

        model = LSTMModel(
            config={"obs_dim": obs_dim, "act_dim": act_dim, "hidden_size": hidden_size},
            path=path,
            linear_loss=linear_loss,
        )

        if path is not None:
            if os.path.isfile(path):
                if skip_if_exists:
                    print("Skipping model initialization")
                    return None, None
                model.load_state_dict(
                    torch.load(path, weights_only=True, map_location=device)
                )
                print(f"Model loaded from {path}")
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        return model, optimizer

    def __init__(
        self,
        config,
        path,
        linear_loss,
    ):
        super(LSTMModel, self).__init__()

        self.linear_loss = linear_loss
        if self.linear_loss:
            self.loss_fn = LinearLoss()
        else:
            self.loss_fn = BradleyTerryLoss()

        self.obs_dim = config.get("obs_dim")
        self.act_dim = config.get("act_dim")
        self.state_dim = self.obs_dim + self.act_dim
        self.hidden_dim = config.get("hidden_size")

        self.lstm = nn.LSTM(
            input_size=self.state_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.path = path
        self.log_path = self.path.replace(".pth", ".csv")

        save_dir = os.path.dirname(path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def forward(self, trajectory, lengths=None):
        if lengths is not None:
            packed_trajectory = nn.utils.rnn.pack_padded_sequence(
                trajectory, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, (h_n, c_n) = self.lstm(packed_trajectory)
        else:
            _, (h_n, c_n) = self.lstm(trajectory)

        score = self.fc(h_n[-1])
        if self.linear_loss:
            score = 1 + torch.tanh(score)

        return score

    def evaluate(self, data_loader):
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

                s0_batch = torch.cat((s0_obs_batch, s0_act_batch), dim=-1)
                s1_batch = torch.cat((s1_obs_batch, s1_act_batch), dim=-1)

                lengths_s0 = (1 - mask0_batch.squeeze()).sum(dim=1)
                lengths_s1 = (1 - mask1_batch.squeeze()).sum(dim=1)

                score_s0 = self.forward(s0_batch, lengths_s0)
                score_s1 = self.forward(s1_batch, lengths_s1)

                loss = self.loss_fn(score_s0, score_s1, mu_batch)

                epoch_loss += loss.item()
                num_batches += 1

        avg_epoch_loss = epoch_loss / num_batches
        return avg_epoch_loss

    def train_model(
        self,
        optimizer,
        train_data_loader,
        val_data_loader,
        num_epochs=10,
    ):
        with open(self.log_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Train Loss", "Validation Loss"])

        loss_history = []
        val_loss_history = []

        for epoch in tqdm(range(num_epochs), desc="learning score function"):
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

                s0_batch = torch.cat((s0_obs_batch, s0_act_batch), dim=-1)
                s1_batch = torch.cat((s1_obs_batch, s1_act_batch), dim=-1)

                lengths_s0 = (1 - mask0_batch.squeeze()).sum(dim=1)
                lengths_s1 = (1 - mask1_batch.squeeze()).sum(dim=1)

                score_s0 = self.forward(s0_batch, lengths_s0)
                score_s1 = self.forward(s1_batch, lengths_s1)

                loss = self.loss_fn(score_s0, score_s1, mu_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(train_data_loader)
            loss_history.append(avg_epoch_loss)

            val_loss = self.evaluate(data_loader=val_data_loader)
            val_loss_history.append(val_loss)

            with open(self.log_path, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([epoch + 1, avg_epoch_loss, val_loss])

        torch.save(self.state_dict(), self.path)


class BradleyTerryLoss(nn.Module):
    def __init__(self):
        super(BradleyTerryLoss, self).__init__()
        self.cross_entropy_loss = nn.BCELoss()

    def forward(self, score_0, score_1, mu):
        prob_s1_wins = torch.sigmoid(score_1 - score_0)
        prob_s1_wins = prob_s1_wins.squeeze()

        loss = self.cross_entropy_loss(prob_s1_wins, mu)
        return loss


class LinearLoss(nn.Module):
    def __init__(self):
        super(LinearLoss, self).__init__()
        self.cross_entropy_loss = nn.BCELoss()

    def forward(self, score_0, score_1, mu):
        prob_s1_wins = score_1 / (score_0 + score_1 + 1e-6)
        prob_s1_wins = prob_s1_wins.squeeze()

        loss = self.cross_entropy_loss(prob_s1_wins, mu)
        return loss
