import csv
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderModel(torch.nn.Module):
    @staticmethod
    def initialize(config, path=None, skip_if_exists=True):
        obs_dim = config.get("obs_dim")
        act_dim = config.get("act_dim")
        hidden_size = config.get("hidden_size", 64)
        embedding_size = config.get("embedding_size", 2)
        lr = config.get("lr", 0.001)

        model = EncoderModel(
            config={
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "hidden_size": hidden_size,
                "embedding_size": embedding_size,
            },
            path=path,
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

    def __init__(self, config, path):
        super(EncoderModel, self).__init__()
        self.obs_dim = config.get("obs_dim")
        self.act_dim = config.get("act_dim")
        self.state_dim = self.obs_dim + self.act_dim
        self.input_dim = self.state_dim * 25
        self.hidden_dim = config.get("hidden_size")
        self.embedding_dim = config.get("embedding_size")

        self.loss_fn = BradleyTerryLoss()

        self.enc = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.embedding_dim),
        )

        self.path = path
        self.log_path = self.path.replace(".pth", ".csv")

        save_dir = os.path.dirname(path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def forward(self, x, lengths=None):
        x = x.view(x.size(0), -1)
        x = self.enc(x)

        output = torch.sum(x**2, dim=-1)
        return output

    def evaluate(self, data_loader):
        # 길이 25만 들어온다고 가정
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
                    _,
                    _,
                ) = [x.to(device) for x in batch]

                s0_batch = torch.cat((s0_obs_batch, s0_act_batch), dim=-1)
                s1_batch = torch.cat((s1_obs_batch, s1_act_batch), dim=-1)

                score_s0 = self.forward(s0_batch)
                score_s1 = self.forward(s1_batch)

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
                    _,
                    _,
                ) = [x.to(device) for x in batch]

                s0_batch = torch.cat((s0_obs_batch, s0_act_batch), dim=-1)
                s1_batch = torch.cat((s1_obs_batch, s1_act_batch), dim=-1)

                score_s0 = self.forward(s0_batch)
                score_s1 = self.forward(s1_batch)

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
