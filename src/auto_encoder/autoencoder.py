import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from auto_encoder.trajectory_dataset import TrajectoryDataset
from data_loading import load_dataset, extract_trajectory_indices
from utils import get_encoder_model_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AutoEncoder(nn.Module):
    """
    AutoEncoder that encodes and decodes the trajectory data
    """

    def __init__(self, input_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def generate_indices(dataset, traj_len=25):
    full_trajectories = extract_trajectory_indices(dataset)

    traj_list = []
    for s, e in full_trajectories:
        for i in range(e - s - traj_len + 1):
            traj_list.append((s + i, s + i + traj_len))

    return traj_list


def train_autoencoder(env_name, num_epochs=50, batch_size=64, lr=0.001):
    """
    Train autoencoder model using mini-batch and limit the number of batches per epoch
    """

    model_path = get_encoder_model_path(env_name)
    log_path = f"model/{env_name}/autoencoder/epoch_loss_log.csv"

    with open(log_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Loss"])

    dataset = load_dataset(env_name)

    obs_dim, act_dim = dataset["observations"].shape[1], dataset["actions"].shape[1]
    traj_len = 25
    input_dim = (obs_dim + act_dim) * traj_len

    all_indices = generate_indices(dataset, traj_len=traj_len)

    traj_dataset = TrajectoryDataset(dataset, all_indices)
    dataloader = DataLoader(
        traj_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    model = AutoEncoder(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        progress_bar = tqdm(
            total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", leave=True
        )

        for _, batch in enumerate(dataloader):
            traj_batch = batch.to(device).view(batch_size, -1)

            optimizer.zero_grad()
            output = model(traj_batch)
            loss = criterion(output, traj_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            progress_bar.update(1)
            progress_bar.set_postfix({"Loss": f"{loss.item():.6f}"})

        progress_bar.close()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.6f}")

        with open(log_path, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, avg_loss])

    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")
