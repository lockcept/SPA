import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from auto_encoder.trajectory_pair_dataset import TrajectoryPairDataset
from data_loading import load_dataset, load_pair
from utils.path import get_ae_model_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AE(nn.Module):
    """
    AutoEncoder (AE) that encodes and decodes trajectory pairs.
    """

    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim * 2),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def loss_function(recon_x, x):
    """
    AutoEncoder Loss = MSE Loss
    """
    return nn.MSELoss()(recon_x, x)


def train_ae(
    env_name, exp_name, pair_algo, num_epochs=50, batch_size=64, lr=0.001, latent_dim=64
):
    model_path = get_ae_model_path(
        env_name=env_name, exp_name=exp_name, pair_algo=pair_algo
    )

    dataset = load_dataset(env_name)
    obs_dim, act_dim = dataset["observations"].shape[1], dataset["actions"].shape[1]
    traj_len = 25
    input_dim = (obs_dim + act_dim) * traj_len

    feedbacks = load_pair(
        env_name=env_name, exp_name=exp_name, pair_type="train", pair_algo=pair_algo
    )
    pairs = [(s0, s1) for s0, s1, _ in feedbacks]

    dataset = TrajectoryPairDataset(dataset, pairs, device=device)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    model = AE(input_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
        )

        for batch in progress_bar:
            batch = batch.to(device)

            optimizer.zero_grad()
            recon_batch = model(batch)

            loss = loss_function(recon_batch, batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.6f}"})

    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")


def evaluate_ae_error(
    env_name, exp_name, pair_algo, pairs, batch_size=64, latent_dim=64
):
    """
    Load the trained AE model and evaluate on given trajectory pairs.
    Computes the reconstruction error (MSE).
    """

    # Load dataset and model
    model_path = get_ae_model_path(
        env_name=env_name, exp_name=exp_name, pair_algo=pair_algo
    )
    dataset = load_dataset(env_name)

    obs_dim, act_dim = dataset["observations"].shape[1], dataset["actions"].shape[1]
    traj_len = 25
    input_dim = (obs_dim + act_dim) * traj_len

    # Convert pairs into Dataset
    test_dataset = TrajectoryPairDataset(dataset, pairs, device=device)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    # Load AE model
    model = AE(input_dim, latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    errors = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating AE"):
            batch = batch.to(device)
            recon_batch = model(batch)  # Get reconstructed trajectory

            # Compute reconstruction error (MSE)
            mse_loss = torch.mean((recon_batch - batch) ** 2, dim=1)
            errors.extend(mse_loss.cpu().numpy())

    error_array = np.array(errors)

    print(
        f"Evaluation complete: Mean Reconstruction Error (MSE): {np.mean(error_array):.6f}"
    )

    return list(zip(pairs, error_array))
