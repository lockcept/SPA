import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from auto_encoder.trajectory_pair_dataset import TrajectoryPairDataset
from data_loading import load_dataset, load_pair
from utils.path import get_vae_model_path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VAE(nn.Module):
    """
    Variational AutoEncoder (VAE) that encodes and decodes trajectory pairs.
    """

    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: Outputs mean and log variance
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(64, latent_dim)  # Mean μ
        self.logvar_layer = nn.Linear(64, latent_dim)  # Log variance log(σ²)

        # Decoder: Reconstructs the trajectory
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim * 2),
        )

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std * eps
        """
        std = torch.exp(0.5 * logvar)  # Variance to std
        eps = torch.randn_like(std)  # Random noise
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)

        # Compute mean and log variance
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)

        # Sample latent vector z
        z = self.reparameterize(mu, logvar)

        # Decode back to original space
        decoded = self.decoder(z)

        return decoded, mu, logvar


def loss_function(recon_x, x, mu, logvar):
    """
    VAE loss = reconstruction loss + KL divergence
    """
    mse_loss = nn.MSELoss()(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return mse_loss + 0.01 * kl_loss


def train_vae(
    env_name, exp_name, pair_algo, num_epochs=50, batch_size=64, lr=0.001, latent_dim=64
):
    model_path = get_vae_model_path(
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
    dataset = TrajectoryPairDataset(dataset, pairs, device)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    model = VAE(input_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if os.path.exists(model_path):
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        print(f"Loaded model from {model_path}")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
        )

        for batch in progress_bar:
            batch = batch.to(device)

            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch)

            loss = loss_function(recon_batch, batch, mu, logvar)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.6f}"})

    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")


def evaluate_vae_error(
    env_name, exp_name, pair_algo, pairs, batch_size=64, latent_dim=64
):
    """
    Load the trained VAE model and evaluate on given trajectory pairs.
    Computes the uncertainty (reconstruction error).

    Args:
        env_name (str): Name of the environment
        exp_name (str): Experiment name
        pair_algo (str): Name of the trajectory pair dataset
        pairs (list): List of trajectory pairs [(s1, e1), (s2, e2)]
        batch_size (int, optional): Batch size for evaluation. Default is 64.
        latent_dim (int, optional): Latent space dimension. Default is 32.

    Returns:
        numpy.ndarray: Uncertainty (Reconstruction error in MSE).
    """

    # Load dataset and model
    model_path = get_vae_model_path(
        env_name=env_name, exp_name=exp_name, pair_algo=pair_algo
    )
    dataset = load_dataset(env_name)

    obs_dim, act_dim = dataset["observations"].shape[1], dataset["actions"].shape[1]
    traj_len = 25
    input_dim = (obs_dim + act_dim) * traj_len

    # Convert pairs into Dataset
    test_dataset = TrajectoryPairDataset(dataset, pairs, device)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    # Load VAE model
    model = VAE(input_dim, latent_dim).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    uncertainties = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating VAE"):
            batch = batch.to(device)
            recon_batch, _, _ = model(batch)  # Get reconstructed trajectory

            # Compute reconstruction error (MSE)
            mse_loss = torch.mean((recon_batch - batch) ** 2, dim=1)
            uncertainties.extend(mse_loss.cpu().numpy())

    uncertainty = np.array(uncertainties)

    print(f"Evaluation complete: Mean Uncertainty (MSE): {np.mean(uncertainty):.6f}")

    return list(zip(pairs, uncertainty))
