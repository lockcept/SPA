import csv

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data_loading import get_dataloader, load_dataset
from utils import get_encoder_model_path, get_classifier_model_path
from auto_encoder import AutoEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Classifier(nn.Module):
    """
    Classifier that uses a pre-trained Autoencoder's encoder
    """

    def __init__(self, encoder, latent_dim=64):
        super(Classifier, self).__init__()

        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Softmax(dim=1),
        )

    def forward(self, s0, s1):
        s0_encoded = self.encoder(s0)
        s1_encoded = self.encoder(s1)
        combined = torch.cat((s0_encoded, s1_encoded), dim=1)
        output = self.classifier(combined)
        return output


def train_classifier(
    env_name, exp_name, pair_algo, num_epochs=200, batch_size=64, lr=0.001
):
    """
    Train classifier using the pre-trained autoencoder's encoder
    """

    model_path = get_classifier_model_path(env_name, exp_name, pair_algo)
    log_path = f"model/{env_name}/{exp_name}/classifier/{pair_algo}/epoch_loss_log.csv"

    with open(log_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Loss", "Accuracy"])

    dataset = load_dataset(env_name)

    obs_dim, act_dim = dataset["observations"].shape[1], dataset["actions"].shape[1]
    traj_len = 25
    input_dim = (obs_dim + act_dim) * traj_len

    dataloader = get_dataloader(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_algo=pair_algo,
        batch_size=batch_size,
    )

    autoencoder_path = get_encoder_model_path(env_name)
    autoencoder = AutoEncoder(input_dim=input_dim)
    autoencoder.load_state_dict(
        torch.load(autoencoder_path, weights_only=True, map_location=device)
    )
    encoder = autoencoder.encoder.to(device)
    encoder.eval()

    for param in encoder.parameters():
        param.requires_grad = False

    model = Classifier(encoder).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(
            total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", leave=True
        )

        model.train()

        for _, batch in enumerate(dataloader):
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

            s0_batch = s0_batch.view(batch_size, -1)
            s1_batch = s1_batch.view(batch_size, -1)

            # 0 > 1 -> class 0
            # 0 < 1 -> class 1
            # else -> class 2
            mu_class = torch.full_like(mu_batch, 2, dtype=torch.long)
            mu_class[mu_batch == 0] = 0
            mu_class[mu_batch == 1] = 1

            optimizer.zero_grad()
            output = model(s0_batch, s1_batch)
            loss = criterion(output, mu_class)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            _, predicted = torch.max(output, 1)
            correct += (predicted == mu_class).sum().item()
            total += mu_batch.size(0)

            progress_bar.update(1)
            progress_bar.set_postfix(
                {"Loss": f"{loss.item():.6f}", "Acc": f"{(correct / total) * 100:.2f}%"}
            )

        progress_bar.close()

        avg_loss = epoch_loss / len(dataloader)
        accuracy = (correct / total) * 100
        print(
            f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%"
        )

        with open(log_path, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, avg_loss, accuracy])

    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")


def evaluate_classifier(model, dataloader, device):
    """
    Evaluate the classifier's performance on a validation dataset.

    Args:
        model (torch.nn.Module): The trained classifier model.
        dataloader (torch.utils.data.DataLoader): DataLoader for validation data.
        device : Device to run the evaluation on.

    Returns:
        avg_loss (float): Average validation loss.
        accuracy (float): Classification accuracy in percentage.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            (
                s0_obs_batch,
                s0_act_batch,
                s1_obs_batch,
                s1_act_batch,
                mu_batch,
                _,
                _,
            ) = [x.to(device) for x in batch]

            s0_batch = torch.cat((s0_obs_batch, s0_act_batch), dim=-1).view(
                s0_obs_batch.shape[0], -1
            )
            s1_batch = torch.cat((s1_obs_batch, s1_act_batch), dim=-1).view(
                s1_obs_batch.shape[0], -1
            )

            mu_class = torch.full_like(mu_batch, 2, dtype=torch.long)
            mu_class[mu_batch == 0] = 0
            mu_class[mu_batch == 1] = 1

            output = model(s0_batch, s1_batch)

            _, predicted = torch.max(output, 1)
            correct += (predicted == mu_class).sum().item()
            total += mu_batch.size(0)

    accuracy = (correct / total) * 100
    return accuracy
