import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data_loading import get_dataloader, load_dataset
from utils import (
    get_encoder_model_path,
    get_binary_classifier_model_path,
)
from auto_encoder import AutoEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
same_threshold = 0.66


class BinaryClassifier(nn.Module):
    """
    Classifier that uses a pre-trained Autoencoder's encoder
    """

    def __init__(self, encoder, latent_dim=64):
        super(BinaryClassifier, self).__init__()

        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, s0, s1):
        s0_encoded = self.encoder(s0)
        s1_encoded = self.encoder(s1)
        combined = torch.cat((s0_encoded, s1_encoded), dim=1)
        output = self.classifier(combined)
        return output


def get_binary_classifier_model(env_name, exp_name, pair_algo, latent_dim=64):
    """
    Load the trained classifier model.
    """
    model_path = get_binary_classifier_model_path(env_name, exp_name, pair_algo)
    autoencoder_path = get_encoder_model_path(env_name)

    autoencoder = AutoEncoder(input_dim=43 * 25)
    autoencoder.load_state_dict(
        torch.load(autoencoder_path, weights_only=True, map_location=device)
    )
    encoder = autoencoder.encoder.to(device)
    encoder.eval()

    model = BinaryClassifier(encoder, latent_dim=latent_dim)
    model.load_state_dict(
        torch.load(model_path, weights_only=True, map_location=device)
    )
    model.to(device)
    model.eval()

    return model


def train_binary_classifier(
    env_name,
    exp_name,
    pair_algo,
    num_epochs=200,
    batch_size=64,
    lr=0.001,
    remove_if_exists=True,
):
    """
    Train classifier using the pre-trained autoencoder's encoder
    """

    model_path = get_binary_classifier_model_path(env_name, exp_name, pair_algo)

    if remove_if_exists and os.path.exists(model_path):
        os.remove(model_path)

    log_path = (
        f"model/{env_name}/{exp_name}/classifier/{pair_algo}/binary_epoch_loss_log.csv"
    )

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
        shuffle=True,
        drop_last=False,
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

    model = BinaryClassifier(encoder).to(device)
    # load if exist
    if os.path.exists(model_path):
        model.load_state_dict(
            torch.load(model_path, weights_only=True, map_location=device)
        )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(
            total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
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

            batch_dim = s0_batch.shape[0]

            s0_batch = s0_batch.view(batch_dim, -1)
            s1_batch = s1_batch.view(batch_dim, -1)

            mu_class = torch.stack([1 - mu_batch, mu_batch], dim=1).float()

            optimizer.zero_grad()
            output = model(s0_batch, s1_batch)
            loss = criterion(output, mu_class)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            probabilities = torch.softmax(output, dim=1)

            predicted = torch.full_like(mu_batch, 2, dtype=torch.long)
            predicted[probabilities[:, 1] < 1 - same_threshold] = 0
            predicted[probabilities[:, 1] > same_threshold] = 1

            mu_class_labels = torch.full_like(mu_batch, 2, dtype=torch.long)
            mu_class_labels[mu_batch == 0] = 0
            mu_class_labels[mu_batch == 1] = 1

            correct += (predicted == mu_class_labels).sum().item()
            total += mu_batch.size(0)

            progress_bar.update(1)
            progress_bar.set_postfix(
                {"Loss": f"{loss.item():.6f}", "Acc": f"{(correct / total) * 100:.2f}%"}
            )

        progress_bar.close()

        avg_loss = epoch_loss / len(dataloader)
        accuracy = (correct / total) * 100

        with open(log_path, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, avg_loss, accuracy])

    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")


def evaluate_binary_classifier(model, dataloader, device):
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

    filtered_correct = 0
    filtered_total = 0

    prob_threshold = 0.99999

    correct_class_probs = {0: [], 1: [], 2: []}
    wrong_class_probs = {0: [], 1: [], 2: []}

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

            probabilities = torch.softmax(output, dim=1)
            print(probabilities)

            predicted = torch.full_like(mu_batch, 2, dtype=torch.long)
            predicted[probabilities[:, 1] < 1 - same_threshold] = 0
            predicted[probabilities[:, 1] > same_threshold] = 1

            correct += ((predicted == mu_class)).sum().item()
            total += (mu_class > -1).sum().item()

            max_prob, predicted = torch.max(probabilities, 1)

            correct_mask = predicted == mu_class
            wrong_mask = ~correct_mask

            for cls in [0, 1, 2]:
                class_mask = mu_class == cls  # 해당 클래스에 속하는 샘플 선택

                correct_samples = class_mask & correct_mask
                correct_probs = probabilities[correct_samples]
                if correct_probs.shape[0] > 0:
                    correct_class_probs[cls].append(correct_probs)

                wrong_samples = class_mask & wrong_mask
                wrong_probs = probabilities[wrong_samples]
                if wrong_probs.shape[0] > 0:
                    wrong_class_probs[cls].append(wrong_probs)

            high_confidence_mask = max_prob >= prob_threshold
            filtered_correct += (
                ((predicted == mu_class) & high_confidence_mask).sum().item()
            )
            filtered_total += high_confidence_mask.sum().item()

    def compute_mean_prob(class_probs):
        means = []
        for cls in [0, 1, 2]:
            if len(class_probs[cls]) > 0:
                means.append(
                    torch.cat(class_probs[cls], dim=0).mean(dim=0).cpu().numpy()
                )
            else:
                means.append(np.array([0.0, 0.0]))
        return np.array(means)

    correct_mean_probs = compute_mean_prob(correct_class_probs)
    wrong_mean_probs = compute_mean_prob(wrong_class_probs)

    correct_count = sum(
        tensor.shape[0]
        for cls in correct_class_probs
        for tensor in correct_class_probs[cls]
    )
    wrong_count = sum(
        tensor.shape[0]
        for cls in wrong_class_probs
        for tensor in wrong_class_probs[cls]
    )

    correct_mean_probs = np.round(correct_mean_probs, 2)
    wrong_mean_probs = np.round(wrong_mean_probs, 2)

    print(f"\n정답 {correct_count} 클래스별 평균 확률 (3×2):")
    print(correct_mean_probs)

    print(f"\n오답 {wrong_count} 클래스별 평균 확률 (3×2):")
    print(wrong_mean_probs)

    accuracy = (correct / total) * 100
    print("Correct:", correct, "Total:", total)
    print("Accuracy:", accuracy)

    filtered_accuracy = (filtered_correct / filtered_total) * 100
    print(f"\n신뢰도 기준: {100 * prob_threshold:.2f}%")
    print(f"샘플 개수: {filtered_total}")
    print(f"정답 개수: {filtered_correct}")
    print(f"정답률: {filtered_accuracy:.4f}%")
    return accuracy
