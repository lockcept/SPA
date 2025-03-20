import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data_loading import get_dataloader
from utils import get_trajectory_pair_classifier_path


class TrajectoryPairClassifier(nn.Module):
    """
    Binary Classifier for TrajectoryPairDataset.
    - Takes (s0, s1) as input
    - Outputs a probability distribution [p(class 0), p(class 1)]
    """

    def __init__(self, input_dim, hidden_dim=64):
        super(TrajectoryPairClassifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x):
        output = self.classifier(x)
        return output  # No softmax (handled in loss function)


def train_trajectory_pair_classifier(
    env_name, exp_name, pair_algo, num_epochs=10, batch_size=32, lr=0.001, device="cpu"
):
    """
    Train and save a binary classifier using TrajectoryPairDataset.
    """
    model_path = get_trajectory_pair_classifier_path(
        env_name=env_name, exp_name=exp_name, pair_algo=pair_algo
    )

    train_loader = get_dataloader(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_algo=pair_algo,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    obs_dim, act_dim = train_loader.dataset.get_dimensions()
    input_dim = (obs_dim + act_dim) * 25  # 25 time steps
    model = TrajectoryPairClassifier(input_dim=input_dim).to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Loaded existing model from {model_path}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        model.train()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            (
                s0_obs_batch,
                s0_act_batch,
                s1_obs_batch,
                s1_act_batch,
                mu_batch,
                _,
                _,
            ) = [x.to(device) for x in batch]

            batch_dim = s0_obs_batch.shape[0]

            s0_batch = torch.cat((s0_obs_batch, s0_act_batch), dim=-1).reshape(
                batch_dim, -1
            )
            s1_batch = torch.cat((s1_obs_batch, s1_act_batch), dim=-1).reshape(
                batch_dim, -1
            )
            batch = torch.cat((s0_batch, s1_batch), dim=-1)

            label_one_hot = torch.zeros(batch.shape[0], 2).to(device)
            label_one_hot[:, 0] = 1 - mu_batch
            label_one_hot[:, 1] = mu_batch

            optimizer.zero_grad()
            output = model(batch)

            loss = criterion(output, label_one_hot)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved at {model_path}")


