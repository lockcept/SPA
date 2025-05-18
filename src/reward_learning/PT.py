import csv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from reward_learning.reward_model_base import RewardModelBase

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PreferenceAttentionLayer(nn.Module):
    def __init__(self, embd_dim):
        super().__init__()
        self.proj = nn.Linear(embd_dim, 2 * embd_dim + 1)

    def forward(self, x):  # x: (B, T, D)
        B, T, D = x.shape
        proj = self.proj(x)  # (B, T, 2D + 1)
        q, k, v = torch.split(proj, [D, D, 1], dim=-1)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (D**0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        weighted_sum = torch.matmul(attn_weights, v)
        reward = weighted_sum.mean(dim=1).squeeze(-1)
        return reward, attn_weights


class CausalLayer(nn.Module):
    def __init__(self, embd_dim, n_head, dropout):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=embd_dim,
            nhead=n_head,
            dim_feedforward=4 * embd_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        return self.layer(
            x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )


class PreferenceTransformer(nn.Module):
    def __init__(
        self,
        obs_dim,
        act_dim,
        embd_dim=256,
        n_layer=3,
        n_head=1,
        dropout=0.1,
        max_len=64,
    ):
        super().__init__()
        self.embd_dim = embd_dim

        self.embed_state = nn.Linear(obs_dim, embd_dim)
        self.embed_action = nn.Linear(act_dim, embd_dim)
        self.embed_timestep = nn.Embedding(max_len, embd_dim)

        self.encoder_layers = nn.ModuleList(
            [CausalLayer(embd_dim, n_head, dropout) for _ in range(n_layer)]
        )

        self.attn = PreferenceAttentionLayer(embd_dim)

    def forward(
        self, obs, act, timestep=None, attn_mask=None, reverse=False, target_idx=1
    ):
        B, T, _ = obs.shape
        if attn_mask is None:
            attn_mask = torch.ones((B, T), dtype=torch.float32, device=obs.device)

        s_emb = self.embed_state(obs)
        a_emb = self.embed_action(act)

        if timestep is None:
            timestep = torch.arange(T, device=obs.device).unsqueeze(0).repeat(B, 1)
        t_emb = self.embed_timestep(timestep)

        s_emb = s_emb + t_emb
        a_emb = a_emb + t_emb

        if reverse:
            stacked = torch.stack([s_emb, a_emb], dim=2)
        else:
            stacked = torch.stack([a_emb, s_emb], dim=2)

        x = stacked.reshape(B, 2 * T, self.embd_dim)
        x = F.layer_norm(x, x.shape[-1:])

        attn_mask = attn_mask.repeat_interleave(2, dim=1)
        padding_mask_bool = ~attn_mask.bool()
        padding_mask_float = padding_mask_bool.float()
        padding_mask_float = padding_mask_float.masked_fill(
            padding_mask_float == 1.0, float("-10000")
        )

        causal_mask = torch.triu(
            torch.full((2 * T, 2 * T), float("-inf"), device=obs.device), diagonal=1
        )

        for i, layer in enumerate(self.encoder_layers):
            x = layer(x, src_mask=causal_mask, src_key_padding_mask=padding_mask_float)
            if torch.isnan(x).any():
                print(f"⚠️ NaN after encoder layer {i}")

        h = x.reshape(B, T, 2, self.embd_dim).transpose(1, 2)[:, target_idx]
        score, attn = self.attn(h)
        if torch.isnan(score).any():
            print("⚠️ NaN in weighted_sum!")
        return {"weighted_sum": score, "value": h.squeeze(-1)}, attn


class PT(RewardModelBase):
    @staticmethod
    def initialize(config, path=None, allow_existing=True, linear_loss=False):
        obs_dim = config.get("obs_dim")
        act_dim = config.get("act_dim")
        lr = config.get("lr", 1e-4)

        model = PT(obs_dim=obs_dim, act_dim=act_dim, path=path, linear_loss=linear_loss)

        if path is not None and os.path.isfile(path):
            if not allow_existing:
                print("Skipping model initialization because already exists")
                return None, None
            model.load_state_dict(
                torch.load(path, map_location=device, weights_only=True)
            )
            print(f"Model loaded from {path}")

        model = model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        return model, optimizer

    def __init__(self, obs_dim, act_dim, path, linear_loss=False):
        super().__init__({}, path)
        if linear_loss:
            self.loss_fn = LinearLoss()
        else:
            self.loss_fn = BradleyTerryLoss()
        self.model = PreferenceTransformer(obs_dim=obs_dim, act_dim=act_dim)

    def forward(self, obs_t, act_t, timestep=None, attn_mask=None):
        B, T, _ = obs_t.shape
        if timestep is None:
            timestep = torch.arange(T, device=obs_t.device).unsqueeze(0).repeat(B, 1)
        out, _ = self.model(obs_t, act_t, timestep, attn_mask)
        return out["weighted_sum"]

    def evaluate(self, data_loader, loss_fn=None):
        self.eval()
        epoch_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in data_loader:
                s0_obs, s0_act, s1_obs, s1_act, mu, *_ = [x.to(device) for x in batch]
                r0 = self(s0_obs, s0_act)
                r1 = self(s1_obs, s1_act)
                loss = self.loss_fn(r0, r1, mu)
                epoch_loss += loss.item()
                num_batches += 1

        return epoch_loss / num_batches

    def train_model(self, optimizer, train_loader, val_loader=None, num_epochs=10000):
        print("[Train started] reward_model_path:", self.path)
        best_train_loss = float("inf")

        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train_loss", "Validation Loss"])

        for epoch in tqdm(range(num_epochs), desc="Training PT reward"):
            self.train()
            epoch_loss = 0.0

            for batch in train_loader:
                s0_obs, s0_act, s1_obs, s1_act, mu, *_ = [x.to(device) for x in batch]
                r0 = self(s0_obs, s0_act)
                r1 = self(s1_obs, s1_act)
                loss = self.loss_fn(r0, r1, mu)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(train_loader)
            with open(self.log_path, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([epoch + 1, avg_epoch_loss, 0.0])

            if avg_epoch_loss < best_train_loss:
                best_train_loss = avg_epoch_loss
                torch.save(self.state_dict(), self.path)

        print("Training completed")


class BradleyTerryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = nn.BCELoss()

    def forward(self, rewards_s0, rewards_s1, mu):
        prob_s1_wins = torch.sigmoid(rewards_s1 - rewards_s0).squeeze()
        diff = rewards_s1 - rewards_s0
        prob_s1_wins = torch.sigmoid(diff)
        if not torch.all((0.0 < prob_s1_wins) & (prob_s1_wins < 1.0)):
            print("⚠️ WARNING: Some sigmoid outputs are outside (0, 1) range!")
        return self.cross_entropy_loss(prob_s1_wins, mu)


class LinearLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = nn.BCELoss()

    def forward(self, rewards_s0, rewards_s1, mu):
        rewards_s0 = 1.0 + torch.tanh(rewards_s0)
        rewards_s1 = 1.0 + torch.tanh(rewards_s1)

        linear_ratio = (rewards_s1 + 1e-6) / (rewards_s1 + rewards_s0 + 2e-6)

        return self.cross_entropy_loss(linear_ratio, mu)
