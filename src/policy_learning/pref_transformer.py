import torch
import torch.nn as nn
import torch.nn.functional as F

class PrefTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=256, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: (B, T, input_dim)
        h = self.input_proj(x)
        h = self.transformer(h)
        return h

class PreferencePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.encoder = PrefTransformer(
            input_dim=input_dim,
            embed_dim=hidden_dim,
            num_heads=n_heads,
            num_layers=n_layers,
            dropout=dropout,
        )
        self.compare_layer = nn.Linear(hidden_dim * 2, 1)

    def encode_traj(self, traj):
        h = self.encoder(traj)  # (B, T, hidden)
        return h.mean(dim=1)

    def forward(self, traj0, traj1):
        traj0_emb = self.encode_traj(traj0)
        traj1_emb = self.encode_traj(traj1)
        pair_emb = torch.cat([traj0_emb, traj1_emb], dim=-1)
        return self.compare_layer(pair_emb).squeeze(-1)
