import torch
import torch.nn as nn

class CEMHead(nn.Module):
    def __init__(self, input_dim, concept_dim, hidden_dim=256):
        super().__init__()
        self.cem_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, concept_dim),
            nn.Sigmoid()  # ogni concetto è attivo (1) o no (0)
        )

    def forward(self, ego_feat):
        """
        ego_feat: [B, C, T, 1, 1]
        Output: [B, T, C] (frame x concept)
        """
        B, C, T, _, _ = ego_feat.shape
        ego_feat = ego_feat.view(B, C, T).permute(0, 2, 1)  # [B, T, C]
        out = self.cem_mlp(ego_feat)  # [B, T, concept_dim]
        return out
