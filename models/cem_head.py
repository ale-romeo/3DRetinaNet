import torch
import torch.nn as nn
import torch.nn.functional as F

class CEMHead(nn.Module):
    def __init__(self, input_dim, concept_dim, emb_dim=16):
        super(CEMHead, self).__init__()
        self.input_dim = input_dim     # tipicamente = head_size
        self.concept_dim = concept_dim # numero di concetti (k)
        self.emb_dim = emb_dim         # dimensione embedding m

        # Due generatori di embedding per ogni concetto (attivo / inattivo)
        self.emb_pos = nn.ModuleList([nn.Linear(input_dim, emb_dim) for _ in range(concept_dim)])
        self.emb_neg = nn.ModuleList([nn.Linear(input_dim, emb_dim) for _ in range(concept_dim)])

        # Scoring layer condiviso (senza sigmoid → logits)
        self.scoring_layer = nn.Linear(2 * emb_dim, 1)

    def forward(self, h):
        # Aspettato: [B, C, T, 1, 1] → convertiamo a [B, T, C]
        if h.dim() == 5:
            h = h.squeeze(-1).squeeze(-1)        # [B, C, T]
            h = h.permute(0, 2, 1).contiguous()  # [B, T, C]
        elif h.dim() != 3:
            raise ValueError(f"Expected input shape [B, T, D] or [B, C, T, 1, 1], got {h.shape}")
        
        B, T, D = h.shape

        concept_logits = []
        concept_embeddings = []

        for i in range(self.concept_dim):
            c_pos = F.leaky_relu(self.emb_pos[i](h))  # [B, T, m]
            c_neg = F.leaky_relu(self.emb_neg[i](h))  # [B, T, m]

            # Scoring: concateniamo gli embedding e calcoliamo il logit
            concat = torch.cat([c_pos, c_neg], dim=-1)      # [B, T, 2m]
            logit_i = self.scoring_layer(concat).squeeze(-1)  # [B, T]

            # Bottleneck: ĉ_i = p̂_i * c_pos + (1 - p̂_i) * c_neg
            p_i = torch.sigmoid(logit_i).unsqueeze(-1)         # [B, T, 1]
            c_i = p_i * c_pos + (1 - p_i) * c_neg              # [B, T, m]

            concept_logits.append(logit_i)
            concept_embeddings.append(c_i)

        concept_logits = torch.stack(concept_logits, dim=-1)         # [B, T, k]
        concept_bottleneck = torch.cat(concept_embeddings, dim=-1)   # [B, T, k * m]

        return concept_bottleneck, concept_logits  # ← logits, non sigmoid!
