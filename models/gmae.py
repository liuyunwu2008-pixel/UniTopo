"""Graph Masked Auto-Encoder for fragment-level molecular reconstruction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.nn.models import MLP


class GMAE_Encoder(nn.Module):
    """GMAE encoder: GIN on fragment graph with masked nodes."""

    def __init__(self, node_dim, hidden_dim=300, num_layers=3):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([hidden_dim, hidden_dim, hidden_dim], norm="batch_norm")
            self.convs.append(GINConv(mlp))
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])

    def forward(self, x, edge_index):
        x = self.node_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            x = x + F.relu(norm(conv(x, edge_index)))
        return x


class GMAE_Decoder(nn.Module):
    """GMAE decoder: reconstructs masked node features."""

    def __init__(self, hidden_dim=300, node_dim=8, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([hidden_dim, hidden_dim, hidden_dim], norm="batch_norm")
            self.convs.append(GINConv(mlp))
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.output = nn.Linear(hidden_dim, node_dim)

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x = x + F.relu(norm(conv(x, edge_index)))
        return self.output(x)


class GraphMaskedAutoEncoder(nn.Module):
    """Fragment-level Graph Masked Auto-Encoder (GMAE).

    Masks a random subset of fragment nodes and reconstructs their features.
    """

    def __init__(self, node_dim=8, hidden_dim=300, num_enc_layers=3,
                 num_dec_layers=2, mask_ratio=0.3, mask_token=None):
        super().__init__()
        self.encoder = GMAE_Encoder(node_dim, hidden_dim, num_enc_layers)
        self.decoder = GMAE_Decoder(hidden_dim, node_dim, num_dec_layers)
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.randn(1, node_dim)) if mask_token is None else mask_token
        self.node_dim = node_dim

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        n_nodes = x.size(0)
        device = x.device

        # Random masking
        n_masked = max(1, int(n_nodes * self.mask_ratio))
        mask_idx = torch.randperm(n_nodes, device=device)[:n_masked]
        mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        mask[mask_idx] = True

        # Store original features and apply mask
        x_original = x.clone()
        x_masked = x.clone()
        x_masked[mask] = self.mask_token.expand(n_masked, -1)

        # Encode
        z = self.encoder(x_masked, edge_index)

        # Decode and reconstruct
        x_recon = self.decoder(z, edge_index)

        # Scaled cosine error loss (GraphMAE formulation)
        scale_factor = 1.2
        recon_loss = self._scaled_cosine_loss(x_recon[mask], x_original[mask], scale_factor)

        return x_recon, recon_loss, mask

    def encode(self, data):
        """Encode without masking (for downstream use)."""
        x = self.encoder(data.x, data.edge_index)
        graph_emb = global_mean_pool(x, data.batch)
        return x, graph_emb

    @staticmethod
    def _scaled_cosine_loss(pred, target, gamma=1.2):
        """Scaled cosine error from GraphMAE."""
        cosine_sim = F.cosine_similarity(pred, target, dim=-1)
        loss = ((1 - cosine_sim) ** gamma).mean()
        return loss
