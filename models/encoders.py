"""Four molecular graph encoders: ME, MT, FM, MG."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool
from torch_geometric.nn.models import MLP


class GINBlock(nn.Module):
    """GIN layer with optional virtual node support."""

    def __init__(self, in_dim, out_dim, eps=True, train_eps=True):
        super().__init__()
        self.mlp = MLP([in_dim, out_dim, out_dim], norm="batch_norm")
        self.conv = GINConv(self.mlp, eps=eps, train_eps=train_eps)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class TransformerEncoderBlock(nn.Module):
    """Standard Transformer encoder block with pre-LN."""

    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x


class ME_Encoder(nn.Module):
    """Atomic graph encoder: GIN + Transformer with virtual node.

    Input: atomic graph Data (x, edge_index, edge_attr)
    Output: node embeddings [N, hidden_dim] and graph embedding [hidden_dim]
    """

    def __init__(self, node_dim, edge_dim, hidden_dim=300, num_gin_layers=4,
                 num_transformer_layers=4, num_heads=8, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Edge MLP to match hidden dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Node projection + virtual node
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.virtual_node = nn.Parameter(torch.randn(1, hidden_dim))

        # GIN layers
        self.gin_layers = nn.ModuleList()
        for _ in range(num_gin_layers):
            self.gin_layers.append(GINBlock(hidden_dim, hidden_dim))
        self.gin_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_gin_layers)])

        # Transformer layers
        self.transformer_layers = nn.ModuleList()
        for _ in range(num_transformer_layers):
            self.transformer_layers.append(
                TransformerEncoderBlock(hidden_dim, num_heads, dropout))

        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (data.x, data.edge_index,
                                            data.edge_attr, data.batch)

        x = self.node_proj(x)
        n_nodes = x.size(0)
        n_graphs = batch.max().item() + 1 if batch.numel() > 0 else 1

        # Expand virtual node per graph
        vn = self.virtual_node.repeat(n_graphs, 1)
        edge_attr_proj = self.edge_mlp(edge_attr)

        # GIN layers with virtual node
        for i, (gin, norm) in enumerate(zip(self.gin_layers, self.gin_norms)):
            # Combine with virtual node
            x_with_vn = torch.cat([x, vn], dim=0)
            vn_idx_start = x.size(0)
            vn_edges_to = torch.stack([
                torch.arange(n_nodes, device=x.device),
                torch.arange(vn_idx_start, vn_idx_start + n_graphs, device=x.device)[batch],
            ])
            vn_edges_from = torch.stack([vn_edges_to[1], vn_edges_to[0]])
            vn_edge_index = torch.cat([edge_index, vn_edges_to, vn_edges_from], dim=1)
            vn_edge_attr = torch.cat([
                edge_attr_proj,
                torch.zeros(vn_edges_to.size(1) * 2, edge_attr_proj.size(1), device=x.device),
            ], dim=0)

            h = gin(x_with_vn, vn_edge_index)
            x_new = h[:n_nodes]
            x = x + self.dropout(norm(x_new))

        # Transformer layers
        # Pool to graph-level, reshape to [n_graphs, hidden_dim]
        x_list = []
        for g in range(n_graphs):
            mask = batch == g
            x_list.append(x[mask])
        # Pad to max nodes per graph
        max_nodes = max(xg.size(0) for xg in x_list)
        padded = torch.zeros(n_graphs, max_nodes, self.hidden_dim, device=x.device)
        mask = torch.zeros(n_graphs, max_nodes, dtype=torch.bool, device=x.device)
        for g, xg in enumerate(x_list):
            padded[g, :xg.size(0)] = xg
            mask[g, :xg.size(0)] = True

        for layer in self.transformer_layers:
            padded = layer(padded)

        # Graph-level readout: mean over non-padded nodes
        graph_emb = (padded * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)

        return padded, graph_emb


class MT_Encoder(nn.Module):
    """Junction tree encoder: GIN on the junction tree graph."""

    def __init__(self, node_dim=10, hidden_dim=300, num_layers=3, dropout=0.3):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.gin_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gin_layers.append(GINBlock(hidden_dim, hidden_dim))
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x = self.node_proj(data.x)
        for gin, norm in zip(self.gin_layers, self.norms):
            x = x + self.dropout(norm(gin(x, data.edge_index)))
        graph_emb = global_mean_pool(x, data.batch)
        return x, graph_emb


class FM_Encoder(nn.Module):
    """Functional group graph encoder: GIN + attention pooling."""

    def __init__(self, node_dim=8, hidden_dim=300, num_layers=3, dropout=0.3):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.gin_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gin_layers.append(GINBlock(hidden_dim, hidden_dim))
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

        # Self-attention for fragment weighting
        self.attn_q = nn.Linear(hidden_dim, hidden_dim)
        self.attn_k = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.node_proj(x)
        for gin, norm in zip(self.gin_layers, self.norms):
            x = x + self.dropout(norm(gin(x, edge_index)))

        # Attention pooling
        q = self.attn_q(x)
        k = self.attn_k(x)
        attn_scores = (q * k).sum(dim=-1) / (self.node_proj.out_features ** 0.5)

        # Softmax per graph
        n_graphs = batch.max().item() + 1 if batch.numel() > 0 else 1
        attn_weights = torch.zeros_like(attn_scores)
        for g in range(n_graphs):
            mask = batch == g
            attn_weights[mask] = F.softmax(attn_scores[mask], dim=0)

        graph_emb = global_add_pool(x * attn_weights.unsqueeze(-1), batch)
        return x, graph_emb


class MG_Encoder(nn.Module):
    """3D topological encoder: CNN on MPH persistence density maps."""

    def __init__(self, grid_size=64, cnn_channels=64, cnn_layers=3,
                 cnn_kernel=3, output_dim=256):
        super().__init__()
        self.grid_size = grid_size

        layers = []
        in_ch = 2  # H0 and H1 density maps
        for i in range(cnn_layers):
            out_ch = cnn_channels * (2 ** i) if i < 2 else cnn_channels * 4
            layers.extend([
                nn.Conv2d(in_ch, out_ch, cnn_kernel, padding=cnn_kernel // 2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ])
            in_ch = out_ch

        self.cnn = nn.Sequential(*layers)

        # Compute output size after pooling
        final_size = grid_size // (2 ** cnn_layers)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_ch * final_size * final_size, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, mph_density_maps):
        # mph_density_maps: [batch, 2, grid_size, grid_size]
        x = self.cnn(mph_density_maps)
        x = self.fc(x)
        return x
