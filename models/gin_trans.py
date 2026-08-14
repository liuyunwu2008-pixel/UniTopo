"""GIN + Transformer molecular graph encoder (ME module).

This is the first molecular representation branch described in the paper:
stacked GIN submodule learns local, short-range features, followed by a
standard Transformer encoder for global, long-distance dependencies.
Uses a virtual node for global message passing.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GINEConv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn.models import MLP
from torch_geometric.utils import to_dense_batch


class GINWithVirtualNode(nn.Module):
    """GIN with virtual node, as described in Eqs. (1)-(2) of the paper."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int = 4,
        mlp_layers: int = 2,
        dropout: float = 0.1,
        edge_dim: int = 11,  # BOND_FDIM from data/mol_graph.py
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # Initial embedding
        self.atom_embed = nn.Linear(in_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_dim, hidden_dim)

        # GINE layers (GIN with edge features, edge_dim=hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP(
                [hidden_dim, hidden_dim, hidden_dim],
                act="relu",
                dropout=dropout,
            )
            self.convs.append(GINEConv(mlp, train_eps=True, edge_dim=hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Virtual node components
        self.vn_mlp = MLP(
            [hidden_dim, hidden_dim, hidden_dim],
            act="relu",
            dropout=dropout,
        )
        self.vn_combine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.vn_agg = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        virtual_node_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with virtual node.

        Args:
            x: Node features (N_total, in_dim)
            edge_index: Graph connectivity (2, E_total)
            edge_attr: Edge features (E_total, 12)
            batch: Batch assignment of nodes (N_total,)
            virtual_node_batch: Batch assignment of virtual nodes

        Returns:
            Node embeddings (N_total, hidden_dim)
        """
        x = self.atom_embed(x)
        if edge_attr is not None:
            edge_attr = self.edge_embed(edge_attr)
        else:
            edge_attr = torch.zeros(edge_index.size(1), x.size(1), device=x.device)

        # Virtual node initialization
        if batch is not None:
            vn_batch = virtual_node_batch if virtual_node_batch is not None else torch.arange(
                batch.max().item() + 1, device=x.device
            )
            vn_x = global_add_pool(x, batch)[vn_batch]
        else:
            vn_x = x.mean(dim=0, keepdim=True)
            vn_batch = torch.zeros(1, dtype=torch.long, device=x.device)

        for l in range(self.num_layers):
            # GINE message passing with edge features
            h = self.convs[l](x, edge_index, edge_attr)
            h = self.bns[l](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            # Virtual node update (Eq. 1)
            vn_msg = self.vn_agg(global_add_pool(h, batch))
            vn_x = self.vn_combine(vn_x + vn_msg)
            vn_x = F.relu(vn_x)

            # Add virtual node feature back to nodes (Eq. 2)
            x = h + vn_x[batch]

        return x


class PositionalEncoding(nn.Module):
    """Absolute (sinusoidal) positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1), :]


class FlashTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with FlashAttention (F.scaled_dot_product_attention).

    Replaces nn.MultiheadAttention with direct QKV projection + sdp attention,
    which automatically selects FlashAttention-2, Memory-Efficient Attention,
    or standard MHA based on hardware and input shape (PyTorch 2.0+).

    Memory: O(N) vs O(N²) for standard attention — enables longer sequences.
    """

    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None, return_per_head=False):
        # Self-attention with FlashAttention
        B, N, D = x.shape
        q = self.q_proj(x).view(B, N, self.nhead, self.head_dim).transpose(1, 2)  # (B, H, N, D_h)
        k = self.k_proj(x).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.nhead, self.head_dim).transpose(1, 2)

        # Build attention mask from padding mask
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            attn_mask = attn_mask.expand(-1, self.nhead, N, -1)     # (B, H, N, N)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout.p if self.training else 0.0,
        )  # (B, H, N, D_h)

        # Per-head features for Multi-Head MoE routing
        per_head_feats = None
        if return_per_head:
            # Mean-pool each head over atoms → (B, H, D_h) independent tokens
            # Each head captures a different attention subspace
            if key_padding_mask is not None:
                valid_mask = (~key_padding_mask).float().unsqueeze(1).unsqueeze(-1)  # (B,1,N,1)
                head_sums = (attn_out * valid_mask).sum(dim=2)  # (B, H, D_h)
                head_counts = valid_mask.sum(dim=2).clamp(min=1)  # (B, H, 1)
                per_head_feats = head_sums / head_counts  # masked mean
            else:
                per_head_feats = attn_out.mean(dim=2)  # (B, H, D_h)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, D)
        attn_out = self.out_proj(attn_out)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        if return_per_head:
            return x, per_head_feats
        return x


class FlashTransformerEncoder(nn.Module):
    """Stack of FlashAttention encoder layers (drop-in replacement for nn.TransformerEncoder)."""

    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            FlashTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, src_key_padding_mask=None, return_per_head=False):
        per_head_feats = None
        for i, layer in enumerate(self.layers):
            is_last = (i == len(self.layers) - 1)
            if return_per_head and is_last:
                x, per_head_feats = layer(x, key_padding_mask=src_key_padding_mask,
                                           return_per_head=True)
            else:
                x = layer(x, key_padding_mask=src_key_padding_mask)
        if return_per_head:
            return x, per_head_feats
        return x


class GINTransformerEncoder(nn.Module):
    """Molecular embedding module (ME): GIN + Transformer (Fig. 2).

    Stacked GIN learns local/short-range features;
    Transformer learns global/long-distance interconnections.
    Set transformer_layers=0 for GIN-only ablation.
    """

    def __init__(
        self,
        atom_dim: int = 148,
        hidden_dim: int = 300,
        gin_layers: int = 4,
        transformer_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_nodes: int = 500,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.transformer_layers = transformer_layers

        self.gin = GINWithVirtualNode(
            in_dim=atom_dim,
            hidden_dim=hidden_dim,
            num_layers=gin_layers,
            dropout=dropout,
        )

        self.pos_enc = PositionalEncoding(hidden_dim, max_len=max_nodes)

        if transformer_layers > 0:
            self.transformer = FlashTransformerEncoder(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                num_layers=transformer_layers,
            )
        else:
            self.transformer = None

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(
        self,
        data,
        return_node_embeds: bool = False,
        return_per_head: bool = False,
    ) -> torch.Tensor:
        """Encode a batch of atomic graphs.

        Args:
            data: PyG Data or list of Data objects
            return_node_embeds: if True, also return per-node embeddings
            return_per_head: if True, also return per-head features (B, H, D_h)
                            from the last Transformer layer for Multi-Head MoE

        Returns:
            Graph-level embeddings (batch_size, hidden_dim)
            + optionally per-head features (B, num_heads, head_dim)
        """
        from torch_geometric.data import Batch

        if isinstance(data, list):
            data = Batch.from_data_list(data)

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # GIN with virtual node
        h = self.gin(x, edge_index, edge_attr, batch)

        per_head_feats = None
        if self.transformer is not None:
            # Pad to dense + positional encoding + Transformer
            h_dense, mask = to_dense_batch(h, batch, fill_value=0.0)
            h_dense = self.pos_enc(h_dense)
            if return_per_head:
                h_trans, per_head_feats = self.transformer(
                    h_dense, src_key_padding_mask=~mask, return_per_head=True)
            else:
                h_trans = self.transformer(h_dense, src_key_padding_mask=~mask)
            h_graph = h_trans.mean(dim=1)
        else:
            # GIN-only: pool directly
            from torch_geometric.nn import global_mean_pool
            h_graph = global_mean_pool(h, batch)

        h_graph = self.readout(h_graph)

        if return_per_head and per_head_feats is not None:
            if return_node_embeds:
                return h_graph, h_trans, per_head_feats
            return h_graph, per_head_feats
        if return_node_embeds:
            if self.transformer is not None:
                return h_graph, h_trans
            return h_graph, h
        return h_graph
