"""MMG-MPP: Full multi-view molecular pre-training and fine-tuning model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import ME_Encoder, MT_Encoder, FM_Encoder, MG_Encoder
from .gmae import GraphMaskedAutoEncoder
from .fusion import HierarchicalFusion


class MMGMPP(nn.Module):
    """Full MMG-MPP model for pre-training.

    Four branches: ME (atomic), MT (junction tree), FM (fragment GMAE), MG (3D MPH).
    """

    def __init__(self, config, node_dim_me=78, edge_dim_me=12, node_dim_mt=10,
                 node_dim_fm=8, use_3d=True):
        super().__init__()
        hidden_dim = config.gin_hidden_dim
        self.use_3d = use_3d

        # Encoders
        self.me_encoder = ME_Encoder(
            node_dim_me, edge_dim_me, hidden_dim,
            config.gin_num_layers, config.transformer_num_layers,
            config.transformer_num_heads, config.gin_dropout,
        )
        self.mt_encoder = MT_Encoder(
            node_dim_mt, hidden_dim, config.gin_num_layers, config.gin_dropout,
        )
        self.fm_gmae = GraphMaskedAutoEncoder(
            node_dim_fm, hidden_dim, num_enc_layers=config.gin_num_layers,
            mask_ratio=config.gmae_mask_ratio,
        )
        self.fm_encoder = FM_Encoder(
            node_dim_fm, hidden_dim, config.gin_num_layers, config.gin_dropout,
        )

        if use_3d:
            self.mg_encoder = MG_Encoder(
                config.mph_grid_size, config.mph_cnn_channels,
                config.mph_cnn_layers, config.mph_cnn_kernel, hidden_dim,
            )

        # Projection heads for contrastive learning
        self.proj_gt = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
        )
        self.proj_frag = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
        )

    def forward(self, data_me, data_mt, data_fm, mph_maps=None):
        """
        Args:
            data_me: atomic graph batch
            data_mt: junction tree batch
            data_fm: fragment graph batch
            mph_maps: 3D MPH density maps [batch, 2, G, G] or None
        Returns:
            dict with contrastive embeddings, GMAE loss, and reconstructions
        """
        # ME: atomic graph
        _, z_me = self.me_encoder(data_me)

        # MT: junction tree
        _, z_mt = self.mt_encoder(data_mt)

        # FM: fragment graph with GMAE
        _, fm_recon_loss, fm_mask = self.fm_gmae(data_fm)
        _, z_fm = self.fm_encoder(data_fm)

        # MG: 3D topology (optional)
        z_mg = None
        if self.use_3d and mph_maps is not None:
            z_mg = self.mg_encoder(mph_maps)

        # Fused representation for positive sample
        z_gt = self.proj_gt(z_me + z_mt)
        if z_mg is not None:
            z_gt = self.proj_gt(z_me + z_mt + z_mg)

        # Fragment representation as anchor
        z_frag = self.proj_frag(z_fm)

        return {
            "z_gt": z_gt,           # positive sample (fused ME+MT [+MG])
            "z_frag": z_frag,       # anchor (fragment-based)
            "gmae_loss": fm_recon_loss,
            "z_me": z_me,
            "z_mt": z_mt,
            "z_fm": z_fm,
            "z_mg": z_mg,
        }

    def encode_all(self, data_me, data_mt, data_fm, mph_maps=None):
        """Encode all views without contrastive heads."""
        _, z_me = self.me_encoder(data_me)
        _, z_mt = self.mt_encoder(data_mt)
        _, z_fm_frag = self.fm_gmae.encode(data_fm)
        _, z_fm = self.fm_encoder(data_fm)

        z_mg = None
        if self.use_3d and mph_maps is not None:
            z_mg = self.mg_encoder(mph_maps)

        return z_me, z_mt, z_fm, z_mg


class MMGMPPFinetune(nn.Module):
    """MMG-MPP with fusion and prediction head for downstream fine-tuning."""

    def __init__(self, pretrained_model, config, num_tasks=1):
        super().__init__()
        self.encoder = pretrained_model
        hidden_dim = config.gin_hidden_dim

        # Hierarchical fusion
        self.fusion = HierarchicalFusion(hidden_dim, use_3d=config.use_3d_branch)

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.gin_dropout),
            nn.Linear(hidden_dim // 2, num_tasks),
        )

    def forward(self, data_me, data_mt, data_fm, mph_maps=None):
        z_me, z_mt, z_fm, z_mg = self.encoder.encode_all(
            data_me, data_mt, data_fm, mph_maps)

        # Hierarchical fusion
        h_final = self.fusion(z_me, z_mt, z_fm, z_mg)

        # Prediction
        out = self.predictor(h_final)
        return out.squeeze(-1)
