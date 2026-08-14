"""Cross-View Fusion Module (EFF) for multi-view integration."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossViewFusionModule(nn.Module):
    """CNN-attention based fusion of two molecular views (EFF stage).

    Fuses two representations z_a and z_b using 2D CNN attention
    to learn view-specific weighting.
    """

    def __init__(self, hidden_dim=300):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 2D CNN for cross-view interaction
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        # Linear projections with residual
        self.fc_a = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.fc_b = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

    def forward(self, z_a, z_b):
        """
        Args:
            z_a: [batch, hidden_dim] first view embeddings
            z_b: [batch, hidden_dim] second view embeddings
        Returns:
            fused: [batch, hidden_dim] attention-weighted fused embeddings
        """
        batch_size = z_a.size(0)

        # Reshape to 2D for CNN: [batch, 2, H, W]
        # Reshape hidden_dim into a square-ish 2D grid
        side = int(self.hidden_dim ** 0.5)
        if side * side < self.hidden_dim:
            side += 1
        padded_dim = side * side

        def reshape_2d(z):
            z_pad = F.pad(z, (0, padded_dim - self.hidden_dim))
            return z_pad.view(batch_size, 1, side, side)

        z_a_2d = reshape_2d(z_a)
        z_b_2d = reshape_2d(z_b)

        # Stack as 2-channel input
        stacked = torch.cat([z_a_2d, z_b_2d], dim=1)  # [batch, 2, side, side]

        # CNN attention
        w_att = self.cnn(stacked)  # [batch, 1, side, side]
        w_att = w_att.view(batch_size, 1, padded_dim)[:, :, :self.hidden_dim]

        # Attention-weighted fusion with residual
        fused = (self.fc_a(z_a) * w_att.squeeze(1) + z_a +
                 self.fc_b(z_b) * (1 - w_att.squeeze(1)) + z_b)

        return fused


class HierarchicalFusion(nn.Module):
    """Two-stage hierarchical fusion: 2D views first, then 3D.

    Stage 1: Fuse ME + MT -> h_gt
    Stage 2: Fuse h_gt + FM -> h_gtf
    Stage 3 (optional): Fuse h_gtf + MG -> h_final
    """

    def __init__(self, hidden_dim=300, use_3d=True):
        super().__init__()
        self.eff_1 = CrossViewFusionModule(hidden_dim)  # ME + MT
        self.eff_2 = CrossViewFusionModule(hidden_dim)  # (ME+MT) + FM
        self.eff_3 = CrossViewFusionModule(hidden_dim) if use_3d else None
        self.use_3d = use_3d

    def forward(self, z_me, z_mt, z_fm, z_mg=None):
        # Stage 1: fuse atomic + junction tree
        h_gt = self.eff_1(z_me, z_mt)

        # Stage 2: fuse with functional groups
        h_gtf = self.eff_2(h_gt, z_fm)

        # Stage 3 (optional): fuse with 3D topology
        if self.use_3d and z_mg is not None:
            h_final = self.eff_3(h_gtf, z_mg)
            return h_final

        return h_gtf
