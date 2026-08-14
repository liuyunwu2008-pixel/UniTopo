"""Pre-training losses: InfoNCE contrastive + GMAE reconstruction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss.

    Fragment-based embeddings are anchors;
    fused ME+MT (+MG) embeddings are positive samples;
    other molecules in the batch are negative samples.
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_frag, z_gt):
        """
        Args:
            z_frag: fragment-based embeddings [batch, dim] (anchors)
            z_gt: fused multi-view embeddings [batch, dim] (positives)
        Returns:
            loss: scalar InfoNCE loss
        """
        batch_size = z_frag.size(0)
        device = z_frag.device

        # Normalize embeddings
        z_frag = F.normalize(z_frag, dim=-1)
        z_gt = F.normalize(z_gt, dim=-1)

        # Compute similarity matrix
        # Positive pairs are at diagonal
        similarity = torch.matmul(z_frag, z_gt.T) / self.temperature  # [B, B]

        # Labels: each row i's positive is at column i
        labels = torch.arange(batch_size, device=device)

        # InfoNCE: cross-entropy over rows
        loss = F.cross_entropy(similarity, labels)

        return loss


class TotalPretrainLoss(nn.Module):
    """Combined loss: L = L_contrastive + alpha * L_generative."""

    def __init__(self, temperature=0.1, gmae_weight=0.25):
        super().__init__()
        self.contrastive_loss = InfoNCELoss(temperature)
        self.gmae_weight = gmae_weight

    def forward(self, z_frag, z_gt, gmae_loss):
        """
        Args:
            z_frag: fragment-based embeddings
            z_gt: fused multi-view embeddings
            gmae_loss: scalar GMAE reconstruction loss
        Returns:
            total_loss, contrastive_loss, weighted_gmae_loss
        """
        l_contrastive = self.contrastive_loss(z_frag, z_gt)
        total = l_contrastive + self.gmae_weight * gmae_loss
        return total, l_contrastive, self.gmae_weight * gmae_loss
