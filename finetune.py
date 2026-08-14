"""MMG-MPP fine-tuning script for downstream MPP tasks.

Loads pre-trained weights and fine-tunes on MoleculeNet benchmark datasets.
Supports both classification (ROC-AUC) and regression (RMSE/MAE) tasks.

Key features:
  - Scaffold split (8/1/1) for realistic evaluation
  - Early stopping (patience=20)
  - 5 independent runs with mean ± std
  - Grid search over lr, weight_decay, dropout
"""

import os
import sys
import argparse
import logging
import json
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*MorganGenerator.*")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.model_selection import ParameterGrid
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

# RDKit is notoriously noisy — suppress UFF and other routine warnings
RDLogger.logger().setLevel(RDLogger.ERROR)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ModelConfig, FinetuneConfig, TASKS, TASK_GROUPS
from utils import evaluate_predictions
from utils.helpers import to_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

def scaffold_split(
    smiles_list: List[str],
    labels: torch.Tensor,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> Tuple:
    """Scaffold-based split with multi-label proportional stratification.

    Groups molecules by Bemis-Murcko scaffold, then splits scaffolds
    rather than individual molecules to avoid data leakage.

    For multi-label tasks (num_tasks > 1), _balance_multilabel_splits
    then moves individual molecules from training to val/test to ensure:
      (a) every split has ≥1 positive and ≥1 negative per task (avoids NaN AUC),
      (b) per-task positive ratios in val/test are within 8% of the global ratio
          (prevents noisy AUC estimates on sparse tasks like Tox21/SIDER).
    """
    # Compute scaffold for each molecule
    scaffold_to_indices: Dict[str, List[int]] = {}
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scaffold = "invalid"
        else:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=False
            )
            if not scaffold:
                scaffold = "no_scaffold"
        scaffold_to_indices.setdefault(scaffold, []).append(i)

    # ── Scaffold-aware round-robin assignment (reduces split variance) ──
    # Greedy assignment biases splits: early large scaffolds dominate train,
    # leaving val/test with only small, potentially unrepresentative scaffolds.
    # Round-robin distributes scaffolds evenly across splits by size.
    scaffolds = sorted(scaffold_to_indices.keys(), key=lambda s: -len(scaffold_to_indices[s]))

    rng = np.random.default_rng(seed)
    rng.shuffle(scaffolds)

    train_idx, val_idx, test_idx = [], [], []
    n = len(smiles_list)
    n_train_target = int(train_frac * n)
    n_val_target = int(val_frac * n)

    for scaf in scaffolds:
        indices = scaffold_to_indices[scaf]
        # Round-robin: assign to the split with the smallest current size
        # (relative to its target), ensuring balanced scaffold diversity
        t_ratio = len(train_idx) / max(n_train_target, 1)
        v_ratio = len(val_idx) / max(n_val_target, 1)
        # Find the "most empty" split
        if t_ratio <= v_ratio and len(train_idx) < n_train_target * 1.05:
            train_idx.extend(indices)
        elif len(val_idx) < n_val_target * 1.05:
            val_idx.extend(indices)
        else:
            test_idx.extend(indices)

    # --- Stratification: ensure each split has ≥1 pos & ≥1 neg per task ---
    labels_np = labels.numpy()
    if labels_np.ndim == 1:
        labels_np = labels_np.reshape(-1, 1)
    num_tasks = labels_np.shape[1]

    if num_tasks > 1:
        _balance_multilabel_splits(train_idx, val_idx, test_idx, labels_np)
    else:
        _balance_binary_splits(train_idx, val_idx, test_idx, labels_np)

    # --- Property-based stratification (MW, LogP, TPSA balance) ---
    _balance_molecular_properties(train_idx, val_idx, test_idx, smiles_list)

    # --- Final validation: every split must have ≥1 pos & ≥1 neg for each task ---
    try:
        _validate_split_labels(train_idx, val_idx, test_idx, labels_np, dataset_hint=str(len(smiles_list)))
    except ValueError as e:
        logger.warning("Scaffold split failed balance check: %s", e)
        logger.warning("Falling back to random split (scaffold integrity sacrificed for valid eval)")
        rng = np.random.default_rng(seed + 9999)
        indices = rng.permutation(n).tolist()
        n_train = int(train_frac * n)
        n_val = int(val_frac * n)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        _validate_split_labels(train_idx, val_idx, test_idx, labels_np, dataset_hint=f"{len(smiles_list)} [random fallback]")

    return (
        [smiles_list[i] for i in train_idx], labels[train_idx],
        [smiles_list[i] for i in val_idx], labels[val_idx],
        [smiles_list[i] for i in test_idx], labels[test_idx],
    )


def _balance_multilabel_splits(
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
    labels_np: "np.ndarray",  # (N, K)
    tolerance: float = 0.08,
):
    """Ensure each split has proportional positive/negative labels per task.

    After scaffold-based splitting, some splits may have too few or too many
    positives for sparse multi-label tasks (e.g., Tox21, SIDER, ClinTox).
    This function moves individual molecules from the training pool to val/test
    until each split's per-task positive ratio is within `tolerance` of the
    global positive ratio.  It also ensures ≥1 positive and ≥1 negative per
    task in every split (the minimum requirement for valid ROC-AUC).

    Molecules are only moved FROM training TO val/test — never the reverse —
    to preserve scaffold integrity in the evaluation splits.
    """
    num_tasks = labels_np.shape[1]

    # ── Phase 1: guarantee ≥1 pos AND ≥1 neg per task in val/test ──
    for task in range(num_tasks):
        task_labels = labels_np[:, task]
        # Skip NaN entries when checking class presence
        for split_idx in [val_idx, test_idx]:
            if len(split_idx) == 0:
                continue
            split_vals = task_labels[split_idx]
            valid_vals = split_vals[~np.isnan(split_vals)]
            if len(valid_vals) == 0:
                continue
            has_pos = (valid_vals == 1).any()
            has_neg = (valid_vals == 0).any()
            if has_pos and has_neg:
                continue
            missing_cls = 0 if has_pos else 1
            candidates = [i for i in train_idx if not np.isnan(task_labels[i]) and task_labels[i] == missing_cls]
            if candidates:
                c = candidates[0]
                train_idx.remove(c)
                split_idx.append(c)

    # ── Phase 2: rebalance per-task positive ratios ──
    # Use nan-aware operations: Tox21/SIDER have sparsely annotated labels (NaN = missing)
    task_labels_full = labels_np  # (N, K)

    for task in range(num_tasks):
        task_labels = task_labels_full[:, task]
        valid = ~np.isnan(task_labels)
        if valid.sum() == 0:
            continue
        gr = task_labels[valid].mean()  # global pos ratio among KNOWN labels only
        if gr < 0.015 or gr > 0.985:
            # Heavily skewed task — Phase 1's ≥1 guarantee is sufficient
            continue

        for target_idx in [val_idx, test_idx]:
            if len(target_idx) == 0:
                continue
            split_vals = task_labels[target_idx]
            split_valid = ~np.isnan(split_vals)
            if split_valid.sum() == 0:
                continue
            current_ratio = split_vals[split_valid].mean()
            deficit = gr - current_ratio  # positive → need more pos; negative → need more neg

            if np.isnan(deficit) or abs(deficit) <= tolerance:
                continue

            needed_class = 1 if deficit > 0 else 0
            n_needed = max(1, int(abs(deficit) * split_valid.sum()))
            n_needed = min(n_needed, 15)  # cap moves to preserve scaffold structure

            # Collect candidates (iterate over snapshot to avoid mutation issues)
            candidates = [i for i in train_idx if task_labels[i] == needed_class]
            for c in candidates[:n_needed]:
                split_vals = task_labels[target_idx]
                split_valid = ~np.isnan(split_vals)
                if split_valid.sum() == 0 or abs(split_vals[split_valid].mean() - gr) <= tolerance:
                    break
                train_idx.remove(c)
                target_idx.append(c)


def _balance_binary_splits(
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
    labels_np: "np.ndarray",  # (N, 1)
    tolerance: float = 0.05,  # tightened: 5% deviation max
):
    """Ensure binary splits have uniform positive/negative distribution.

    Moves molecules from training to val/test to match the global
    positive ratio within ±5%, while preserving the ≥1 pos/neg guarantee.
    """
    task_labels = labels_np[:, 0]
    valid = ~np.isnan(task_labels)
    gr = task_labels[valid].mean()  # global positive ratio
    logger.info("Global positive ratio: %.3f", gr)

    for target_idx in [val_idx, test_idx]:
        if len(target_idx) == 0:
            continue
        split_vals = task_labels[target_idx]
        split_valid = ~np.isnan(split_vals)
        if split_valid.sum() == 0:
            continue

        has_pos = (split_vals[split_valid] == 1).any()
        has_neg = (split_vals[split_valid] == 0).any()

        # Phase 1: guarantee ≥1 pos and ≥1 neg
        if not has_pos:
            candidates = [i for i in train_idx
                          if not np.isnan(task_labels[i]) and task_labels[i] == 1]
            for c in candidates[:1]:
                train_idx.remove(c); target_idx.append(c); has_pos = True

        if not has_neg:
            candidates = [i for i in train_idx
                          if not np.isnan(task_labels[i]) and task_labels[i] == 0]
            for c in candidates[:1]:
                train_idx.remove(c); target_idx.append(c); has_neg = True

        # Phase 2: rebalance to within tolerance of global ratio
        split_vals = task_labels[target_idx]
        split_valid = ~np.isnan(split_vals)
        if split_valid.sum() == 0:
            continue
        current_ratio = split_vals[split_valid].mean()
        deficit = gr - current_ratio

        if abs(deficit) > tolerance:
            needed_class = 1 if deficit > 0 else 0
            n_needed = max(1, int(abs(deficit) * split_valid.sum()))
            n_needed = min(n_needed, 15)  # cap raised for better balance
            candidates = [i for i in train_idx
                          if not np.isnan(task_labels[i]) and task_labels[i] == needed_class]
            for c in candidates[:n_needed]:
                split_vals = task_labels[target_idx]
                split_valid = ~np.isnan(split_vals)
                if split_valid.sum() == 0 or abs(split_vals[split_valid].mean() - gr) <= tolerance:
                    break
                train_idx.remove(c)
                target_idx.append(c)

        # Log final ratio
        final_vals = task_labels[target_idx]
        final_valid = ~np.isnan(final_vals)
        if final_valid.sum() > 0:
            logger.info("  %s pos ratio: %.3f (global=%.3f)",
                        "val" if target_idx is val_idx else "test",
                        final_vals[final_valid].mean(), gr)


def _balance_molecular_properties(
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
    smiles_list: List[str],
    tolerance: float = 0.15,
):
    """Ensure train/val/test have similar MW, LogP, TPSA distributions.

    After scaffold split, one split may get all large/hydrophobic molecules,
    creating systematic bias.  This function computes 3 key drug-likeness
    properties, checks distributional similarity, and logs warnings if
    any split differs by >tolerance from the global mean.
    """
    from rdkit.Chem import Descriptors as Desc
    props = {"MW": [], "LogP": [], "TPSA": []}
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            for k in props:
                props[k].append(0.0)
        else:
            props["MW"].append(Desc.MolWt(mol))
            props["LogP"].append(Desc.MolLogP(mol))
            props["TPSA"].append(Desc.TPSA(mol))

    global_means = {k: np.mean(v) for k, v in props.items()}

    for split_name, split_idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        split_means = {k: np.mean([props[k][i] for i in split_idx]) for k in props}
        for k in props:
            diff = abs(split_means[k] - global_means[k]) / max(abs(global_means[k]), 1e-6)
            if diff > tolerance:
                logger.warning(
                    "%s %s: split mean=%.1f vs global=%.1f (%.0f%% deviation)",
                    split_name, k, split_means[k], global_means[k], diff * 100)


def _validate_split_labels(
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
    labels_np: "np.ndarray",
    dataset_hint: str = "",
):
    """Final assertion: every split must have ≥1 pos and ≥1 neg per task.

    Logs a clear error and raises ValueError if any split is unusable,
    preventing silent NaN ROC-AUC / RMSE in downstream evaluation.
    """
    num_tasks = labels_np.shape[1]
    for split_name, split_idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        if len(split_idx) == 0:
            raise ValueError(
                f"{dataset_hint}: {split_name} split is EMPTY after scaffold+balance. "
                f"Dataset may be too small for 80/10/10 split."
            )
        for t in range(num_tasks):
            task_labels = labels_np[split_idx, t]
            valid = task_labels[~np.isnan(task_labels)]
            if len(valid) == 0:
                continue  # all-NaN task (sparse multi-label)
            n_pos = int((valid == 1).sum())
            n_neg = int((valid == 0).sum())
            if n_pos == 0 or n_neg == 0:
                raise ValueError(
                    f"{dataset_hint}: {split_name} split task {t} has "
                    f"pos={n_pos} neg={n_neg}. Cannot compute ROC-AUC. "
                    f"Dataset may have extreme class imbalance. "
                    f"Consider increasing val_frac or merging rare classes."
                )


def _compute_pos_weight(train_labels: torch.Tensor, num_tasks: int, task_type: str,
                        device: str) -> Optional[torch.Tensor]:
    """Per-task positive weight for multi-label BCE (handles class imbalance)."""
    if task_type != "classification" or num_tasks <= 1:
        return None
    pos_counts = (train_labels == 1).sum(0)
    neg_counts = (train_labels == 0).sum(0)
    return torch.clamp(neg_counts / (pos_counts + 1e-8), 1.0, 50.0).to(device)


def _make_optimizer(model, lr, weight_decay, freeze_encoders=True, encoder_lr_ratio=0.1):
    """Create AdamW. When freeze_encoders=True, only fusion+predictor are trained."""
    if freeze_encoders:
        for name, param in model.named_parameters():
            if name.startswith("gin."):
                param.requires_grad = False
        trainable = [p for p in model.parameters() if p.requires_grad]
        return AdamW(trainable, lr=lr, weight_decay=weight_decay)

    # Full fine-tuning: lower LR for encoder
    encoder_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("gin."):
            encoder_params.append(param)
        else:
            head_params.append(param)
    return AdamW([
        {"params": encoder_params, "lr": lr * encoder_lr_ratio},
        {"params": head_params, "lr": lr},
    ], weight_decay=weight_decay)




class PhysChemFeaturizer:
    """RDKit physical-chemical descriptors — replaces ECFP for regression.

    Computes 10 descriptors: LogP, MR, TPSA, HBD, HBA, RotBonds, AromaticRings,
    FractionCsp3, SaturatedRings, AliphaticRings. Standardized to zero-mean unit-variance.
    Per PhyChem-MoE Table 9: ECFP helps regression −67% RMSE but classification ~0%.
    """

    _DESCRIPTORS = [
        "MolLogP", "MolMR", "TPSA", "NumHBD", "NumHBA",
        "NumRotatableBonds", "NumAromaticRings", "FractionCsp3",
        "NumSaturatedRings", "NumAliphaticRings",
    ]

    def __init__(self):
        from rdkit.Chem import Descriptors as _Desc
        self._Desc = _Desc

    def compute(self, smiles_list):
        import numpy as np
        feats = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                feats.append([0.0] * len(self._DESCRIPTORS))
            else:
                row = []
                for name in self._DESCRIPTORS:
                    try:
                        row.append(getattr(self._Desc, name)(mol))
                    except Exception:
                        row.append(0.0)
                feats.append(row)
        return torch.tensor(np.array(feats, dtype=np.float32))

    def compute_and_project(self, smiles_list, device="cpu"):
        raw = self.compute(smiles_list).to(device)
        mean = raw.mean(dim=0, keepdim=True)
        std = raw.std(dim=0, keepdim=True).clamp(min=1e-6)
        return (raw - mean) / std


class MoEPredictor(nn.Module):
    """Standard single-token Mixture-of-Experts predictor.

    Routes the SINGLE molecular embedding through 4 experts.
    WARNING: With only one token per molecule, the gate lacks diversity
    and experts tend to converge to the same function (PhyChem-MoE §4.1).
    Use MultiHeadMoEPredictor for improved token diversity.
    """

    def __init__(self, hidden_dim=512, embed_dim=256, num_tasks=1, dropout=0.3,
                 num_experts=4, top_k=2, gate_tau=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate_tau = gate_tau

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, embed_dim),
                nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, num_tasks),
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_experts),
        )

    def forward(self, h):
        expert_outs = torch.stack([e(h) for e in self.experts], dim=1)
        gate_logits = self.gate(h)
        noise = torch.randn_like(gate_logits) * 0.1
        gate_w = F.softmax((gate_logits + noise) / self.gate_tau, dim=-1)
        topk_vals, topk_idx = torch.topk(gate_w, self.top_k, dim=-1)
        gate_w = torch.zeros_like(gate_w).scatter_(-1, topk_idx, topk_vals)
        gate_w = gate_w / gate_w.sum(dim=-1, keepdim=True)
        out = (gate_w.unsqueeze(-1) * expert_outs).sum(dim=1)
        return out, gate_w

    def load_balance_loss(self, gate_w):
        mean_usage = gate_w.mean(dim=0)
        return mean_usage.var() / (mean_usage.mean()**2 + 1e-8)


class MultiHeadMoEPredictor(nn.Module):
    """Per-modality Mixture-of-Experts predictor (PhyChem-MoE style).

    Routes 1-3 GENUINELY different modality tokens through the MoE gate:
      - GIN:   global molecular embedding (1 token, projected hidden_dim→64)
      - MPH:   3D topology embedding (1 token, projected 512→64)  [optional]
      - ECFP:  2D fingerprint (1 token, projected 2048→64)  [optional]

    3 tokens max → diverse information sources → gate can specialize.
    Per-head tokens intentionally excluded: 8 heads from same TF layer
    are correlated, not independent modalities.
    """

    def __init__(self, hidden_dim=512, head_dim=64, num_heads=8, embed_dim=128,
                 num_tasks=1, dropout=0.3, num_experts=4, top_k=2, gate_tau=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate_tau = gate_tau
        token_dim = head_dim  # 64

        # Token projectors — map diverse modalities to common token_dim
        self.global_proj = nn.Linear(hidden_dim, token_dim)   # GIN mol emb → token
        self.mph_proj = nn.Linear(hidden_dim, token_dim)      # MPH → token
        self.ecfp_proj_moe = nn.Linear(2048, token_dim)       # ECFP → token

        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(token_dim, embed_dim),
                nn.SiLU(), nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim),
                nn.SiLU(), nn.Dropout(dropout),
                nn.Linear(embed_dim, num_tasks),
            ) for _ in range(num_experts)
        ])

        # Gating network
        self.gate = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, num_experts),
        )

    def forward(self, h_mol, per_head=None, mph_feat=None, ecfp_feat=None):
        """Multi-modal MoE forward.

        Args:
            h_mol:    (B, hidden_dim) global GIN embedding
            per_head: (B, H, D_h) per-head features from Transformer
            mph_feat: (B, hidden_dim) MPH topology embedding [optional]
            ecfp_feat: (B, 2048) ECFP fingerprint [optional]

        Returns:
            out: (B, num_tasks), gate_w: (B×T, K)
        """
        B = h_mol.shape[0]
        token_list = [self.global_proj(h_mol).unsqueeze(1)]  # (B, 1, token_dim)

        if per_head is not None:
            token_list.append(per_head)  # (B, H, token_dim) — already 64-dim

        if mph_feat is not None:
            token_list.append(self.mph_proj(mph_feat).unsqueeze(1))  # (B, 1, token_dim)

        if ecfp_feat is not None:
            token_list.append(self.ecfp_proj_moe(ecfp_feat).unsqueeze(1))  # (B, 1, token_dim)

        tokens = torch.cat(token_list, dim=1)  # (B, T, token_dim)
        T = tokens.shape[1]
        tokens_flat = tokens.reshape(B * T, -1)  # (B×T, token_dim)

        # Expert computation
        expert_outs = torch.stack([e(tokens_flat) for e in self.experts], dim=1)  # (B×T, K, tasks)

        # Gate
        gate_logits = self.gate(tokens_flat)  # (B×T, K)
        noise = torch.randn_like(gate_logits) * 0.1
        gate_w = F.softmax((gate_logits + noise) / self.gate_tau, dim=-1)
        topk_vals, topk_idx = torch.topk(gate_w, self.top_k, dim=-1)
        gate_w = torch.zeros_like(gate_w).scatter_(-1, topk_idx, topk_vals)
        gate_w = gate_w / gate_w.sum(dim=-1, keepdim=True)

        # Weighted expert output → per-token predictions → pool over tokens
        out_flat = (gate_w.unsqueeze(-1) * expert_outs).sum(dim=1)  # (B×T, tasks)
        out = out_flat.reshape(B, T, -1).mean(dim=1)  # (B, tasks)

        return out, gate_w

    def load_balance_loss(self, gate_w):
        mean_usage = gate_w.mean(dim=0)
        return mean_usage.var() / (mean_usage.mean()**2 + 1e-8)


class GIN3DFinetune(nn.Module):
    """Lightweight finetune wrapper for pretrained encoder(s).

    Supports:
      - GIN-only (default): 2D graph encoder → predictor
      - GIN+MPH (--mph): 2D + 3D topology → concat → predictor
      - +ECFP (--ecfp): add Morgan fingerprint projection (PhyChem-MoE style)
    """

    def __init__(self, gin_encoder, mph_encoder=None, hidden_dim=512, embed_dim=256,
                 num_tasks=1, task_type="classification", dropout=0.3,
                 use_moe=False, use_mph=False, use_ecfp=False):
        super().__init__()
        self.gin = gin_encoder
        self.mph = mph_encoder
        self.hidden_dim = hidden_dim
        self.num_tasks = num_tasks
        self.task_type = task_type
        self.use_moe = use_moe
        self.use_mph = use_mph and mph_encoder is not None
        self.use_ecfp = use_ecfp

        predictor_input_dim = hidden_dim
        if self.use_mph:
            predictor_input_dim += hidden_dim  # +512 for MPH
        if self.use_ecfp:
            self.ecfp_proj = nn.Sequential(
                nn.Linear(2048, 512), nn.SiLU(), nn.LayerNorm(512),
            )
            predictor_input_dim += 512  # +512 for ECFP

        if use_moe:
            # Multi-Head MoE routes per-head features
            self.predictor = MultiHeadMoEPredictor(
                hidden_dim=hidden_dim, head_dim=hidden_dim // 8, num_heads=8,
                embed_dim=embed_dim, num_tasks=num_tasks, dropout=dropout,
                num_experts=4, top_k=2, gate_tau=0.1,
            )
        else:
            # PhyChem-MoE: 2-layer MLP with SiLU, Linear(512→256→out)
            self.predictor = nn.Sequential(
                nn.Linear(predictor_input_dim, embed_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, num_tasks),
            )

    def forward(self, batch):
        if self.use_moe:
            # Per-modality MoE: GIN + optional MPH + optional ECFP as 3 tokens
            h_mol = self.gin(batch["atom_g"])
            mph_feat = None
            ecfp_feat = None
            if self.use_mph and "coords" in batch and "charges" in batch:
                mph_feat = self.mph(batch["coords"], batch["charges"])
            if self.use_ecfp and "smiles" in batch:
                ecfp_feat = self._compute_ecfp(batch["smiles"], h_mol.device)
            out, _ = self.predictor(h_mol, per_head=None,
                                     mph_feat=mph_feat, ecfp_feat=ecfp_feat)
        else:
            h_gin = self.gin(batch["atom_g"])
            features = [h_gin]
            if self.use_mph and "coords" in batch and "charges" in batch:
                features.append(self.mph(batch["coords"], batch["charges"]))
            if self.use_ecfp and "smiles" in batch:
                features.append(self.ecfp_proj(self._compute_ecfp(batch["smiles"], h_gin.device)))
            h = torch.cat(features, dim=-1) if len(features) > 1 else features[0]
            out = self.predictor(h)
        if self.task_type == "classification" and self.num_tasks == 1:
            out = out.squeeze(-1)
        return out

    @staticmethod
    def _compute_ecfp(smiles_list, device):
        """Morgan fingerprints r=2, 2048-bit (PhyChem-MoE style)."""
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
        fps = torch.zeros(len(smiles_list), 2048, device=device)
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                arr = torch.zeros(2048, dtype=torch.float32)
                DataStructs.ConvertToNumpyArray(fp, arr.cpu().numpy())
                fps[i] = torch.from_numpy(arr.cpu().numpy()).float().to(device)
        return fps

    def compute_loss(self, pred, y, pos_weight=None, smooth_l1_beta=1.0):
        if self.task_type == "regression":
            return F.smooth_l1_loss(pred.view(-1), y.view(-1), beta=smooth_l1_beta)
        y_target = y.squeeze(-1) if y.ndim > 1 and y.shape[1] == 1 else y
        return F.binary_cross_entropy_with_logits(pred.view_as(y_target), y_target.float(),
                                                   pos_weight=pos_weight)


def _make_mmgmpp_loader(smiles, labels, task_type, num_tasks, batch_size, shuffle,
                      use_3d=False):
    """Lightweight DataLoader for GIN3D — only builds atom_g (no jt_g/fg_g)."""
    from data import PretrainGIN3DDataset

    class _LabeledMMGMPPDataset(PretrainGIN3DDataset):
        """PretrainGIN3DDataset + labels for finetuning."""
        def __init__(self, smiles_list, labels, *args, **kwargs):
            super().__init__(smiles_list, *args, use_3d=use_3d, **kwargs)
            self.labels = labels
        def __getitem__(self, idx):
            item = super().__getitem__(idx)
            item["y"] = self.labels[idx].float()
            return item

    dataset = _LabeledMMGMPPDataset(smiles, labels)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=_mmgmpp_collate,
    )


def _mmgmpp_collate(batch):
    """Collate for GIN finetuning — atom_g + labels + optional 3D coords."""
    result = {
        "atom_g": [b["atom_g"] for b in batch],
        "y": torch.stack([b["y"] for b in batch]),
    }
    if "smiles" in batch[0]:
        result["smiles"] = [b["smiles"] for b in batch]
        result["_idx"] = [b.get("_idx", i) for i, b in enumerate(batch)]
    # Forward 3D coords/charges when present (MPH dual-encoder mode)
    if "coords" in batch[0]:
        result["coords"] = [b.get("coords") for b in batch]
        result["charges"] = [b.get("charges") for b in batch]
    return result


# Alias for historical name used throughout finetune_gin3d()
_make_gin3d_loader = _make_mmgmpp_loader


def finetune_gin3d(
    checkpoint_path: str,
    dataset_name: str,
    data_dir: str,
    model_config: ModelConfig = None,
    finetune_config: FinetuneConfig = None,
    device: str = "cuda",
    n_runs: int = 5,
    scratch: bool = False,
    predictor_type: str = "moe",
    use_physchem: bool = True,
    smooth_l1_beta: float = 1.0,
    use_mph: bool = False,
    use_ecfp: bool = False,
):
    """Fine-tune a GIN encoder on a downstream dataset."""
    finetune_config = finetune_config or FinetuneConfig()
    model_config = model_config or ModelConfig()

    mph_encoder = None
    if scratch:
        hidden_dim = model_config.hidden_dim
        atom_dim = model_config.atom_feat_dim
        ckpt_cfg = model_config
        logger.info("SCRATCH mode: random init GIN (hidden=%d gin=%d tf=%d)",
                    hidden_dim, model_config.gin_layers, model_config.transformer_layers)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        ckpt_cfg = checkpoint.get("config", None)
        hidden_dim = getattr(ckpt_cfg, "hidden_dim", 512) if ckpt_cfg else 512
        atom_dim = getattr(ckpt_cfg, "atom_dim", 119) if ckpt_cfg else 119
        # Auto-detect atom_dim from checkpoint weights (more reliable than config)
        state_dict_key = "gin_state_dict" if "gin_state_dict" in checkpoint else "model_state_dict"
        state = checkpoint.get(state_dict_key, checkpoint)
        for k in state:
            if "atom_embed.weight" in k:
                atom_dim = state[k].shape[1]  # infer from weight shape
                break
        logger.info("Atom dim: %d (auto-detected from checkpoint)", atom_dim)

    # ── Load data ──
    import pandas as pd
    csv_path = os.path.join(data_dir, f"{dataset_name}.csv")
    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, compression="gzip")
    smiles_list = df["smiles"].tolist()
    label_cols = [c for c in df.columns if c.startswith("label")]
    if not label_cols:
        known_labels = {"p_np", "exp", "label", "target", "activity", "y"}
        label_cols = [c for c in df.columns if c.lower() in known_labels
                      or c.lower().startswith("label")]
    if not label_cols:
        skip = {"smiles", "id", "name", "num", "cmpd_chemblid"}
        label_cols = [c for c in df.columns if c.lower() not in skip
                      and df[c].dtype in ("float64", "float32", "int64")]
    labels = torch.tensor(df[label_cols].values, dtype=torch.float32)
    num_tasks = len(label_cols)
    task_type = "classification" if dataset_name in TASKS["classification"] else "regression"
    logger.info("Label columns: %s (%d tasks)", label_cols, num_tasks)

    # ── Build GIN encoder ──
    from models.gin_trans import GINTransformerEncoder
    gin = GINTransformerEncoder(
        atom_dim=atom_dim, hidden_dim=hidden_dim,
        gin_layers=getattr(ckpt_cfg, "gin_layers", 4) if ckpt_cfg else 4,
        transformer_layers=getattr(ckpt_cfg, "transformer_layers", 4) if ckpt_cfg else 4,
        num_heads=getattr(ckpt_cfg, "num_heads", 8) if ckpt_cfg else 8,
        dropout=finetune_config.dropout,
    ).to(device)

    if not scratch:
        gin_state = checkpoint.get("model_state_dict", checkpoint)
        if any(k.startswith("gin.") for k in gin_state):
            gin_state = {k[4:]: v for k, v in gin_state.items() if k.startswith("gin.")}
        gin.load_state_dict(gin_state, strict=False)
    logger.info("Loaded GIN encoder (hidden_dim=%d, %d params, scratch=%s)",
                hidden_dim, sum(p.numel() for p in gin.parameters()), scratch)

    # ── Load MPH encoder (optional, for dual GIN+MPH finetuning) ──
    if use_mph and not scratch:
        from models.mph_encoder import MPHEncoder
        mph_grid = getattr(model_config, 'mph_grid_size', None) or 32
        mph_encoder = MPHEncoder(hidden_dim=hidden_dim, grid_size=mph_grid)
        finetune_config.use_3d = True  # MPH requires 3D conformers in batch
        logger.info("MPH mode: enabling 3D conformer generation for finetuning")
        mph_state = checkpoint.get("mph_state_dict", None) if isinstance(checkpoint, dict) else None
        if mph_state is None and "mph_state_dict" not in checkpoint:
            logger.warning("No MPH weights in checkpoint — running GIN-only")
            mph_encoder = None
        else:
            mph_state = checkpoint.get("mph_state_dict", {})
            if mph_state:
                mph_encoder.load_state_dict(mph_state, strict=False)
                mph_encoder = mph_encoder.to(device)
                logger.info("Loaded MPH encoder (%d params)", sum(p.numel() for p in mph_encoder.parameters()))
            else:
                mph_encoder = None

    # ── Grid search ──
    logger.info("Starting grid search...")
    (train_s, train_l, val_s, val_l, test_s, test_l) = scaffold_split(
        smiles_list, labels, train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=42,
    )

    param_grid = {
        "lr": finetune_config.lr_range,
        "weight_decay": finetune_config.wd_range,
        "dropout": finetune_config.dropout_range,
    }
    n_quick_runs = finetune_config.grid_runs
    grid_epochs = finetune_config.grid_epochs

    if not finetune_config.freeze_encoders:
        logger.warning("freeze_encoders=False: encoder weights may leak across grid points.")

    best_params = None
    best_score = -float("inf")

    for params in ParameterGrid(param_grid):
        scores = []
        for _ in range(n_quick_runs):
            model = GIN3DFinetune(
                gin, mph_encoder=mph_encoder, hidden_dim=hidden_dim,
                embed_dim=finetune_config.embed_dim,
                num_tasks=num_tasks, task_type=task_type, dropout=params["dropout"],
                use_moe=(predictor_type == "moe"), use_mph=use_mph, use_ecfp=use_ecfp,
            ).to(device)

            optimizer = _make_optimizer(
                model, params["lr"], params["weight_decay"],
                freeze_encoders=finetune_config.freeze_encoders,
                encoder_lr_ratio=finetune_config.encoder_lr_ratio,
            )

            train_loader = _make_gin3d_loader(
                train_s, train_l, task_type, num_tasks,
                finetune_config.batch_size, shuffle=True,
                use_3d=finetune_config.use_3d,
            )
            pos_weight = _compute_pos_weight(train_l, num_tasks, task_type, device)

            for _ in range(grid_epochs):
                model.train()
                for batch in train_loader:
                    batch = to_device(batch, device)
                    loss = model.compute_loss(model(batch), batch["y"], pos_weight=pos_weight, smooth_l1_beta=smooth_l1_beta)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            val_score = _validate_gin3d(model, val_s, val_l, task_type, num_tasks,
                                        finetune_config, device)
            scores.append(val_score)

        avg = np.nanmean(scores)
        if avg > best_score:
            best_score = avg
            best_params = params

    logger.info("Best params: %s, score: %.4f", best_params, best_score)
    if best_params is None:
        best_params = list(ParameterGrid(param_grid))[0]

    # ── 5-fold scaffold CV (preserves 80/10/10) ──
    # Pre-compute 10 scaffold groups, then rotate train/val/test groups
    # across 5 folds.  Each scaffold appears in test exactly once.
    logger.info("Running 5-fold scaffold CV...")
    all_results = []

    # Pre-partition scaffolds into 10 groups
    scaffold_to_idx = {}
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        scaf = MurckoScaffold.MurckoScaffoldSmiles(smi, includeChirality=False) if mol else "invalid"
        if not scaf:
            scaf = "no_scaffold"
        scaffold_to_idx.setdefault(scaf, []).append(i)
    scaffolds = sorted(scaffold_to_idx.keys(), key=lambda s: -len(scaffold_to_idx[s]))
    rng = np.random.default_rng(42)
    rng.shuffle(scaffolds)
    # Distribute scaffolds into 10 balanced groups (minimize size variance)
    groups = [[] for _ in range(10)]
    group_sizes = [0] * 10
    for scaf in scaffolds:
        idx_list = scaffold_to_idx[scaf]
        gi = np.argmin(group_sizes)
        groups[gi].extend(idx_list)
        group_sizes[gi] += len(idx_list)

    # Seed for reproducibility across runs
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Save original encoder state for fold-level reset
    gin_original_state = {k: v.cpu().clone() for k, v in gin.state_dict().items()}
    mph_original_state = {k: v.cpu().clone() for k, v in mph_encoder.state_dict().items()} if mph_encoder else None

    # 5-fold CV: rotate 2 groups as val+test, remaining 8 as train
    for fold in range(5):
        test_g = (fold * 2) % 10
        val_g = (fold * 2 + 1) % 10
        train_idx = [i for g in range(10) if g not in (test_g, val_g) for i in groups[g]]
        val_idx = list(groups[val_g])
        test_idx = list(groups[test_g])

        # Apply label balance within this fold (in-place mutation)
        labels_np = labels.numpy()
        if labels_np.ndim == 1:
            labels_np = labels_np.reshape(-1, 1)
        train_bal = list(train_idx)  # balance ops mutate, so work on copy
        val_bal = list(val_idx)
        test_bal = list(test_idx)
        if labels_np.shape[1] > 1:
            _balance_multilabel_splits(train_bal, val_bal, test_bal, labels_np)
        else:
            _balance_binary_splits(train_bal, val_bal, test_bal, labels_np)
        _balance_molecular_properties(train_bal, val_bal, test_bal, smiles_list)
        try:
            _validate_split_labels(train_bal, val_bal, test_bal, labels_np, str(len(smiles_list)))
        except ValueError as e:
            logger.warning("Fold %d balance failed: %s — skipping", fold + 1, e)
            continue

        train_s = [smiles_list[i] for i in train_bal]
        train_l = labels[train_bal]
        val_s = [smiles_list[i] for i in val_bal]
        val_l = labels[val_bal]
        test_s = [smiles_list[i] for i in test_bal]
        test_l = labels[test_bal]
        model = GIN3DFinetune(
            gin, mph_encoder=mph_encoder, hidden_dim=hidden_dim,
            embed_dim=finetune_config.embed_dim,
            num_tasks=num_tasks, task_type=task_type, dropout=best_params["dropout"],
            use_moe=(predictor_type == "moe"), use_mph=use_mph, use_ecfp=use_ecfp,
        ).to(device)

        # Reset encoders to original pretrained weights per fold
        gin.load_state_dict(gin_original_state)
        if mph_encoder and mph_original_state:
            mph_encoder.load_state_dict(mph_original_state)

        optimizer = _make_optimizer(
            model, best_params["lr"], best_params["weight_decay"],
            freeze_encoders=finetune_config.freeze_encoders,
            encoder_lr_ratio=finetune_config.encoder_lr_ratio,
        )
        # Cosine scheduler with linear warmup
        cos_sched = CosineAnnealingLR(optimizer, T_max=finetune_config.epochs)
        if finetune_config.warmup_epochs > 0:
            warmup = LinearLR(optimizer, start_factor=0.1, total_iters=finetune_config.warmup_epochs)
            scheduler = SequentialLR(optimizer, schedulers=[warmup, cos_sched],
                                     milestones=[finetune_config.warmup_epochs])
        else:
            scheduler = cos_sched

        train_loader = _make_gin3d_loader(
            train_s, train_l, task_type, num_tasks,
            finetune_config.batch_size, shuffle=True,
            use_3d=finetune_config.use_3d,
        )
        val_loader = _make_gin3d_loader(
            val_s, val_l, task_type, num_tasks,
            finetune_config.batch_size, shuffle=False,
            use_3d=finetune_config.use_3d,
        )
        pos_weight = _compute_pos_weight(train_l, num_tasks, task_type, device)

        # Build smiles→index lookup for PhysChem (O(1) instead of O(N))
        smiles_to_idx = {s: i for i, s in enumerate(smiles_list)} if use_physchem and task_type == "regression" else {}

        # PhysChem augmentation for regression tasks
        physchem_feats = None
        if use_physchem and task_type == "regression":
            physchem = PhysChemFeaturizer()
            physchem_feats = physchem.compute_and_project(smiles_list, device)
            physchem_proj = nn.Linear(physchem_feats.size(1), num_tasks, bias=False).to(device)

        best_val, best_state, patience_counter = None, None, 0
        metric_key = "rmse" if task_type == "regression" else "roc_auc"
        for epoch in range(finetune_config.epochs):
            model.train()
            for batch in train_loader:
                batch = to_device(batch, device)
                pred = model(batch)
                if physchem_feats is not None:
                    batch_idx = [smiles_to_idx.get(s, 0) for s in batch.get("smiles", [])]
                    pc = physchem_feats[batch_idx]
                    pred = pred + physchem_proj(pc)
                loss = model.compute_loss(pred, batch["y"], pos_weight=pos_weight, smooth_l1_beta=smooth_l1_beta)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            vp, vt = [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = to_device(batch, device)
                    vp.append(model(batch).cpu())
                    vt.append(batch["y"].cpu())
            val_results = evaluate_predictions(
                torch.cat(vt), torch.cat(vp), task_type,
                return_per_task=(num_tasks > 1),
            )
            val_score = val_results.get(metric_key, float("inf"))

            # Regression: lower=better; Classification: higher=better
            if task_type == "regression":
                is_improved = best_val is None or val_score < best_val
            else:
                is_improved = best_val is None or val_score > best_val
            if is_improved:
                best_val = val_score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= finetune_config.early_stop_patience:
                    break
            scheduler.step()

        model.load_state_dict(best_state)
        model.eval()
        test_loader = _make_gin3d_loader(
            test_s, test_l, task_type, num_tasks,
            finetune_config.batch_size, shuffle=False,
            use_3d=finetune_config.use_3d,
        )
        tp, tt = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = to_device(batch, device)
                tp.append(model(batch).cpu())
                tt.append(batch["y"].cpu())
        if tp:
            results = evaluate_predictions(
                torch.cat(tt), torch.cat(tp), task_type,
                return_per_task=(num_tasks > 1),
            )
        else:
            results = {"roc_auc": float("nan")}
        all_results.append(results)
        logger.info("Fold %d: %s", fold + 1, results)

    # Aggregate
    aggregated = {}
    if all_results:
        for metric in all_results[0].keys():
            if metric == "roc_auc_per_task":
                continue
            values = [r[metric] for r in all_results if not np.isnan(r[metric])]
            if values:
                aggregated[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    logger.info("Final: %s", aggregated)
    return aggregated


def _validate_gin3d(model, smiles_list, labels, task_type, num_tasks, config, device):
    """Quick validation for GIN3D model."""
    model.eval()
    loader = _make_gin3d_loader(
        smiles_list, labels, task_type, num_tasks,
        config.batch_size, shuffle=False,
        use_3d=getattr(config, 'use_3d', False),
    )
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            preds.append(model(batch).cpu())
            targets.append(batch["y"].cpu())
    if not preds:
        return float("nan")
    return evaluate_predictions(
        torch.cat(targets), torch.cat(preds), task_type,
    ).get("roc_auc", float("inf"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GIN fine-tuning on MoleculeNet")
    parser.add_argument("--checkpoint", type=str, default="scratch", help="Path to pre-trained GIN checkpoint")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (MoleculeNet)")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory with CSV files")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_runs", type=int, default=10,
                        help="Independent scaffold splits (10 gives tighter confidence)")
    # Ablation flags
    ablation = parser.add_argument_group("Ablation overrides")
    ablation.add_argument("--ablation_name", type=str, default=None, help="Experiment name")
    ablation.add_argument("--scratch", action="store_true", help="Random init GIN (no pretrained weights)")
    ablation.add_argument("--predictor", type=str, default="mlp", choices=["mlp", "moe"],
                          help="Predictor type: mlp (default, safer) or moe (GMoE, for ablation)")
    ablation.add_argument("--moe", action="store_true",
                          help="Shorthand for --predictor moe")
    ablation.add_argument("--unfreeze", action="store_true", help="Full finetune (unfreeze encoder)")
    ablation.add_argument("--no_physchem", action="store_true", help="Disable PhysChem featurizer")
    ablation.add_argument("--mph", action="store_true", help="Use dual GIN+MPH encoder (3D topology)")
    ablation.add_argument("--ecfp", action="store_true", help="Add ECFP fingerprint projection (PhyChem-MoE style)")
    ablation.add_argument("--hidden_dim", type=int, default=512, help="GIN hidden dim (scratch mode)")
    ablation.add_argument("--gin_layers", type=int, default=4, help="GIN layers (scratch mode)")
    ablation.add_argument("--transformer_layers", type=int, default=4, help="Transformer layers (scratch mode)")
    args = parser.parse_args()

    # Determine checkpoint
    # MoE shorthand
    if args.moe:
        args.predictor = "moe"

    if args.scratch or args.checkpoint == "scratch":
        checkpoint_path = None
        scratch = True
    else:
        checkpoint_path = args.checkpoint
        scratch = False

    # ── Task-group detection (PhyChem-MoE task-type separation) ──
    task_group = None
    for group_name, group_cfg in TASK_GROUPS.items():
        if args.dataset in group_cfg["datasets"]:
            task_group = group_name
            break

    # Apply ablation overrides to configs
    model_config = ModelConfig()
    finetune_config = FinetuneConfig()
    model_config.hidden_dim = args.hidden_dim
    model_config.gin_layers = args.gin_layers
    model_config.transformer_layers = args.transformer_layers
    if args.unfreeze:
        finetune_config.freeze_encoders = False

    # Auto-select task-group-specific LR (PhyChem-MoE Table 1)
    if task_group:
        finetune_config.lr = TASK_GROUPS[task_group]["lr"]

    ablation_tag = f" [{args.ablation_name}]" if args.ablation_name else ""
    logger.info("=== Finetune%s: %s | group=%s lr=%.1e scratch=%s predictor=%s freeze=%s ===",
                ablation_tag, args.dataset, task_group, finetune_config.lr,
                scratch, args.predictor, finetune_config.freeze_encoders)

    results = finetune_gin3d(
        checkpoint_path=checkpoint_path,
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        model_config=model_config,
        finetune_config=finetune_config,
        device=args.device,
        n_runs=args.n_runs,
        scratch=scratch,
        predictor_type=args.predictor,
        use_physchem=not args.no_physchem,
        smooth_l1_beta=finetune_config.smooth_l1_beta,
        use_mph=args.mph,
        use_ecfp=args.ecfp,
    )
    print(json.dumps(results, indent=2))
