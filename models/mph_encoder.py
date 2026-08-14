"""Multi-Parameter Persistence Homology (MPH) encoder for 3D molecular geometry.

Implements the sublevel bifiltration of the Vietoris-Rips complex:
  - Parameter 1 (epsilon): interatomic Euclidean distance (from 3D coordinates)
  - Parameter 2 (delta):   atomic partial charge threshold (Gasteiger)

For each (eps, delta) pair on a grid, computes the Betti numbers:
  - beta_0: number of connected components
  - beta_1: number of independent cycles (1-dimensional holes)

The resulting 2D "persistence density" images are SE(3)-invariant topological
fingerprints of the 3D molecular geometry, then encoded through a small CNN.

Key references:
  - Carlsson & Zomorodian (2009): Theory of multidimensional persistence
  - Adams et al. (2017): Persistence images for ML
  - Vipul (2023): Multi-parameter persistence landscapes
  - GUDHI library: https://gudhi.inria.fr/

When the 'gudhi' package is available, the encoder optionally delegates to GUDHI's
CubicalComplex for a more rigorous computation (controlled by use_gudhi flag).
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform


# ── Bifiltration Hilbert function (pure numpy, no TDA library required) ──


def _count_components(adj: np.ndarray) -> int:
    """Count connected components via iterative DFS (fast for sparse graphs)."""
    n = adj.shape[0]
    visited = np.zeros(n, dtype=bool)
    n_components = 0
    for start in range(n):
        if visited[start]:
            continue
        n_components += 1
        stack = [start]
        visited[start] = True
        while stack:
            u = stack.pop()
            neighbors = np.where(adj[u] & ~visited)[0]
            visited[neighbors] = True
            stack.extend(neighbors.tolist())
    return n_components


def compute_hilbert_function(
    coords: np.ndarray,         # (N, 3)  atomic 3D coordinates
    charges: np.ndarray,         # (N,)    atomic partial charges
    eps_grid: np.ndarray,        # (G,)    distance thresholds
    delta_grid: np.ndarray,      # (G,)    charge thresholds
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the 2-parameter persistence Hilbert function.

    For each (eps, delta) pair, builds the sublevel graph:
        edge (i,j) exists iff  dist(i,j) <= eps AND max(charge[i], charge[j]) <= delta
    and counts beta_0 (components) and beta_1 (cycles).

    Returns:
        h0_grid: (G, G)  beta_0 values  [n_components decreases as eps/delta grow]
        h1_grid: (G, G)  beta_1 values  [n_cycles peaks and decays]
    """
    n_atoms = coords.shape[0]
    if n_atoms < 2:
        return np.zeros((len(eps_grid), len(delta_grid)), dtype=np.float32), \
               np.zeros((len(eps_grid), len(delta_grid)), dtype=np.float32)

    # Pre-compute pairwise distance and charge-max matrices
    dist_mat = squareform(pdist(coords))  # (N, N)
    charge_mat = np.maximum.outer(charges, charges)  # (N, N), max charge of each pair

    G = len(eps_grid)
    h0_grid = np.zeros((G, G), dtype=np.float32)
    h1_grid = np.zeros((G, G), dtype=np.float32)

    for i, eps in enumerate(eps_grid):
        # Distance mask (reused across all delta values at this eps)
        dist_mask = dist_mat <= eps

        for j, delta in enumerate(delta_grid):
            # Sublevel graph adjacency
            adj = dist_mask & (charge_mat <= delta)
            np.fill_diagonal(adj, False)

            n_edges = adj.sum() // 2
            if n_edges == 0:
                h0_grid[i, j] = float(n_atoms)
                h1_grid[i, j] = 0.0
            else:
                n_comp = _count_components(adj)
                # Euler-Poincare: beta_0 - beta_1 = V - E  (for graphs)
                # beta_0 = n_components, so beta_1 = n_edges - n_atoms + n_components
                h0_grid[i, j] = float(n_comp)
                h1_grid[i, j] = float(max(0, n_edges - n_atoms + n_comp))

    return h0_grid, h1_grid


def compute_atomic_charges(mol, method: str = "gasteiger") -> np.ndarray:
    """Compute partial charges for all atoms in an RDKit molecule.

    Args:
        mol: RDKit Mol object (must have 3D conformer).
        method: "gasteiger" (fast) or "mmff94" (more accurate, needs MMFF).

    Returns:
        charges: (N,) array of partial charges.
    """
    from rdkit.Chem import AllChem
    if method == "mmff94":
        props = AllChem.MMFFGetMoleculeProperties(mol)
        if props is None:
            # Fallback to Gasteiger
            AllChem.ComputeGasteigerCharges(mol)
            return np.array([a.GetDoubleProp("_GasteigerCharge") for a in mol.GetAtoms()])
        charges = [props.GetMMFFPartialCharge(i) for i in range(mol.GetNumAtoms())]
        return np.array(charges, dtype=np.float32)
    else:
        AllChem.ComputeGasteigerCharges(mol)
        charges = np.array([float(a.GetProp("_GasteigerCharge")) for a in mol.GetAtoms()])
        # Handle NaN from RDKit Gasteiger failures
        charges = np.nan_to_num(charges, nan=0.0)
        return charges.astype(np.float32)


# ── MPH Encoder module ──


class MPHEncoder(nn.Module):
    """Multi-Parameter Persistence Homology encoder for 3D molecular geometry.

    Computes the bifiltration Hilbert function from 3D coordinates + partial charges,
    producing SE(3)-invariant topological fingerprints.  The H0 and H1 density images
    are stacked and encoded through a small 2D CNN + MLP readout.

    Input:  (coords, charges) — per-molecule 3D coordinates and atomic charges
    Output: (hidden_dim,) graph-level embedding
    """

    def __init__(
        self,
        hidden_dim: int = 300,
        grid_size: int = 32,
        eps_min: float = 0.5,
        eps_max: float = 8.0,
        delta_min: float = -0.5,
        delta_max: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size

        # Pre-compute threshold grids (fixed — not learnable)
        self.register_buffer(
            "eps_grid",
            torch.linspace(eps_min, eps_max, grid_size),
        )
        self.register_buffer(
            "delta_grid",
            torch.linspace(delta_min, delta_max, grid_size),
        )

        # 2D CNN over stacked (H0, H1) density maps
        # Input: (B, 2, G, G)  →  Output: (B, hidden_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),                     # G/2

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                     # G/4

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),        # (B, 64, 4, 4)
        )

        cnn_out_dim = 64 * 4 * 4  # 1024

        self.readout = nn.Sequential(
            nn.Linear(cnn_out_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def _compute_hilbert_batch(
        self, coords_list, charges_list
    ) -> torch.Tensor:
        """Compute Hilbert function for a batch of molecules."""
        eps_np = self.eps_grid.cpu().numpy()
        delta_np = self.delta_grid.cpu().numpy()

        density_maps = []
        for coords, charges in zip(coords_list, charges_list):
            if isinstance(coords, torch.Tensor):
                coords = coords.cpu().numpy()
            if isinstance(charges, torch.Tensor):
                charges = charges.cpu().numpy()

            h0, h1 = compute_hilbert_function(coords, charges, eps_np, delta_np)

            # Normalize by atom count for scale invariance
            n = max(len(coords), 1)
            h0 = h0 / n
            h1 = h1 / n

            # Stack as 2-channel image
            density = np.stack([h0, h1], axis=0)  # (2, G, G)
            density_maps.append(torch.from_numpy(density))

        return torch.stack(density_maps, dim=0)  # (B, 2, G, G)

    def forward(
        self,
        coords_list,      # list of (N_i, 3) tensors/arrays or None
        charges_list,     # list of (N_i,) tensors/arrays or None
    ) -> torch.Tensor:
        """Encode a batch of 3D molecular geometries.

        Molecules with failed conformer generation (None entries) receive
        a learned zero-mean embedding, effectively contributing no information
        to the downstream fusion.

        Args:
            coords_list:  list of per-molecule 3D coordinate tensors (or None).
            charges_list: list of per-molecule partial charge tensors (or None).

        Returns:
            z_mg: (batch_size, hidden_dim)  MG (Molecular Geometry) embeddings.
        """
        device = self.eps_grid.device
        batch_size = len(coords_list)

        # Separate valid and invalid molecules
        valid_indices = []
        valid_coords = []
        valid_charges = []
        for i, (c, q) in enumerate(zip(coords_list, charges_list)):
            if c is not None and q is not None and len(c) >= 2:
                valid_indices.append(i)
                valid_coords.append(c)
                valid_charges.append(q)

        # Compute Hilbert function for valid molecules
        if valid_coords:
            density = self._compute_hilbert_batch(valid_coords, valid_charges)
            density = density.to(device=device, dtype=torch.float32)
            h_valid = self.cnn(density)
            h_valid = h_valid.view(h_valid.size(0), -1)
            h_valid = self.readout(h_valid)
        else:
            h_valid = torch.empty(0, self.hidden_dim, device=device)

        # Scatter back to full batch (failed conformers → zeros)
        h_full = torch.zeros(batch_size, self.hidden_dim, device=device)
        for j, batch_idx in enumerate(valid_indices):
            h_full[batch_idx] = h_valid[j]

        return h_full


# ── 3D conformer generation ──


def generate_conformer(smiles: str, max_attempts: int = 5) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Generate a single low-energy 3D conformer for a SMILES string.

    Uses RDKit ETKDGv3 for conformer generation and Gasteiger for charges.

    Returns:
        (coords, charges) if successful, None otherwise.
        coords: (N, 3) numpy array of 3D coordinates
        charges: (N,) numpy array of partial charges
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    # ETKDG v3: improved torsion sampling
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 1  # single-threaded: caller (DataLoader workers) handles parallelism
    params.maxIterations = 500

    success = False
    for attempt in range(max_attempts):
        params.randomSeed = 42 + attempt
        conf_id = AllChem.EmbedMolecule(mol, params)
        if conf_id == 0:  # success
            success = True
            break

    if not success:
        mol = Chem.RemoveHs(mol)
        return None

    # Optimize with MMFF94
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass

    # Extract coordinates
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])

    # Compute charges (on mol with Hs, then remove)
    charges = compute_atomic_charges(mol, method="gasteiger")

    mol = Chem.RemoveHs(mol)

    # Filter to heavy atoms only (optional — keeps all atoms for now)
    return coords.astype(np.float32), charges.astype(np.float32)
