"""Atomic molecular graph construction with RDKit."""

from typing import Tuple, List

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdmolops
from torch_geometric.data import Data

# Atom feature dimensions:
#   116 (atom type) + 11 (degree) + 5 (charge) + 4 (chiral)
#   + 5 (num H) + 5 (hyb) + 1 (aromatic) + 1 (ring) = 148
ATOM_FDIM = 148
BOND_FDIM = 11

# Periodic table atomic numbers up to Xe
ALLOWED_ATOMS = list(range(1, 55)) + list(range(56, 85)) + list(range(86, 119))

# Hybridization types
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

# Bond types
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def atom_features(atom: Chem.Atom) -> np.ndarray:
    """Extract atom features (148-dim)."""
    feats = []

    # Atom type (one-hot, 118 dims → we use 54)
    atom_type = np.zeros(len(ALLOWED_ATOMS))
    if atom.GetAtomicNum() in ALLOWED_ATOMS:
        atom_type[ALLOWED_ATOMS.index(atom.GetAtomicNum())] = 1
    feats.append(atom_type)

    # Degree (one-hot, 11 dims)
    degree = np.zeros(11)
    degree[min(atom.GetDegree(), 10)] = 1
    feats.append(degree)

    # Formal charge (one-hot, 5 dims)
    charge = np.zeros(5)
    charge[min(atom.GetFormalCharge() + 2, 4)] = 1
    feats.append(charge)

    # Chiral tag (one-hot, 4 dims)
    chiral = np.zeros(4)
    chiral[int(atom.GetChiralTag())] = 1
    feats.append(chiral)

    # Number of hydrogens (one-hot, 5 dims)
    num_h = np.zeros(5)
    num_h[min(atom.GetTotalNumHs(), 4)] = 1
    feats.append(num_h)

    # Hybridization (one-hot, 5 dims)
    hyb = np.zeros(len(HYBRIDIZATIONS))
    if atom.GetHybridization() in HYBRIDIZATIONS:
        hyb[HYBRIDIZATIONS.index(atom.GetHybridization())] = 1
    feats.append(hyb)

    # Aromaticity
    aromatic = np.array([int(atom.GetIsAromatic())])
    feats.append(aromatic)

    # Ring membership
    in_ring = np.array([int(atom.IsInRing())])
    feats.append(in_ring)

    return np.concatenate(feats)


def bond_features(bond: Chem.Bond) -> np.ndarray:
    """Extract bond features (11-dim)."""
    feats = []

    # Bond type (one-hot, 5 dims including no-bond)
    bt = np.zeros(len(BOND_TYPES) + 1)
    if bond is None:
        bt[0] = 1
    elif bond.GetBondType() in BOND_TYPES:
        bt[BOND_TYPES.index(bond.GetBondType()) + 1] = 1
    feats.append(bt)

    # Conjugated
    conjugated = np.array([int(bond.GetIsConjugated()) if bond else 0])
    feats.append(conjugated)

    # In ring
    in_ring = np.array([int(bond.IsInRing()) if bond else 0])
    feats.append(in_ring)

    # Stereochemistry (one-hot, 4 dims)
    stereo = np.zeros(4)
    if bond:
        stereo[int(bond.GetStereo())] = 1
    feats.append(stereo)

    return np.concatenate(feats)


def mol_to_graph(smiles: str) -> Data:
    """Convert a SMILES string to a PyG Data object (atomic graph).

    Returns:
        Data with x (N, ATOM_FDIM), edge_index (2, E), edge_attr (E, BOND_FDIM).

    Note: virtual nodes are handled by GINWithVirtualNode inside the model,
    not at the graph-construction level.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles}")

    # === Atom features ===
    x = np.array([atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)

    # === Edges ===
    src, dst, bond_fs = [], [], []
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        src.extend([u, v])
        dst.extend([v, u])
        bond_fs.extend([bf, bf])

    if len(src) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_FDIM), dtype=torch.float32)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(np.array(bond_fs), dtype=torch.float32)

    z = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)
    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        z=z,
        edge_index=edge_index,
        edge_attr=edge_attr,
        smiles=smiles,
    )
