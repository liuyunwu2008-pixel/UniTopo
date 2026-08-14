"""Molecular graph featurization for MMG-MPP.

Constructs atomic graphs with 78-dim node features and 12-dim edge features.
"""

from typing import Tuple, Optional
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from torch_geometric.data import Data


class AtomFeaturizer:
    """78-dimensional atom feature vectors."""

    # Atom type mapping (atomic numbers up to 53 commonly found in drugs)
    ATOM_TYPES = [1, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20,
                  25, 26, 27, 28, 29, 30, 33, 34, 35, 46, 53]

    DEGREES = [0, 1, 2, 3, 4, 5, 6]
    FORMAL_CHARGES = [-3, -2, -1, 0, 1, 2, 3]
    HYBRIDIZATIONS = [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    CHIRALITIES = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ]
    NUM_HYDROGENS = [0, 1, 2, 3, 4]

    @classmethod
    def get_dim(cls) -> int:
        return (len(cls.ATOM_TYPES) + len(cls.DEGREES) +
                len(cls.FORMAL_CHARGES) + len(cls.HYBRIDIZATIONS) +
                len(cls.CHIRALITIES) + len(cls.NUM_HYDROGENS) + 3)

    @classmethod
    def featurize(cls, atom: Chem.Atom) -> np.ndarray:
        features = []

        # Atom type (one-hot, 26 dims)
        atom_type = cls._onehot(atom.GetAtomicNum(), cls.ATOM_TYPES)
        features.append(atom_type)

        # Degree (one-hot, 7 dims)
        degree = cls._onehot(atom.GetTotalDegree(), cls.DEGREES)
        features.append(degree)

        # Formal charge (one-hot, 7 dims)
        charge = cls._onehot(atom.GetFormalCharge(), cls.FORMAL_CHARGES)
        features.append(charge)

        # Hybridization (one-hot, 6 dims)
        hyb = cls._onehot(atom.GetHybridization(), cls.HYBRIDIZATIONS)
        features.append(hyb)

        # Chirality (one-hot, 4 dims)
        chirality = cls._onehot(atom.GetChiralTag(), cls.CHIRALITIES)
        features.append(chirality)

        # Number of hydrogens (one-hot, 5 dims)
        num_h = cls._onehot(atom.GetTotalNumHs(), cls.NUM_HYDROGENS)
        features.append(num_h)

        # Aromaticity (1 dim)
        features.append(np.array([float(atom.GetIsAromatic())]))

        # Ring membership (1 dim)
        features.append(np.array([float(atom.IsInRing())]))

        # Mass (1 dim, normalized)
        features.append(np.array([atom.GetMass() * 0.01]))

        return np.concatenate(features).astype(np.float32)

    @staticmethod
    def _onehot(value, categories) -> np.ndarray:
        vec = np.zeros(len(categories))
        try:
            idx = categories.index(value)
            vec[idx] = 1.0
        except ValueError:
            pass
        return vec


class BondFeaturizer:
    """12-dimensional bond feature vectors."""

    BOND_TYPES = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    STEREOS = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ]

    @classmethod
    def get_dim(cls) -> int:
        return len(cls.BOND_TYPES) + 3 + len(cls.STEREOS) + 1

    @classmethod
    def featurize(cls, bond: Chem.Bond) -> np.ndarray:
        features = []

        # Bond type (one-hot, 4 dims)
        bt = cls._onehot(bond.GetBondType(), cls.BOND_TYPES)
        features.append(bt)

        # Conjugated (1 dim)
        features.append(np.array([float(bond.GetIsConjugated())]))

        # In ring (1 dim)
        features.append(np.array([float(bond.IsInRing())]))

        # Aromatic (treat as bool, 1 dim)
        features.append(np.array([float(bond.GetBondType() == Chem.rdchem.BondType.AROMATIC)]))

        # Stereochemistry (one-hot, 4 dims)
        stereo = cls._onehot(bond.GetStereo(), cls.STEREOS)
        features.append(stereo)

        # Rotatable (1 dim, estimated)
        features.append(np.array([float(_is_rotatable(bond))]))

        return np.concatenate(features).astype(np.float32)

    @staticmethod
    def _onehot(value, categories) -> np.ndarray:
        vec = np.zeros(len(categories))
        try:
            idx = categories.index(value)
            vec[idx] = 1.0
        except ValueError:
            pass
        return vec


def _is_rotatable(bond: Chem.Bond) -> bool:
    """Check if a bond is rotatable (single, non-ring, non-terminal)."""
    if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
        return False
    if bond.IsInRing():
        return False
    a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
    if a1.GetDegree() == 1 or a2.GetDegree() == 1:
        return False
    return True


def construct_atomic_graph(
    mol: Chem.Mol,
    add_virtual_node: bool = True,
) -> Data:
    """Construct atomic graph with atom/bond features.

    Args:
        mol: RDKit molecule
        add_virtual_node: If True, adds a virtual node connected to all atoms

    Returns:
        PyG Data object with x, edge_index, edge_attr
    """
    n_atoms = mol.GetNumAtoms()

    # Node features
    x = np.stack([AtomFeaturizer.featurize(mol.GetAtomWithIdx(i))
                   for i in range(n_atoms)])

    # Edge features
    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = BondFeaturizer.featurize(bond)
        # Undirected: add both directions
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(feat)
        edge_attr.append(feat)

    if len(edge_index) == 0:
        # Single-atom molecule
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, BondFeaturizer.get_dim()), dtype=np.float32)
    else:
        edge_index = np.array(edge_index, dtype=np.int64).T
        edge_attr = np.array(edge_attr, dtype=np.float32)

    # Virtual node
    if add_virtual_node:
        virtual_offset = n_atoms
        virtual_feat = np.ones((1, AtomFeaturizer.get_dim()), dtype=np.float32)
        x = np.concatenate([x, virtual_feat], axis=0)

        virtual_edges = np.array(
            [[i, virtual_offset] for i in range(n_atoms)] +
            [[virtual_offset, i] for i in range(n_atoms)],
            dtype=np.int64
        ).T
        virtual_attr = np.zeros((virtual_edges.shape[1], BondFeaturizer.get_dim()),
                                dtype=np.float32)

        edge_index = np.concatenate([edge_index, virtual_edges], axis=1)
        edge_attr = np.concatenate([edge_attr, virtual_attr], axis=0)

    return Data(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr),
        n_atoms=n_atoms,
    )


def generate_3d_conformer(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Generate a low-energy 3D conformer using ETKDGv3 + MMFF94.

    Returns a copy of mol with 3D coordinates, or None if generation fails.
    """
    mol = Chem.Mol(mol)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    mol = Chem.RemoveHs(mol)
    return mol
