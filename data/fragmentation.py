"""Molecular fragmentation: junction tree, BRICS, and functional groups."""

from typing import Dict, List, Set, Tuple
from collections import defaultdict
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import BRICS
from torch_geometric.data import Data

SMARTS_PATTERNS = {
    "carboxyl": "[CX3](=O)[OX2H1]",
    "amino_primary": "[NX3;H2;!$(NC=O)]",
    "amino_secondary": "[NX3;H1;!$(NC=O)]",
    "amino_tertiary": "[NX3;H0;!$(NC=O)]",
    "hydroxyl": "[OX2H]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "ester": "[#6][CX3](=O)[OX2H0][#6]",
    "amide": "[NX3][CX3](=O)[#6]",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
    "nitrile": "[NX1]#[CX2]",
    "phenyl": "c1ccccc1",
    "pyridine": "c1ccncc1",
    "thiol": "[#16X2H]",
    "ether": "[OD2]([#6])[#6]",
}

def brics_decompose(mol):
    bonds = list(BRICS.FindBRICSBonds(mol))
    if not bonds:
        return [mol]
    bond_indices = [(b[0][0], b[0][1]) for b in bonds]
    editable = Chem.EditableMol(mol)
    for a1, a2 in sorted(bond_indices, reverse=True):
        try:
            editable.RemoveBond(a1, a2)
        except Exception:
            pass
    fragmented = editable.GetMol()
    try:
        Chem.SanitizeMol(fragmented)
    except Exception:
        return [mol]
    return [f for f in Chem.GetMolFrags(fragmented, asMols=True) if f.GetNumAtoms() > 0]

def functional_group_matching(mol):
    matches = {}
    for name, smarts in SMARTS_PATTERNS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue
        hits = mol.GetSubstructMatches(pattern)
        if hits:
            matches[name] = [set(h) for h in hits]
    return matches

def junction_tree_decomposition(mol):
    rings = mol.GetRingInfo().AtomRings()
    clusters = []
    for ring in rings:
        ring_set = set(ring)
        merged = False
        for cluster in clusters:
            if len(ring_set & cluster) >= 2:
                cluster.update(ring_set)
                merged = True
                break
        if not merged:
            clusters.append(ring_set)
    n_atoms = mol.GetNumAtoms()
    atom_assignments = [-1] * n_atoms
    for i, cluster in enumerate(clusters):
        for atom_idx in cluster:
            atom_assignments[atom_idx] = i
    for atom_idx in range(n_atoms):
        if atom_assignments[atom_idx] >= 0:
            continue
        atom = mol.GetAtomWithIdx(atom_idx)
        assigned = False
        for nb in atom.GetNeighbors():
            if atom_assignments[nb.GetIdx()] >= 0:
                atom_assignments[atom_idx] = atom_assignments[nb.GetIdx()]
                clusters[atom_assignments[nb.GetIdx()]].add(atom_idx)
                assigned = True
                break
        if not assigned:
            clusters.append({atom_idx})
            atom_assignments[atom_idx] = len(clusters) - 1
    n_clusters = len(clusters)
    x = np.zeros((n_clusters, 10), dtype=np.float32)
    for i, cluster in enumerate(clusters):
        x[i, 0] = len(cluster) / 20.0
        x[i, 1] = float(any(mol.GetAtomWithIdx(a).IsInRing() for a in cluster))
        x[i, 2] = sum(mol.GetAtomWithIdx(a).GetAtomicNum() for a in cluster) / 100.0
        x[i, 3] = sum(mol.GetAtomWithIdx(a).GetFormalCharge() for a in cluster)
        x[i, 4] = sum(mol.GetAtomWithIdx(a).GetIsAromatic() for a in cluster) / max(len(cluster), 1)
        x[i, 5] = sum(mol.GetAtomWithIdx(a).GetTotalDegree() for a in cluster) / max(len(cluster), 1)
        elements = {mol.GetAtomWithIdx(a).GetAtomicNum() for a in cluster}
        x[i, 6] = len(elements) / 6.0
        x[i, 7] = 1.0 if 6 in elements else 0.0
        x[i, 8] = 1.0 if 7 in elements else 0.0
        x[i, 9] = 1.0 if 8 in elements else 0.0
    cluster_edges = set()
    for atom_idx in range(n_atoms):
        atom = mol.GetAtomWithIdx(atom_idx)
        cluster_i = atom_assignments[atom_idx]
        for nb in atom.GetNeighbors():
            cluster_j = atom_assignments[nb.GetIdx()]
            if cluster_i != cluster_j:
                cluster_edges.add((min(cluster_i, cluster_j), max(cluster_i, cluster_j)))
    if cluster_edges:
        edge_index = np.array([[e[0], e[1]] for e in cluster_edges] +
                              [[e[1], e[0]] for e in cluster_edges], dtype=np.int64).T
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    return Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(edge_index)), {
        "clusters": clusters, "atom_assignments": atom_assignments, "n_clusters": n_clusters}

def construct_fragment_graph(mol):
    fg_matches = functional_group_matching(mol)
    fragment_sets, fragment_names = [], []
    for name, match_list in fg_matches.items():
        for atom_set in match_list:
            if atom_set not in fragment_sets:
                fragment_sets.append(atom_set)
                fragment_names.append(name)
    covered_atoms = set()
    for fs in fragment_sets:
        covered_atoms.update(fs)
    remaining_atoms = set(range(mol.GetNumAtoms())) - covered_atoms
    if remaining_atoms:
        for frag in brics_decompose(mol):
            frag_atoms = {atom.GetIdx() for atom in frag.GetAtoms()}
            if frag_atoms & remaining_atoms and frag_atoms not in fragment_sets:
                fragment_sets.append(frag_atoms)
                fragment_names.append("brics")
    n_frags = len(fragment_sets)
    if n_frags <= 1:
        fragment_sets = [{i} for i in range(mol.GetNumAtoms())]
        fragment_names = ["atom"] * mol.GetNumAtoms()
        n_frags = len(fragment_sets)
    x = np.zeros((n_frags, 8), dtype=np.float32)
    for i, fs in enumerate(fragment_sets):
        x[i, 0] = len(fs) / 15.0
        elements = [mol.GetAtomWithIdx(a).GetAtomicNum() for a in fs]
        x[i, 1] = sum(elements) / (100.0 * max(len(fs), 1))
        x[i, 2] = float(any(mol.GetAtomWithIdx(a).IsInRing() for a in fs))
        x[i, 3] = float(any(mol.GetAtomWithIdx(a).GetIsAromatic() for a in fs))
        x[i, 4] = sum(mol.GetAtomWithIdx(a).GetFormalCharge() for a in fs)
        x[i, 5] = len(set(elements)) / 6.0
        x[i, 6] = 1.0 if 6 in elements else 0.0
        x[i, 7] = 1.0 if (7 in elements or 8 in elements) else 0.0
    atom_to_frag = {}
    for i, fs in enumerate(fragment_sets):
        for atom_idx in fs:
            atom_to_frag[atom_idx] = i
    edges = set()
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        f1 = atom_to_frag.get(a1, -1)
        f2 = atom_to_frag.get(a2, -1)
        if f1 >= 0 and f2 >= 0 and f1 != f2:
            edges.add((min(f1, f2), max(f1, f2)))
    if edges:
        edge_index = np.array([[e[0], e[1]] for e in edges] +
                              [[e[1], e[0]] for e in edges], dtype=np.int64).T
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    return Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(edge_index))
