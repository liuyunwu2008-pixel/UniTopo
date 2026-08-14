"""MoleculeNet dataset loading with four-view preprocessing."""

import torch
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

from .featurization import construct_atomic_graph, generate_3d_conformer
from .fragmentation import junction_tree_decomposition, construct_fragment_graph
from ..utils.mph import compute_mph_density_map


class MoleculeDataset(Dataset):
    """Converts SMILES to four molecular graph views per molecule."""

    def __init__(self, smiles_list, labels=None, use_3d=True):
        self.smiles_list = smiles_list
        self.labels = labels
        self.use_3d = use_3d

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        data_me = construct_atomic_graph(mol, add_virtual_node=True)
        data_mt, _ = junction_tree_decomposition(mol)
        data_fm = construct_fragment_graph(mol)

        result = {"atomic": data_me, "tree": data_mt, "fragment": data_fm}

        if self.use_3d:
            mol_3d = generate_3d_conformer(mol)
            if mol_3d is not None:
                conf = mol_3d.GetConformer()
                pos = np.array([list(conf.GetAtomPosition(i))
                                for i in range(mol_3d.GetNumAtoms())])
                AllChem.ComputeGasteigerCharges(mol_3d)
                charges = np.array([float(a.GetProp('_GasteigerCharge'))
                                    for a in mol_3d.GetAtoms()])
                h0, h1 = compute_mph_density_map(pos, charges)
                result["mph_maps"] = torch.from_numpy(np.stack([h0, h1]))
            else:
                result["mph_maps"] = torch.zeros(2, 64, 64)

        if self.labels is not None:
            result["label"] = torch.tensor(
                self.labels[idx], dtype=torch.float32)

        return result


def collate_fn(batch):
    """Custom collate for variable-size molecular graphs."""
    from torch_geometric.data import Batch

    result = {}
    for key in batch[0].keys():
        values = [b[key] for b in batch]
        if key in ("atomic", "tree", "fragment"):
            result[key] = Batch.from_data_list(values)
        elif key == "mph_maps":
            result[key] = torch.stack(values, dim=0)
        elif key == "label":
            result[key] = torch.stack(values)
    return result


def load_moleculenet_dataset(dataset_name, data_dir="data/moleculenet"):
    """Load a MoleculeNet dataset."""
    import deepchem as dc
    tasks, (train, valid, test), _ = dc.molnet.load_moleculenet(
        dataset_name, data_dir, reload=True)

    def extract(ds):
        return ds.ids.tolist(), ds.y

    train_s, train_l = extract(train)
    valid_s, valid_l = extract(valid)
    test_s, test_l = extract(test)
    return (train_s, train_l), (valid_s, valid_l), (test_s, test_l)
