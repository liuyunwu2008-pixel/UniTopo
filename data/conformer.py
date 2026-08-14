"""3D conformer generation and caching for MPH encoding.

Generates low-energy 3D conformers using RDKit ETKDGv3 and caches
(coordinates, charges) to disk for fast reload across epochs.
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np


class ConformerCache:
    """Disk-backed cache for 3D conformer coordinates and partial charges.

    Conformer generation (ETKDG + MMFF optimization) is the most expensive
    data-preparation step.  Caching avoids recomputation across epochs.

    Auto-saves to disk every ``save_interval`` new entries so that multiple
    FinetuneDataset instances (grid search, val, test) share the same cache.
    """

    def __init__(self, cache_path: Optional[str] = None, save_interval: int = 25):
        self.cache_path = cache_path
        self.save_interval = save_interval
        self._mem_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._unsaved = 0  # new entries since last save
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self._mem_cache = pickle.load(f)

    def get(self, smiles: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return self._mem_cache.get(smiles)

    def put(self, smiles: str, coords: np.ndarray, charges: np.ndarray):
        self._mem_cache[smiles] = (coords, charges)
        self._unsaved += 1
        if self._unsaved >= self.save_interval and self.cache_path:
            self.save()

    def save(self):
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            tmp = self.cache_path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(self._mem_cache, f)
            os.replace(tmp, self.cache_path)  # atomic rename
            self._unsaved = 0

    def __len__(self):
        return len(self._mem_cache)

    def __contains__(self, smiles: str) -> bool:
        return smiles in self._mem_cache


def generate_conformer_batch(
    smiles_list: List[str],
    cache: Optional[ConformerCache] = None,
    max_failures: int = 5,
) -> Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
    """Generate 3D conformers for a batch of SMILES with caching.

    Args:
        smiles_list: list of SMILES strings.
        cache: optional ConformerCache for disk persistence.
        max_failures: silently return None for molecules that fail this many times.

    Returns:
        coords_list: list of (N_i, 3) arrays (or None for failures).
        charges_list: list of (N_i,) arrays (or None for failures).
    """
    from models.mph_encoder import generate_conformer

    coords_list = []
    charges_list = []

    for smi in smiles_list:
        # Check cache first
        if cache and smi in cache:
            c, q = cache.get(smi)
            coords_list.append(c)
            charges_list.append(q)
            continue

        result = generate_conformer(smi, max_attempts=max_failures)
        if result is not None:
            coords, charges = result
            coords_list.append(coords)
            charges_list.append(charges)
            if cache:
                cache.put(smi, coords, charges)
        else:
            coords_list.append(None)
            charges_list.append(None)

    return coords_list, charges_list
