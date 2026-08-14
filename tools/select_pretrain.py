"""Select pretraining molecules similar to downstream tasks.

Strategy: Tanimoto-based filtering — for each downstream molecule, select
top-K most similar ZINC/ChemBL molecules. Creates a pretraining set biased
toward the downstream chemical space.

Usage:
    python tools/select_pretrain.py --target_data data/moleculenet \
        --source data/zinc_chembl_merged.csv --k 20 --output data/pretrain_selected.csv
"""

import argparse, random, os
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm


def compute_fingerprints(smiles_list, radius=2, nbits=1024):
    """Batch compute Morgan fingerprints."""
    fps = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits))
            valid_idx.append(i)
    return fps, valid_idx


def select_similar(source_smiles, target_smiles_list, top_k=50000, threshold=0.35):
    """Select top-K source molecules by max Tanimoto to any downstream molecule."""
    # Compute target fingerprints
    print("Computing target fingerprints...")
    target_all = []
    for smis in target_smiles_list:
        target_all.extend(smis)
    target_sample = random.sample(target_all, min(len(target_all), 3000))
    target_fps, _ = compute_fingerprints(target_sample)
    print(f"  {len(target_fps)} downstream molecules")

    # Compute source fingerprints in chunks
    print("Computing source similarities (chunked)...")
    n_source = min(len(source_smiles), 300000)
    source_sample = random.sample(source_smiles, n_source)
    max_sims = np.zeros(n_source)

    chunk = 1000
    for i in tqdm(range(0, n_source, chunk)):
        s_chunk_smi = source_sample[i:i + chunk]
        s_chunk_fps, _ = compute_fingerprints(s_chunk_smi)
        for j, s_fp in enumerate(s_chunk_fps):
            sims = [DataStructs.TanimotoSimilarity(s_fp, t_fp) for t_fp in target_fps[:200]]
            max_sims[i + j] = max(sims)

    # Select top_k by max similarity
    order = np.argsort(max_sims)[::-1]
    selected = []
    for j in order[:top_k]:
        if max_sims[j] >= threshold:
            selected.append(source_sample[j])
        if len(selected) >= top_k:
            break

    print(f"Selected {len(selected)} molecules (threshold={threshold})")
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="ZINC/ChemBL CSV")
    parser.add_argument("--target_data", required=True, help="Moleculenet dir")
    parser.add_argument("--k", type=int, default=50000)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load target SMILES from all downstream datasets
    target_smiles = []
    datasets = ["BBBP", "BACE", "ClinTox", "SIDER", "Tox21", "ESOL", "FreeSolv", "Lipo"]
    for ds in datasets:
        path = os.path.join(args.target_data, f"{ds}.csv")
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
            except UnicodeDecodeError:
                df = pd.read_csv(path, compression="gzip")
            smi_col = "smiles" if "smiles" in df.columns else df.columns[0]
            smis = df[smi_col].dropna().tolist()
            target_smiles.append(smis)
            print(f"{ds}: {len(smis)} molecules")

    # Load source
    print(f"\nLoading source from {args.source}...")
    source_df = pd.read_csv(args.source, nrows=500000)
    source_smiles = source_df["smiles"].dropna().tolist()
    print(f"Source: {len(source_smiles)} molecules")

    # Select similar molecules
    selected = select_similar(source_smiles, target_smiles, top_k=args.k, threshold=args.threshold)

    # Also add some random molecules for diversity (20% of selected)
    n_extra = int(len(selected) * 0.2)
    remaining = list(set(range(len(source_smiles))) - set(range(len(selected))))
    extra_idx = random.sample(remaining, min(n_extra, len(remaining)))
    for i in extra_idx:
        selected.append(source_smiles[i])

    # Save
    out_df = pd.DataFrame({"smiles": selected})
    out_df.to_csv(args.output, index=False)
    print(f"\nSaved {len(selected)} molecules to {args.output}")

    # Statistics
    print(f"\nStats:")
    print(f"  Selected: {len(selected)}")
    print(f"  Target coverage: {len(set(selected)) / len(source_smiles) * 100:.1f}% of source")
    print(f"  Suggested pretrain: --smiles {args.output} --max_smiles {len(selected)}")
