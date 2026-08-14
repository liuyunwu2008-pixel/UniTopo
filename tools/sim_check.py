import pandas as pd, numpy as np, random, sys
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

pretrain = pd.read_csv(sys.argv[1], nrows=5000)
pretrain_smi = pretrain["smiles"].dropna().tolist()[:5000]

datasets = {}
for ds, path in [("BBBP","BBBP.csv"),("Tox21","Tox21.csv"),("ClinTox","ClinTox.csv"),("BACE","BACE.csv"),("SIDER","SIDER.csv")]:
    try: df = pd.read_csv(f"{sys.argv[2]}/{path}")
    except: df = pd.read_csv(f"{sys.argv[2]}/{path}", compression="gzip")
    datasets[ds] = df["smiles"].dropna().tolist()

pretrain_sample = random.sample(pretrain_smi, min(500, len(pretrain_smi)))
pretrain_fps = []
for smi in pretrain_sample:
    mol = Chem.MolFromSmiles(smi)
    if mol: pretrain_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))

print("Dataset    Mols   Tanimoto  MaxSim  >0.5   ScafOv")
for ds, smis in datasets.items():
    ds_sample = random.sample(smis, min(300, len(smis)))
    sims = []
    ds_scaf = set()
    for smi in ds_sample:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            max_sim = max(DataStructs.TanimotoSimilarity(fp, pf) for pf in pretrain_fps[:200])
            sims.append(max_sim)
            s = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            if s: ds_scaf.add(s)
    pt_scaf = set()
    for smi in pretrain_sample[:300]:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            s = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            if s: pt_scaf.add(s)
    ov = len(ds_scaf & pt_scaf) / max(len(ds_scaf),1) * 100
    gt05 = sum(1 for s in sims if s>0.5)/max(len(sims),1)*100
    print("{:<12} {:>4}  {:>8.4f}  {:>6.4f}  {:>5.1f}%  {:>5.1f}%".format(ds, len(smis), np.mean(sims), np.max(sims), gt05, ov))
