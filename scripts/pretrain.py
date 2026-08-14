"""MMG-MPP pre-training entry point."""

import argparse
import torch
from torch.utils.data import DataLoader
import pandas as pd

import sys
sys.path.insert(0, '..')

from config import ModelConfig, PretrainConfig
from models import MMGMPP
from pretrain import PretrainTrainer
from data.dataset import MoleculeDataset, collate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/pretrain_smiles.csv")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_3d", action="store_true")
    parser.add_argument("--save", type=str, default="pretrained_mmg_mpp.pt")
    args = parser.parse_args()

    # Config
    model_cfg = ModelConfig()
    model_cfg.use_3d_branch = not args.no_3d
    pretrain_cfg = PretrainConfig()
    pretrain_cfg.batch_size = args.batch_size
    pretrain_cfg.num_epochs = args.epochs
    pretrain_cfg.learning_rate = args.lr

    # Load SMILES data
    df = pd.read_csv(args.data)
    smiles_list = df["smiles"].tolist()

    # Split
    split = int(len(smiles_list) * 0.95)
    train_smiles = smiles_list[:split]
    val_smiles = smiles_list[split:]

    use_3d = not args.no_3d
    train_ds = MoleculeDataset(train_smiles, use_3d=use_3d)
    val_ds = MoleculeDataset(val_smiles, use_3d=use_3d)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size,
                        shuffle=False, collate_fn=collate_fn, num_workers=4)

    # Model
    model = MMGMPP(model_cfg, use_3d=use_3d)

    # Train
    trainer = PretrainTrainer(model, pretrain_cfg, device=args.device)
    trainer.train(train_dl, val_dl, save_path=args.save)


if __name__ == "__main__":
    main()
