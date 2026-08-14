"""MMG-MPP fine-tuning entry point for MoleculeNet benchmarks."""

import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np

import sys
sys.path.insert(0, '..')

from config import ModelConfig, FinetuneConfig
from models import MMGMPP, MMGMPPFinetune
from finetune import FinetuneTrainer
from data.dataset import MoleculeDataset, collate_fn, load_moleculenet_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="MoleculeNet dataset name (e.g., BBBP, FreeSolv, QM7)")
    parser.add_argument("--pretrained", type=str, default="pretrained_mmg_mpp.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_3d", action="store_true")
    parser.add_argument("--save", type=str, default="finetuned_mmg_mpp.pt")
    args = parser.parse_args()

    # Config
    model_cfg = ModelConfig()
    model_cfg.use_3d_branch = not args.no_3d
    finetune_cfg = FinetuneConfig()
    finetune_cfg.batch_size = args.batch_size
    finetune_cfg.num_epochs = args.epochs
    finetune_cfg.learning_rate = args.lr

    # Load dataset
    (train_s, train_l), (val_s, val_l), (test_s, test_l) = \
        load_moleculenet_dataset(args.dataset)

    use_3d = not args.no_3d
    train_ds = MoleculeDataset(train_s, train_l, use_3d=use_3d)
    val_ds = MoleculeDataset(val_s, val_l, use_3d=use_3d)
    test_ds = MoleculeDataset(test_s, test_l, use_3d=use_3d)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size,
                        shuffle=False, collate_fn=collate_fn, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size,
                         shuffle=False, collate_fn=collate_fn, num_workers=2)

    # Determine task type
    classification_datasets = {"BBBP", "Tox21", "SIDER", "ClinTox", "BACE", "HIV"}
    task_type = "classification" if args.dataset in classification_datasets else "regression"

    # Load pre-trained model
    pretrained = MMGMPP(model_cfg, use_3d=use_3d)
    pretrained.load_state_dict(torch.load(args.pretrained, map_location=args.device))

    # Build fine-tune model
    num_tasks = train_l.shape[1] if train_l.ndim > 1 else 1
    model = MMGMPPFinetune(pretrained, model_cfg, num_tasks=num_tasks)

    # Train
    trainer = FinetuneTrainer(model, finetune_cfg, task_type=task_type,
                              device=args.device)
    test_metric, preds, labels = trainer.train(
        train_dl, val_dl, test_dl, save_path=args.save)

    print(f"\n{'='*50}")
    print(f"Dataset: {args.dataset} | Task: {task_type}")
    print(f"Test {'AUC' if task_type == 'classification' else 'RMSE'}: {test_metric:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
