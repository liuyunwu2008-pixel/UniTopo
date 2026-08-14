"""MMG-MPP main entry point.

Usage:
  # Pre-training
  python run.py --mode pretrain --smiles data/chemberta_10m.smi

  # Fine-tuning on all benchmark datasets
  python run.py --mode finetune --checkpoint checkpoints/best_model.pt --datasets all

  # Fine-tuning on a single dataset
  python run.py --mode finetune --checkpoint checkpoints/best_model.pt --datasets BBBP
"""

import os
import sys
import argparse
import json
import logging
from typing import List, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

from config import FinetuneConfig, TASKS

ALL_DATASETS = TASKS["classification"] + TASKS["regression"]


def run_pretrain(args):
    """Stage 1: Pre-train GIN ↔ MPH on unlabeled data."""
    from pretrain_mmgmpp import pretrain_mmgmpp

    pretrain_mmgmpp(
        smiles_file=args.smiles,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        max_smiles=getattr(args, 'max_smiles', 100000),
    )


def run_finetune(args):
    """Stage 2: Fine-tune on downstream MPP benchmarks."""
    from finetune import finetune_gin3d

    datasets = ALL_DATASETS if args.datasets.lower() == "all" else args.datasets.split(",")

    all_results = {}
    for dataset in datasets:
        dataset = dataset.strip()
        logger.info("=" * 60)
        logger.info("Evaluating on %s", dataset)

        try:
            results = finetune_gin3d(
                checkpoint_path=args.checkpoint,
                dataset_name=dataset,
                data_dir=args.data_dir,
                device=args.device,
                n_runs=args.n_runs,
            )
            all_results[dataset] = results
        except FileNotFoundError as e:
            logger.warning("Skipping %s: %s", dataset, e)

    # Summary table
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    for dataset, results in all_results.items():
        if "roc_auc" in results:
            logger.info(
                "  %s: ROC-AUC = %.4f ± %.4f",
                dataset,
                results["roc_auc"]["mean"],
                results["roc_auc"]["std"],
            )
        elif "rmse" in results:
            logger.info(
                "  %s: RMSE = %.4f ± %.4f, MAE = %.4f ± %.4f",
                dataset,
                results["rmse"]["mean"],
                results["rmse"]["std"],
                results.get("mae", {}).get("mean", float("nan")),
                results.get("mae", {}).get("std", float("nan")),
            )

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("Results saved to results/benchmark_results.json")


def main():
    parser = argparse.ArgumentParser(
        description="MMG-MPP: Multi-view Molecular Graph representation for MPP"
    )
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["pretrain", "finetune"],
        help="Execution mode",
    )

    # Pre-training args
    parser.add_argument("--smiles", type=str, help="Path to SMILES file for pre-training")
    parser.add_argument("--max_smiles", type=int, default=100000, help="Max molecules for pretraining")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)

    # Fine-tuning args
    parser.add_argument("--checkpoint", type=str, help="Path to pre-trained checkpoint")
    parser.add_argument("--datasets", type=str, default="all", help="Dataset names or 'all'")
    parser.add_argument("--data_dir", type=str, default="./data/moleculenet")
    parser.add_argument("--n_runs", type=int, default=5)

    # Common
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    if args.mode == "pretrain":
        if not args.smiles:
            parser.error("--smiles is required for pretrain mode")
        run_pretrain(args)
    elif args.mode == "finetune":
        if not args.checkpoint:
            parser.error("--checkpoint is required for finetune mode")
        run_finetune(args)


if __name__ == "__main__":
    main()
