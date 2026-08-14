"""ICML/NeurIPS-level benchmark: ablation + multi-dataset evaluation.

Usage:
    # Full model on all classification datasets
    python benchmark.py --checkpoint checkpoints_gin3d/best_model.pt --all_cls

    # Ablation: contrast-only pretraining
    python benchmark.py --ablation contrast_only --dataset BBBP

    # Multi-dataset sweep
    python benchmark.py --datasets BBBP Tox21 ClinTox SIDER BACE --runs 5
"""

import os, sys, argparse, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from finetune import finetune_gin3d
from config import FinetuneConfig


MOLECULENET_CLASSIFICATION = ["BBBP", "Tox21", "ToxCast", "SIDER", "ClinTox", "BACE", "HIV"]
MOLECULENET_REGRESSION = ["ESOL", "FreeSolv", "Lipo", "QM8", "QM9"]

# ── Ablation configurations (current CLI flags) ──
ABLATION_PRETRAIN_ARGS = {
    "full":              "",
    "no_chem_aware":     "--no_chem_aware",
    "mph_single_param":  "--mph_single_param",
    "extended_eps":      "--mph_eps_min 0.3 --mph_eps_max 12.0",
    "gin_only":          "--transformer_layers 0",
}


def run_benchmark(
    checkpoint_path: str,
    datasets: list,
    data_dir: str = "data/moleculenet",
    device: str = "cuda",
    n_runs: int = 5,
    finetune_config: FinetuneConfig = None,
):
    """Run finetuning on multiple datasets and collect results."""
    config = finetune_config or FinetuneConfig()
    results = {}

    for ds in datasets:
        logger.info("=" * 60)
        logger.info("Dataset: %s", ds)
        t0 = time.time()

        try:
            result = finetune_gin3d(
                checkpoint_path=checkpoint_path,
                dataset_name=ds,
                data_dir=data_dir,
                device=device,
                n_runs=n_runs,
                finetune_config=config,
            )
            elapsed = (time.time() - t0) / 60
            results[ds] = result
            logger.info("%s: %s (%.1f min)", ds,
                        {k: v for k, v in result.items() if k != "roc_auc_per_task"},
                        elapsed)
        except Exception as e:
            logger.error("%s: FAILED — %s", ds, e)
            results[ds] = {"error": str(e)}

    # ── Summary table ──
    print("\n" + "=" * 70)
    print("Benchmark Summary")
    print("=" * 70)
    print(f"{'Dataset':<12} {'ROC-AUC':>10} {'Std':>8} {'Time(min)':>10}")
    print("-" * 45)
    for ds, r in results.items():
        if "roc_auc" in r:
            print(f"{ds:<12} {r['roc_auc']['mean']:>10.4f} {r['roc_auc']['std']:>8.4f}")
        elif "rmse" in r:
            print(f"{ds:<12} {r['rmse']['mean']:>10.4f} {r['rmse']['std']:>8.4f}")
        else:
            print(f"{ds:<12} {'ERROR':>10}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMG-MPP Benchmark Suite")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--datasets", nargs="+", default=["BBBP"])
    parser.add_argument("--all_cls", action="store_true", help="All 7 classification datasets")
    parser.add_argument("--all_reg", action="store_true", help="All 5 regression datasets")
    parser.add_argument("--ablation", choices=list(ABLATION_PRETRAIN_ARGS.keys()),
                        default="full", help="Ablation variant")
    parser.add_argument("--data_dir", type=str, default="data/moleculenet")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--freeze", type=int, default=1, help="freeze_encoders (1=True)")
    args = parser.parse_args()

    if args.all_cls:
        args.datasets = MOLECULENET_CLASSIFICATION
    elif args.all_reg:
        args.datasets = MOLECULENET_REGRESSION

    config = FinetuneConfig()
    config.freeze_encoders = bool(args.freeze)
    config.n_runs_override = args.runs

    logger.info("Ablation: %s", args.ablation)
    logger.info("Datasets: %s", args.datasets)
    logger.info("Runs per dataset: %d", args.runs)

    results = run_benchmark(
        checkpoint_path=args.checkpoint,
        datasets=args.datasets,
        data_dir=args.data_dir,
        device=args.device,
        n_runs=args.runs,
        finetune_config=config,
    )

    # Save results
    out_path = f"benchmark_{args.ablation}_{time.strftime('%Y%m%d_%H%M')}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_path)
