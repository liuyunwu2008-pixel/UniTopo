"""Automated ablation experiment runner for GIN ↔ MPH pretraining.

Runs systematic ablation experiments across all 7 groups defined in
architecture_methods.md. Each experiment:
  1. Pre-trains a GIN encoder with specific ablation settings
  2. Fine-tunes on all available MoleculeNet datasets
  3. Aggregates results into ablation_results.json

Usage:
    # Run all ablations (full suite, very long)
    python tools/run_ablations.py --smiles data/zinc_chembl_selected.csv \
        --data_dir data/moleculenet --device cuda

    # Run only top-4 priority experiments
    python tools/run_ablations.py --smiles data/zinc_chembl_selected.csv \
        --data_dir data/moleculenet --device cuda --preset top4

    # Dry-run: print commands without executing
    python tools/run_ablations.py --smiles data/zinc_chembl_selected.csv --dry_run

    # Resume from a partially completed run
    python tools/run_ablations.py --smiles data/zinc_chembl_selected.csv --resume
"""

import os
import sys
import json
import argparse
import subprocess
import logging
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = "./ablation_results"

# ── Dataset lists ──
CLASSIFICATION_TASKS = ["BBBP", "BACE", "ClinTox", "SIDER", "Tox21"]
REGRESSION_TASKS = ["Lipo", "QM7"]

# ── Ablation experiment definitions ──
# Each experiment defines pretraining overrides + finetuning overrides

ABLATION_GROUPS = {
    # ═══════════════════════════════════════════════════════════════
    # Group A: Modality ablation (5 experiments)
    # ═══════════════════════════════════════════════════════════════
    "A1_scratch": {
        "desc": "Scratch baseline (no pretraining)",
        "pretrain": None,  # skip pretraining
        "finetune": {"scratch": True},
    },
    "A2_full": {
        "desc": "Full method (GIN ↔ MPH, Chem-Aware NCE)",
        "pretrain": {"ablation_name": "A2_full"},
        "finetune": {},
    },
    "A3_standard_nce": {
        "desc": "Standard NCE (no Tanimoto weighting)",
        "pretrain": {"ablation_name": "A3_standard_nce", "no_chem_aware": True},
        "finetune": {},
    },
    "A4_mphepsilon_only": {
        "desc": "MPH epsilon-only (single-parameter persistence)",
        "pretrain": {"ablation_name": "A4_single_param", "mph_single_param": True},
        "finetune": {},
    },
    "A5_extended_eps": {
        "desc": "MPH extended epsilon range (0.3-12.0 Angstrom)",
        "pretrain": {"ablation_name": "A5_ext_eps", "mph_eps_max": 12.0},
        "finetune": {},
    },

    # ═══════════════════════════════════════════════════════════════
    # Group B: Encoder depth ablation (5 experiments)
    # ═══════════════════════════════════════════════════════════════
    "B1_gin6_tf0": {
        "desc": "Deep GIN-only (6 GIN layers, 0 transformer)",
        "pretrain": {"ablation_name": "B1_gin6_tf0", "gin_layers": 6, "transformer_layers": 0},
        "finetune": {"gin_layers": 6, "transformer_layers": 0},
    },
    "B2_gin4_tf0": {
        "desc": "Shallow GIN-only (4 GIN layers, 0 transformer)",
        "pretrain": {"ablation_name": "B2_gin4_tf0", "gin_layers": 4, "transformer_layers": 0},
        "finetune": {"gin_layers": 4, "transformer_layers": 0},
    },
    "B3_gin4_tf1": {
        "desc": "GIN + 1 transformer layer",
        "pretrain": {"ablation_name": "B3_gin4_tf1", "gin_layers": 4, "transformer_layers": 1},
        "finetune": {"gin_layers": 4, "transformer_layers": 1},
    },
    "B4_gin4_tf4": {
        "desc": "GIN + 4 transformer layers (baseline)",
        "pretrain": {"ablation_name": "B4_gin4_tf4", "gin_layers": 4, "transformer_layers": 4},
        "finetune": {"gin_layers": 4, "transformer_layers": 4},
    },
    "B5_gin4_tf8": {
        "desc": "GIN + 8 transformer layers (deep transformer)",
        "pretrain": {"ablation_name": "B5_gin4_tf8", "gin_layers": 4, "transformer_layers": 8},
        "finetune": {"gin_layers": 4, "transformer_layers": 8, "hidden_dim": 512},
    },

    # ═══════════════════════════════════════════════════════════════
    # Group C: MPH resolution ablation (5 experiments)
    # ═══════════════════════════════════════════════════════════════
    "C1_coarse16": {
        "desc": "Coarse MPH grid (16x16)",
        "pretrain": {"ablation_name": "C1_coarse16", "mph_grid_size": 16},
        "finetune": {},
    },
    "C2_medium32": {
        "desc": "Medium MPH grid (32x32, baseline)",
        "pretrain": {"ablation_name": "C2_medium32", "mph_grid_size": 32},
        "finetune": {},
    },
    "C3_fine64": {
        "desc": "Fine MPH grid (64x64)",
        "pretrain": {"ablation_name": "C3_fine64", "mph_grid_size": 64},
        "finetune": {},
    },
    "C4_extended_eps": {
        "desc": "Extended epsilon range (0.3–12.0 Å)",
        "pretrain": {"ablation_name": "C4_ext_eps", "mph_eps_max": 12.0},
        "finetune": {},
    },
    "C5_single_param": {
        "desc": "Single-parameter: epsilon only, no delta",
        "pretrain": {"ablation_name": "C5_single", "mph_single_param": True},
        "finetune": {},
    },

    # ═══════════════════════════════════════════════════════════════
    # Group D: Data scale ablation (5 experiments)
    # ═══════════════════════════════════════════════════════════════
    "D1_10k": {
        "desc": "10K molecules",
        "pretrain": {"ablation_name": "D1_10k", "max_smiles": 10000, "epochs": 30, "early_stop_patience": 8},
        "finetune": {},
    },
    "D2_30k": {
        "desc": "30K molecules",
        "pretrain": {"ablation_name": "D2_30k", "max_smiles": 30000, "epochs": 40, "early_stop_patience": 10},
        "finetune": {},
    },
    "D3_100k": {
        "desc": "100K molecules (baseline)",
        "pretrain": {"ablation_name": "D3_100k", "max_smiles": 100000},
        "finetune": {},
    },
    "D4_300k": {
        "desc": "300K molecules",
        "pretrain": {"ablation_name": "D4_300k", "max_smiles": 300000},
        "finetune": {},
    },
    "D5_selected": {
        "desc": "Tanimoto-selected molecules (~450K)",
        "pretrain": {"ablation_name": "D5_selected", "smiles_file": "data/pretrain_selected.csv",
                      "max_smiles": None},  # use all molecules in the file
        "finetune": {},
    },

    # ═══════════════════════════════════════════════════════════════
    # Group E: Temperature & hidden_dim ablation (5 experiments)
    # ═══════════════════════════════════════════════════════════════
    "E1_cold_tau04": {
        "desc": "Cold contrast (tau=0.04)",
        "pretrain": {"ablation_name": "E1_cold", "tau": 0.04},
        "finetune": {},
    },
    "E2_baseline_tau07": {
        "desc": "Baseline temperature (tau=0.07)",
        "pretrain": {"ablation_name": "E2_baseline"},
        "finetune": {},
    },
    "E3_warm_tau15": {
        "desc": "Warm contrast (tau=0.15)",
        "pretrain": {"ablation_name": "E3_warm", "tau": 0.15},
        "finetune": {},
    },
    "E4_reduced_dim256": {
        "desc": "Reduced hidden dim (256)",
        "pretrain": {"ablation_name": "E4_dim256", "hidden_dim": 256, "batch_size": 256},
        "finetune": {"hidden_dim": 256},
    },
    "E5_mini_dim128": {
        "desc": "Minimal model (dim=128, gin=3, tf=2)",
        "pretrain": {"ablation_name": "E5_mini", "hidden_dim": 128, "gin_layers": 3,
                      "transformer_layers": 2, "batch_size": 512, "epochs": 30,
                      "num_heads": 4},
        "finetune": {"hidden_dim": 128, "gin_layers": 3, "transformer_layers": 2},
    },

    # ═══════════════════════════════════════════════════════════════
    # Group F: Finetuning strategy ablation (5 experiments)
    # ═══════════════════════════════════════════════════════════════
    "F1_frozen_mlp": {
        "desc": "Frozen encoder + plain MLP predictor (baseline)",
        "pretrain": {"ablation_name": "F_base"},
        "finetune": {"predictor": "mlp"},
    },
    "F2_frozen_single_moe": {
        "desc": "Frozen encoder + single-token MoE (expected harmful)",
        "pretrain": {"ablation_name": "F_base"},
        "finetune": {"predictor": "moe"},  # uses old MoEPredictor
        "note": "Single-token MoE degrades to 4x over-parameterized MLP"
    },
    "F3_frozen_multihead_moe": {
        "desc": "Frozen encoder + Multi-Head MoE (9 tokens/mol: 1 global + 8 heads)",
        "pretrain": {"ablation_name": "F_base"},
        "finetune": {"predictor": "moe"},  # now uses MultiHeadMoEPredictor
    },
    "F4_full_finetune_mlp": {
        "desc": "Full finetune (unfrozen encoder) + MLP",
        "pretrain": {"ablation_name": "F_base"},
        "finetune": {"predictor": "mlp", "unfreeze": True},
    },
    "F5_physchem": {
        "desc": "Frozen MLP + PhysChem descriptors (regression only)",
        "pretrain": {"ablation_name": "F_base"},
        "finetune": {"predictor": "mlp", "no_physchem": False},
    },
}

# ── Top-4 priority experiments (for quick runs) ──
TOP4_PRESET = ["A1_scratch", "A2_full", "A4_mphepsilon_only", "A3_standard_nce"]


def build_pretrain_cmd(smiles_file, ablation, checkpoint_base, override):
    """Build pretraining command for an ablation experiment."""
    if ablation is None:
        return None

    name = override.get("ablation_name", "unnamed")
    ckpt_dir = os.path.join(checkpoint_base, name)

    cmd = [
        "python3", "-u", "pretrain_mmgmpp.py",
        "--smiles", override.get("smiles_file", smiles_file),
        "--checkpoint_dir", ckpt_dir,
        "--ablation_name", name,
    ]

    # Numeric overrides
    for key in ["max_smiles", "hidden_dim", "gin_layers", "transformer_layers",
                "mph_grid_size", "mph_eps_max", "tau", "batch_size", "epochs", "num_heads"]:
        if key in override:
            cmd.append(f"--{key}")
            cmd.append(str(override[key]))

    # Boolean flags
    if override.get("no_chem_aware"):
        cmd.append("--no_chem_aware")
    if override.get("mph_single_param"):
        cmd.append("--mph_single_param")

    return cmd, ckpt_dir


def build_finetune_cmds(checkpoint_dir, data_dir, override, scratch=False):
    """Build finetuning commands for all datasets."""
    cmds = []
    all_datasets = CLASSIFICATION_TASKS + REGRESSION_TASKS

    for dataset in all_datasets:
        if scratch:
            cmd = [
                "python3", "-u", "finetune.py",
                "--dataset", dataset,
                "--data_dir", data_dir,
                "--scratch",
            ]
        else:
            best_pt = os.path.join(checkpoint_dir, "best_model.pt")
            if not os.path.exists(best_pt):
                logger.warning("  No checkpoint at %s — skipping %s", best_pt, dataset)
                continue
            cmd = [
                "python3", "-u", "finetune.py",
                "--checkpoint", best_pt,
                "--dataset", dataset,
                "--data_dir", data_dir,
            ]

        # Predictor type
        if override.get("predictor"):
            cmd.extend(["--predictor", override["predictor"]])
        if override.get("unfreeze"):
            cmd.append("--unfreeze")
        if override.get("no_physchem"):
            cmd.append("--no_physchem")
        if override.get("hidden_dim"):
            cmd.extend(["--hidden_dim", str(override["hidden_dim"])])
        if override.get("gin_layers"):
            cmd.extend(["--gin_layers", str(override["gin_layers"])])
        if override.get("transformer_layers"):
            cmd.extend(["--transformer_layers", str(override["transformer_layers"])])

        cmds.append(cmd)

    return cmds


def run_cmd(cmd, log_file, dry_run=False):
    """Execute a command and log output. Returns (success, elapsed_seconds)."""
    logger.info("  RUN: %s", " ".join(cmd))
    if dry_run:
        logger.info("  [DRY RUN] skipped")
        return True, 0

    start = datetime.now()
    try:
        with open(log_file, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"COMMAND: {' '.join(cmd)}\n")
            f.write(f"START:   {start.isoformat()}\n")
            f.write(f"{'='*80}\n")
            f.flush()
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                    cwd=os.path.dirname(os.path.dirname(__file__)),
                                    timeout=86400)  # 24h max
        elapsed = (datetime.now() - start).total_seconds()
        success = result.returncode == 0
        logger.info("  %s (%.0fs, exit=%d)", "OK" if success else "FAILED",
                    elapsed, result.returncode)
        return success, elapsed
    except subprocess.TimeoutExpired:
        logger.error("  TIMEOUT after 24h")
        return False, 86400
    except Exception as e:
        logger.error("  ERROR: %s", e)
        return False, 0


def load_results():
    """Load existing results to support resume."""
    results_file = os.path.join(RESULTS_DIR, "ablation_results.json")
    if os.path.exists(results_file):
        with open(results_file) as f:
            return json.load(f)
    return {}


def save_results(results):
    """Save results atomically."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_file = os.path.join(RESULTS_DIR, "ablation_results.json")
    tmp = results_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, default=str)
    os.replace(tmp, results_file)


def main():
    parser = argparse.ArgumentParser(description="Ablation experiment runner")
    parser.add_argument("--smiles", type=str, default="data/zinc_chembl_selected.csv",
                        help="Pretraining SMILES file")
    parser.add_argument("--data_dir", type=str, default="./data/moleculenet",
                        help="MoleculeNet CSV directory")
    parser.add_argument("--checkpoint_base", type=str, default="./ablation_checkpoints",
                        help="Base dir for pretraining checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dry_run", action="store_true", help="Print commands, don't execute")
    parser.add_argument("--resume", action="store_true", help="Skip completed experiments")
    parser.add_argument("--preset", type=str, default="all",
                        choices=["all", "top4", "group_a", "group_b", "group_c",
                                 "group_d", "group_e", "group_f"],
                        help="Which ablation groups to run")
    parser.add_argument("--skip_pretrain", action="store_true",
                        help="Skip pretraining (assume checkpoints exist)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(args.checkpoint_base, exist_ok=True)

    # Select experiments
    if args.preset == "top4":
        exp_names = TOP4_PRESET
    elif args.preset.startswith("group_"):
        group_letter = args.preset[-1].upper()
        exp_names = [k for k in ABLATION_GROUPS if k.startswith(group_letter)]
    else:
        exp_names = list(ABLATION_GROUPS.keys())

    results = load_results() if args.resume else {}

    logger.info("=" * 70)
    logger.info("Ablation Runner: %d experiments, %s",
                len(exp_names), args.preset)
    logger.info("=" * 70)

    for exp_name in exp_names:
        exp = ABLATION_GROUPS[exp_name]
        desc = exp["desc"]

        # Skip if already completed
        if args.resume and exp_name in results:
            logger.info("[%s] %s — SKIPPED (already done)", exp_name, desc)
            continue

        logger.info("[%s] %s", exp_name, desc)
        log_file = os.path.join(RESULTS_DIR, f"{exp_name}.log")

        # ── Phase 1: Pretraining ──
        pretrain_cfg = exp.get("pretrain")
        finetune_cfg = exp.get("finetune", {})

        if pretrain_cfg is not None and not args.skip_pretrain:
            cmd, ckpt_dir = build_pretrain_cmd(
                args.smiles, pretrain_cfg, args.checkpoint_base, pretrain_cfg
            )
            if cmd:
                success, elapsed = run_cmd(cmd, log_file, args.dry_run)
                if not success and not args.dry_run:
                    logger.error("  PRETRAIN FAILED — skipping finetune")
                    results[exp_name] = {"status": "pretrain_failed", "desc": desc}
                    save_results(results)
                    continue

        # ── Phase 2: Finetuning ──
        if pretrain_cfg is None:
            ckpt_dir = None  # scratch mode
        else:
            ckpt_dir = os.path.join(args.checkpoint_base,
                                    pretrain_cfg.get("ablation_name", exp_name))

        finetune_cmds = build_finetune_cmds(
            ckpt_dir, args.data_dir, finetune_cfg,
            scratch=(pretrain_cfg is None),
        )

        dataset_results = {}
        for cmd in finetune_cmds:
            ds_name = [a for i, a in enumerate(cmd) if cmd[i-1] == "--dataset"][0]
            ds_log = os.path.join(RESULTS_DIR, f"{exp_name}_{ds_name}.log")
            success, elapsed = run_cmd(cmd, ds_log, args.dry_run)
            dataset_results[ds_name] = {"success": success, "elapsed_s": elapsed}

        results[exp_name] = {
            "desc": desc,
            "status": "completed" if all(
                v["success"] for v in dataset_results.values()
            ) else "partial",
            "pretrain": pretrain_cfg,
            "finetune": finetune_cfg,
            "datasets": dataset_results,
        }
        save_results(results)

    # ── Final summary ──
    logger.info("\n" + "=" * 70)
    logger.info("ABLATION SUMMARY")
    logger.info("=" * 70)
    for exp_name in exp_names:
        r = results.get(exp_name, {})
        status = r.get("status", "not_started")
        ds_count = len(r.get("datasets", {}))
        ds_ok = sum(1 for v in r.get("datasets", {}).values() if v.get("success"))
        logger.info("  %-20s  %-15s  %s/%s datasets  %s",
                    exp_name, status, ds_ok, ds_count, r.get("desc", ""))
    logger.info(f"Results saved to {RESULTS_DIR}/ablation_results.json")


if __name__ == "__main__":
    main()
