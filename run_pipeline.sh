#!/bin/bash
# Post-reboot: run MMG-MPP pre-training + fine-tuning
# Usage: bash run_pipeline.sh
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmg_mpp

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

SMILES_DATA="${1:-$HOME/cot_moe_project/data/chemberta_10m.smi}"
CHECKPOINT_DIR="$HOME/mmg-mpp/checkpoints"
DATA_DIR="$HOME/cot_moe_project/data"

log "=== MMG-MPP Full Pipeline ==="

# Verify GPU
python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE!'; print(f'GPU OK: {torch.cuda.get_device_name(0)}')"

# ── Stage 1: Pre-training (100 epochs, ~48 hours) ──
log "Stage 1: Pre-training..."
if [ -f "$SMILES_DATA" ]; then
    python ~/mmg-mpp/pretrain.py \
        --smiles "$SMILES_DATA" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --device cuda
else
    log "ERROR: SMILES data not found at $SMILES_DATA"
    log "Downloading ChemBERTa-like SMILES from ZINC..."
    python -c "
import urllib.request, os
# Download ZINC 250k as fallback pre-training set
url = 'https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc/250k_rndm_zinc_drugs_clean_3.csv'
urllib.request.urlretrieve(url, '/tmp/zinc250k.csv')
print('Downloaded ZINC 250k')
"
    SMILES_DATA="/tmp/zinc250k.csv"
    python ~/mmg-mpp/pretrain.py \
        --smiles "$SMILES_DATA" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --device cuda
fi

log "Pre-training complete."

# ── Stage 2: Fine-tuning on all MoleculeNet benchmarks ──
log "Stage 2: Fine-tuning on 10 MoleculeNet datasets..."
python ~/mmg-mpp/run.py \
    --mode finetune \
    --checkpoint "$CHECKPOINT_DIR/best_model.pt" \
    --datasets all \
    --data_dir "$DATA_DIR" \
    --device cuda \
    --n_runs 5

log "=== Pipeline complete! ==="
log "Results saved to ~/mmg-mpp/results/benchmark_results.json"
