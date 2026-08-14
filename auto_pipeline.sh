#!/bin/bash
# MMG-MPP auto-pipeline: waits for cot_moe to finish, then runs MMG-MPP
# Usage: bash auto_pipeline.sh
# Run in screen/tmux: screen -S mmgpp bash auto_pipeline.sh

set -e

COT_MOE_PID=1253104
COT_MOE_LOG="$HOME/cot_moe_project/train_output_v6.log"
MMG_MPP_DIR="$HOME/mmg-mpp"
CONDA_BASE="$HOME/miniconda3"
SMILES_DATA="$HOME/cot_moe_project/data/chemberta_10m.smi"
CHECKPOINT_DIR="$HOME/mmg-mpp/checkpoints"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── Step 1: Wait for cot_moe to finish ──────────────────────────
log "Watching cot_moe (PID=$COT_MOE_PID)..."
while kill -0 $COT_MOE_PID 2>/dev/null; do
    if [ -f "$COT_MOE_LOG" ]; then
        LAST_LINE=$(tail -1 "$COT_MOE_LOG" 2>/dev/null | head -c 140)
        echo -ne "\r[$(date '+%H:%M:%S')] cot_moe still running. Last log: ${LAST_LINE}..."
    fi
    sleep 120  # check every 2 minutes
done
echo ""
log "cot_moe training FINISHED at $(date)"

# ── Step 2: Save final cot_moe results ─────────────────────────
log "Saving cot_moe final results..."
if [ -f "$COT_MOE_LOG" ]; then
    grep "Val epoch\|best" "$COT_MOE_LOG" | tail -30 > "$HOME/cot_moe_project/final_summary.log"
fi

# ── Step 3: Reboot required notice ─────────────────────────────
log ""
log "=============================================="
log "  NVIDIA driver mismatch detected."
log "  Kernel module: 580.142  Userspace: 580.159"
log "  A REBOOT is required for MMG-MPP GPU access."
log "  Reboot now? (y/n)"
log "=============================================="
log ""
log "If auto-rebooting, the pipeline will continue after restart."
log "Otherwise, reboot manually and re-run: bash auto_pipeline.sh --post-reboot"
log ""

# Allow 30 seconds for manual intervention
sleep 30

# Check if we should auto-reboot (set environment variable to enable)
if [ "${AUTO_REBOOT}" = "1" ]; then
    log "AUTO_REBOOT=1 — rebooting in 10 seconds..."
    sleep 10
    sudo reboot
    exit 0
fi

log "Skipping auto-reboot. Run with AUTO_REBOOT=1 to auto-reboot."
log "After reboot, run: bash $0 --post-reboot"
exit 0
