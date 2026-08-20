#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# auto_resume.sh  — Auto-restart Tibia OT-FM training on crash / OOM.
#
# Usage:
#   bash auto_resume.sh            # foreground
#   nohup bash auto_resume.sh &    # background (survives logout)
# ─────────────────────────────────────────────────────────────────────────────

CKPT_DIR="/home/imaging/Desktop/Ragavan_Tibia/output_Tibia/checkpoints"
LATEST="$CKPT_DIR/latest.pt"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train.py"
LOG="$SCRIPT_DIR/auto_resume_tibia.log"

MAX_RESTARTS=100
RESTART_DELAY=10

echo ""                                                      | tee -a "$LOG"
echo "============================================"          | tee -a "$LOG"
echo "  TIBIA OT-FM AUTO-RESUME  $(date)"                   | tee -a "$LOG"
echo "  Script : $TRAIN_SCRIPT"                             | tee -a "$LOG"
echo "  Ckpt   : $CKPT_DIR"                                 | tee -a "$LOG"
echo "============================================"          | tee -a "$LOG"

count=0
while [ $count -lt $MAX_RESTARTS ]; do
    echo ""                                                         | tee -a "$LOG"
    echo "--- Run $((count+1)) / $MAX_RESTARTS  $(date) ---"       | tee -a "$LOG"

    if [ -f "$LATEST" ]; then
        echo "  Resuming from: $LATEST"  | tee -a "$LOG"
        python3 "$TRAIN_SCRIPT" --resume "$LATEST" 2>&1 | tee -a "$LOG"
    else
        echo "  No checkpoint — starting fresh."  | tee -a "$LOG"
        python3 "$TRAIN_SCRIPT" 2>&1 | tee -a "$LOG"
    fi

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "  ✓ Training finished cleanly."             | tee -a "$LOG"
        echo "  Best ckpt: $CKPT_DIR/best_asd.pt"        | tee -a "$LOG"
        break
    else
        echo "  !! Crashed (exit $EXIT_CODE) — waiting ${RESTART_DELAY}s..." | tee -a "$LOG"
        sleep $RESTART_DELAY
    fi
    count=$((count + 1))
done
