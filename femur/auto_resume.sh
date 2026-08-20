#!/bin/bash
# auto_resume.sh
# Automatically restarts training whenever it crashes (segfault, OOM, etc.)
# Usage:  bash auto_resume.sh
# Logs each restart with timestamp to auto_resume.log

CKPT="/home/imaging/Desktop/Ragavan/9_landmarks_output/checkpoints/latest.pt"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$SCRIPT_DIR/auto_resume.log"
MAX_RESTARTS=50
RESTART_DELAY=5   # seconds to wait before restarting

echo "=======================================" | tee -a "$LOG"
echo "  AUTO-RESUME started  $(date)"          | tee -a "$LOG"
echo "  Script dir : $SCRIPT_DIR"              | tee -a "$LOG"
echo "  Checkpoint : $CKPT"                    | tee -a "$LOG"
echo "=======================================" | tee -a "$LOG"

count=0
while [ $count -lt $MAX_RESTARTS ]; do
    echo ""                                                        | tee -a "$LOG"
    echo "--- Run $((count+1)) / $MAX_RESTARTS  $(date) ---"      | tee -a "$LOG"

    if [ -f "$CKPT" ]; then
        echo "  Resuming from: $CKPT"  | tee -a "$LOG"
        python "$SCRIPT_DIR/train.py" --resume "$CKPT"
    else
        echo "  No checkpoint found — starting from scratch."  | tee -a "$LOG"
        python "$SCRIPT_DIR/train.py"
    fi

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo ""                                           | tee -a "$LOG"
        echo "  Training finished cleanly (exit 0)."     | tee -a "$LOG"
        echo "  $(date)"                                  | tee -a "$LOG"
        break
    else
        echo ""                                                          | tee -a "$LOG"
        echo "  !! Process exited with code $EXIT_CODE (crash/segfault)" | tee -a "$LOG"
        echo "  Waiting ${RESTART_DELAY}s then restarting..."            | tee -a "$LOG"
        sleep $RESTART_DELAY
    fi

    count=$((count + 1))
done

if [ $count -ge $MAX_RESTARTS ]; then
    echo ""                                                        | tee -a "$LOG"
    echo "  !! Reached max restarts ($MAX_RESTARTS). Stopping."   | tee -a "$LOG"
    echo "  $(date)"                                               | tee -a "$LOG"
fi
