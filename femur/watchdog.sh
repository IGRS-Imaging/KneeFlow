#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# WATCHDOG: Auto-resume training after segfault / crash / power cut
#
# Usage:
#   bash watchdog.sh                    # starts fresh (first time)
#   bash watchdog.sh ep700.pt           # resume from a specific checkpoint
#
# This script:
#   1. Runs train.py
#   2. If it crashes (segfault, OOM, any error), waits 60s then auto-resumes
#   3. Always resumes from latest.pt (saved every epoch, now atomic)
#   4. Logs each restart to watchdog.log
#   5. Stops automatically when training completes (exit 0)
#
# Start it in a tmux session so it keeps running after SSH disconnect:
#   tmux new -s train
#   bash watchdog.sh
#   Ctrl+B then D   (detach from tmux)
#   tmux attach -t train   (reattach later to check progress)
# ══════════════════════════════════════════════════════════════════════════════

BASE="/home/imaging/Desktop/Ragavan"
CKPT_DIR="$BASE/9L_working_output/checkpoints"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$BASE/9L_working_output/watchdog.log"
LATEST="$CKPT_DIR/latest.pt"

# First resume target: argument if provided, else latest.pt, else fresh start
FIRST_RESUME="${1:-}"

RESTART_DELAY=60   # seconds to wait after crash before restarting
MAX_RESTARTS=50    # safety limit — stops after 50 consecutive crashes

log() {
    local ts
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$ts] $1" | tee -a "$LOG"
}

cd "$SCRIPT_DIR" || { echo "Cannot cd to $SCRIPT_DIR"; exit 1; }

log "═══════════════════════════════════════════════════"
log " WATCHDOG STARTED"
log " Script dir : $SCRIPT_DIR"
log " CKPT dir   : $CKPT_DIR"
log " Max restarts: $MAX_RESTARTS"
log "═══════════════════════════════════════════════════"

restart_count=0
resume_arg="$FIRST_RESUME"

while true; do
    # Build the python command
    if [ -n "$resume_arg" ] && [ -f "$resume_arg" ]; then
        CMD="python3 train.py --resume $resume_arg"
        log "RUN #$((restart_count+1)): $CMD"
    elif [ -f "$LATEST" ]; then
        CMD="python3 train.py --resume $LATEST"
        log "RUN #$((restart_count+1)): resuming from latest.pt"
    else
        CMD="python3 train.py"
        log "RUN #$((restart_count+1)): fresh start (no checkpoint found)"
    fi

    # Run training
    start_time=$(date +%s)
    eval "$CMD"
    EXIT_CODE=$?
    end_time=$(date +%s)
    elapsed=$(( end_time - start_time ))

    if [ $EXIT_CODE -eq 0 ]; then
        log "═══════════════════════════════════════════════════"
        log " TRAINING COMPLETED SUCCESSFULLY (exit 0)"
        log " Total restarts: $restart_count"
        log "═══════════════════════════════════════════════════"
        exit 0
    fi

    restart_count=$((restart_count + 1))
    log "CRASH: exit code $EXIT_CODE  (ran for ${elapsed}s)"

    if [ $restart_count -ge $MAX_RESTARTS ]; then
        log "ERROR: reached MAX_RESTARTS=$MAX_RESTARTS — giving up."
        log "Check the training log for details."
        exit 1
    fi

    # After first run, always resume from latest.pt
    resume_arg="$LATEST"

    # Check if latest.pt is valid (not corrupted / zero bytes)
    if [ -f "$LATEST" ]; then
        SIZE=$(stat -c%s "$LATEST" 2>/dev/null || echo 0)
        if [ "$SIZE" -lt 1000 ]; then
            log "WARNING: latest.pt is too small ($SIZE bytes) — may be corrupted."
            # Try to find the most recent periodic checkpoint instead
            FALLBACK=$(ls -t "$CKPT_DIR"/ep*.pt 2>/dev/null | head -1)
            if [ -n "$FALLBACK" ]; then
                log "Falling back to: $FALLBACK"
                resume_arg="$FALLBACK"
            else
                log "No fallback checkpoint found — starting fresh."
                resume_arg=""
            fi
        else
            log "latest.pt OK ($SIZE bytes) — will resume from it."
        fi
    else
        log "WARNING: latest.pt not found — starting fresh."
        resume_arg=""
    fi

    log "Waiting ${RESTART_DELAY}s before restart #$((restart_count+1))..."
    sleep "$RESTART_DELAY"
done
