#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# Evaluation pipeline: run after ep700.pt (or best_asd.pt) is available
# Copy this script to your GPU machine alongside the other .py files
#
# Usage:
#   bash run_eval_ep700.sh              # uses best_asd.pt
#   bash run_eval_ep700.sh ep700        # uses ep700.pt
#   bash run_eval_ep700.sh ep800        # uses ep800.pt
# ────────────────────────────────────────────────────────────────────────────

CKPT_TAG=${1:-best_asd}
BASE="/home/imaging/Desktop/Ragavan"
CKPT="$BASE/9L_working_output/checkpoints/${CKPT_TAG}.pt"

echo "=================================================="
echo " Evaluating checkpoint: $CKPT"
echo "=================================================="

# Step 1: Make sure GT point clouds exist
# (skip if test_bones_gt/ already populated)
GT_DIR="$BASE/9L_working_output/test_bones_gt"
N_GT=$(ls "$GT_DIR"/*.ply 2>/dev/null | wc -l)
if [ "$N_GT" -lt 29 ]; then
    echo ""
    echo "Step 1: Exporting GT point clouds (found $N_GT/29)..."
    python3 export_test_gt.py
else
    echo "Step 1: GT point clouds OK ($N_GT files found)"
fi

# Step 2: Run inference + metrics on all 29 test bones
echo ""
echo "Step 2: Running inference + evaluation..."
python3 eval_all_test.py --ckpt "$CKPT" 2>&1 | tee "$BASE/9L_working_output/eval_${CKPT_TAG}_log.txt"

# Step 3: Generate error-colored PLY files for paper figures
echo ""
echo "Step 3: Generating error visualization PLYs..."
python3 visualize_errors.py --batch 2>&1 | tee -a "$BASE/9L_working_output/eval_${CKPT_TAG}_log.txt"

echo ""
echo "=================================================="
echo " Done! Results saved to:"
echo "   $BASE/9L_working_output/eval_ep*_results.csv    (metrics table)"
echo "   $BASE/9L_working_output/test_infer/              (generated PLYs)"
echo "   $BASE/9L_working_output/error_viz/               (error-colored PLYs)"
echo "   $BASE/9L_working_output/eval_${CKPT_TAG}_log.txt (full log)"
echo ""
echo " Open error_viz/*.ply in MeshLab for paper figures."
echo " Color key: GREEN < 2mm | YELLOW 2-4mm | ORANGE 4-6mm | RED > 6mm"
echo "=================================================="
