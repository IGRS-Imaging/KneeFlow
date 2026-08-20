# KneeFlow — Femur

Conditional OT flow-matching model reconstructing a full femur point cloud
(8192 pts) from condylar boundary points + 9 anatomical landmarks.

## Files

- `net.py` — BoundEnc (Point Transformer) + LMEnc + VFNet architecture
- `data.py` — data loading, normalisation, train/test split
- `cfg.py` — paths, hyperparameters
- `train.py` — training entry point
- `infer.py` — single-bone inference from boundary + landmarks
- `eval_all_test.py` — batch inference + metrics (ASD/HD95/NSD) on the test set
- `eval_solver.py` — Euler vs Heun solver step-count ablation
- `export_test_gt.py` — exports ground-truth point clouds for the test set
- `visualize_errors.py` — per-point error colour-map PLYs (MeshLab/CloudCompare)
- `register.py`, `register_and_infer.py`, `prepare_og.py`, `convert_test_bones.py`,
  `refine.py` — data preparation / registration utilities
- `checkpoints/best_asd.pt` — checkpoint used for the paper's reported femur results
- `checkpoints/norm_stats.json` — normalisation stats matching that checkpoint

## Test set

Training uses a **fixed** 29-bone held-out test set (`TEST_BONES` in `data.py`),
not a random split — the same 29 bones are excluded from training every run so
results are reproducible against the paper's Table 1 numbers (see
`selected_29_dice78_82.csv`).

## Usage

```
python train.py                       # train from scratch
python train.py --resume checkpoints/latest.pt   # resume

python infer.py --bound boundary.ply --lm landmarks.csv \
    --ckpt checkpoints/best_asd.pt --out out.ply

python eval_all_test.py --ckpt checkpoints/best_asd.pt
```
