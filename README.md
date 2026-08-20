# KneeFlow

A conditional Optimal Transport Flow Matching (OT-FM) model for patient-specific
femur and tibia point-cloud reconstruction from sparse intra-operative inputs —
anatomical landmarks and condylar boundary points — for imageless total knee
arthroplasty (ITKA).

KneeFlow removes the need for dense surface-patch digitization used by existing
ITKA systems, cutting intra-operative acquisition from 4–7 minutes down to
under a minute per knee, while reconstructing each bone in ~7 seconds on a
single RTX 4090.

## Results

| Bone  | ASD (mm) | HD95 (mm) | NSD@2mm |
|-------|----------|-----------|---------|
| Femur (n=29) | 2.03 | 6.04 | 0.66 |
| Tibia (n=33) | 1.29 | 2.33 | 0.86 |

Outperforms GAN, Point Transformer, and DDPM baselines on both bones. See
`femur/` and `tibia/` for the per-bone training/eval code and reported
checkpoints.

## Repository layout

```
femur/
  net.py, data.py, cfg.py     — model, data pipeline, config
  train.py                    — training entry point
  infer.py                    — single-bone inference
  eval_all_test.py            — batch eval on the 29-bone held-out test set
  eval_solver.py               — Euler vs Heun solver ablation
  checkpoints/best_asd.pt     — checkpoint used for the reported femur results
tibia/
  net.py, data.py, cfg.py     — model, data pipeline, config
  train.py                    — training entry point
  infer.py                    — single-bone inference (handles L/R mirroring)
  eval_all_test.py            — batch eval on the held-out test set
  checkpoints/ep800.pt        — checkpoint used for the reported tibia results
```

## Checkpoints

Checkpoints are stored via [Git LFS](https://git-lfs.com/). Install it once,
then clone/pull normally:

```
git lfs install
git clone https://github.com/Ragavan09/KneeFlow.git
```

## Inference example

```
# Femur
cd femur
python infer.py --bound boundary.ply --lm landmarks.csv \
    --ckpt checkpoints/best_asd.pt --out out_femur.ply

# Tibia
cd tibia
python infer.py --bound boundary.ply --lm landmarks.csv \
    --ckpt checkpoints/ep800.pt --out out_tibia.ply --side R
```
