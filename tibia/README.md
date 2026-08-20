# KneeFlow — Tibia

Conditional OT flow-matching model reconstructing a full tibia point cloud
from tibial boundary points + 8 anatomical landmarks.

## Files

- `net.py` — BoundEnc + LMEnc + VFNet architecture (shared design with femur)
- `data.py` — data loading, normalisation, train/test split
- `cfg.py` — paths, hyperparameters
- `train.py` — training entry point (324 bones, 90/10 random split, seed=42)
- `infer.py` — single-bone inference, handles left/right mirroring (`--side R`)
- `eval_all_test.py` — batch inference + metrics (ASD/HD95/NSD) on the test set
- `Final_Tracking.py` — intra-operative tracking / visualization pipeline
- `checkpoints/ep800.pt` — checkpoint used for the paper's reported tibia results
- `checkpoints/norm_stats.json` — normalisation stats matching that checkpoint

## Usage

```
python train.py                       # train from scratch
python train.py --resume latest.pt    # resume

python infer.py --bound boundary.ply --lm landmarks.csv \
    --ckpt checkpoints/ep800.pt --out out.ply --side R

python eval_all_test.py --ckpt checkpoints/ep800.pt
```

Landmark CSV format expected (same as training):

```
Landmark,X,Y,Z
Tibial Knee Centre,x,y,z
Medial Plateau,x,y,z
Lateral Plateau,x,y,z
Tibial Tuberosity,x,y,z
Medial Malleolus,x,y,z
Lateral Malleolus,x,y,z
Ankle Centre,x,y,z
Posterior Cruciate Ligament,x,y,z
```
