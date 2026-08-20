"""
Inference: condylar boundary + landmarks → full femur point cloud
"""
import argparse, os
import numpy as np
import torch
import cfg
from net import FlowMatch
from data import load_pts, load_lm, clean_pts, rsample, load_norm_stats


def save_ply(path, pts):
    pts = np.asarray(pts)
    with open(path, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in pts: f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bound',       required=True,  help='Condylar boundary .ply or .csv')
    p.add_argument('--lm',          required=True,  help='Landmarks CSV')
    p.add_argument('--ckpt',        required=True,  help='Checkpoint .pt')
    p.add_argument('--out',         required=True,  help='Output PLY path')
    p.add_argument('--norm_stats',  default=None,   help='Path to norm_stats.json (overrides cfg.OUTPUT)')
    p.add_argument('--npts',        type=int, default=cfg.N_FEMUR)
    p.add_argument('--steps',       type=int, default=cfg.INF_STEPS)
    a = p.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {dev}")

    model = FlowMatch().to(dev)
    ck    = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(ck['model']); model.eval()
    best = ck.get('best_train_loss', ck.get('best_loss', float('nan')))
    print(f"Loaded: epoch={ck['epoch']}, best_train_loss={best:.4f}")

    # Load global norm stats — same coordinate frame as training
    gc, gs = load_norm_stats(a.norm_stats)
    print(f"Norm: centroid={gc}, std={gs:.4f}")

    # Condylar boundary
    bp = clean_pts(load_pts(a.bound))
    if len(bp) == 0: raise ValueError(f"No valid points in {a.bound}")
    print(f"Boundary points loaded: {len(bp)}")
    bp_norm = (bp - gc) / gs
    bp_t    = torch.from_numpy(rsample(bp_norm, cfg.N_BOUND)).unsqueeze(0).to(dev)

    # Landmarks
    lm = load_lm(a.lm)
    la = np.zeros((cfg.N_LM, 3), dtype=np.float32)
    for i, n in enumerate(cfg.LM_NAMES):
        if n in lm: la[i] = lm[n]
        else: print(f"  Warning: landmark '{n}' missing")
    la_norm = (la - gc) / gs
    la_t    = torch.from_numpy(la_norm).unsqueeze(0).to(dev)

    print(f"Generating {a.npts} points, {a.steps} steps...")
    with torch.no_grad():
        g = model.generate(bp_t, la_t, n_pts=a.npts, steps=a.steps)

    # Denormalize → real mm coordinates
    pts = g[0].cpu().numpy() * gs + gc
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    save_ply(a.out, pts)
    print(f"Saved {len(pts)} points → {a.out}")


if __name__ == "__main__":
    main()
