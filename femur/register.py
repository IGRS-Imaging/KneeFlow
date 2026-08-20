"""
register.py  —  Full pipeline: Photon space → Canonical → Generate → Photon space

Patient landmarks + condylar boundary are in photon (scanner) coordinates.
Training femurs live in a canonical registered coordinate system.

Steps:
  1. Load patient landmarks + boundary (photon space)
  2. Load canonical reference landmarks (training mean OR a reference CSV)
  3. Compute rigid Procrustes transform: photon → canonical  (no scaling)
  4. Transform inputs to canonical space, normalize
  5. Run flow-matching inference → generated femur (canonical space)
  6. Denormalize, apply inverse transform → photon space
  7. Save output PLY in photon space

Usage:
  python register.py \
    --lm       patient_landmarks.csv \
    --bound    patient_boundary.csv \
    --ckpt     ..\output\checkpoints\ep700.pt \
    --out      ..\output\generated_femur.ply

  # Optional: supply a specific reference landmark file instead of training mean
  python register.py ... --ref_lm canonical_ref_landmarks.csv
"""
import argparse, os, glob
import numpy as np
import torch
import cfg
from net import FlowMatch
from data import load_pts, load_lm, clean_pts, rsample, load_norm_stats


def save_ply(path, pts):
    pts = np.asarray(pts)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in pts:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def rigid_procrustes(src, tgt):
    """
    Compute R, t such that:  tgt ≈ src @ R.T + t   (no scaling)
    Uses SVD; det correction ensures proper rotation (no reflection).
    """
    src_c = src.mean(0)
    tgt_c = tgt.mean(0)
    H = (src - src_c).T @ (tgt - tgt_c)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T  # (3, 3)
    t = tgt_c - src_c @ R.T
    return R.astype(np.float32), t.astype(np.float32)


def apply_transform(pts, R, t):
    return pts @ R.T + t


def apply_inv_transform(pts, R, t):
    return (pts - t) @ R


def load_lm_array(csv_path):
    """Load landmark CSV → (N_LM, 3) array ordered by cfg.LM_NAMES."""
    d = load_lm(csv_path)
    arr = np.zeros((cfg.N_LM, 3), dtype=np.float32)
    missing = []
    for i, n in enumerate(cfg.LM_NAMES):
        if n in d:
            arr[i] = d[n]
        else:
            missing.append(n)
    if missing:
        print(f"  Warning: missing landmarks: {missing}")
    return arr


def compute_mean_landmarks(lm_dirs):
    """Average landmark positions over all training CSV files."""
    accum = {n: [] for n in cfg.LM_NAMES}
    count = 0
    for d in lm_dirs:
        if not os.path.isdir(d):
            print(f"  Warning: landmark dir not found: {d}")
            continue
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith('.csv'):
                continue
            lm = load_lm(os.path.join(d, f))
            for n in cfg.LM_NAMES:
                if n in lm:
                    accum[n].append(lm[n])
            count += 1
    if count == 0:
        raise RuntimeError(
            "No training landmark CSVs found. "
            "Supply --ref_lm pointing to a canonical reference landmark file."
        )
    mean_lm = np.zeros((cfg.N_LM, 3), dtype=np.float32)
    for i, n in enumerate(cfg.LM_NAMES):
        if accum[n]:
            mean_lm[i] = np.mean(accum[n], axis=0)
        else:
            print(f"  Warning: no data for landmark '{n}' — defaulting to zero")
    print(f"  Mean landmarks computed from {count} training files.")
    return mean_lm


def main():
    p = argparse.ArgumentParser(
        description="Photon-space → Canonical → Generate → Photon-space"
    )
    p.add_argument('--lm',         required=True,
                   help='Patient landmarks CSV (photon/scanner space)')
    p.add_argument('--bound',      required=True,
                   help='Condylar boundary CSV or PLY (photon/scanner space)')
    p.add_argument('--ckpt',       required=True,
                   help='Checkpoint .pt (e.g. ep700.pt)')
    p.add_argument('--out',        required=True,
                   help='Output PLY path (result will be in photon space)')
    p.add_argument('--ref_lm',     default=None,
                   help='Canonical reference landmarks CSV. '
                        'If omitted, mean of training landmarks is used.')
    p.add_argument('--norm_stats', default=None,
                   help='Path to norm_stats.json (default: output/norm_stats.json)')
    p.add_argument('--npts',       type=int, default=cfg.N_FEMUR,
                   help=f'Output point count (default {cfg.N_FEMUR})')
    p.add_argument('--steps',      type=int, default=cfg.INF_STEPS,
                   help=f'Heun integration steps (default {cfg.INF_STEPS})')
    a = p.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {dev}")

    # ── Load model ───────────────────────────────────────────────────────────
    model = FlowMatch().to(dev)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(ck['model'])
    model.eval()
    print(f"Checkpoint loaded: epoch={ck['epoch']}, best_loss={ck['best_loss']:.4f}")

    # ── Norm stats (computed during training) ─────────────────────────────
    gc, gs = load_norm_stats(a.norm_stats)
    print(f"Norm stats: centroid={gc}, std={gs:.4f}")

    # ── Patient data in photon space ──────────────────────────────────────
    lm_photon = load_lm_array(a.lm)
    bp_photon = clean_pts(load_pts(a.bound))
    if len(bp_photon) == 0:
        raise ValueError(f"No valid boundary points in {a.bound}")
    print(f"Patient boundary: {len(bp_photon)} points (photon space)")
    print(f"Patient landmarks loaded from: {a.lm}")

    # ── Canonical reference landmarks ─────────────────────────────────────
    if a.ref_lm:
        lm_canon_ref = load_lm_array(a.ref_lm)
        print(f"Reference landmarks: {a.ref_lm}")
    else:
        print("Computing canonical reference from training landmark mean ...")
        lm_canon_ref = compute_mean_landmarks([cfg.LEFT_LM, cfg.RIGHT_LM])

    # ── Rigid registration: photon → canonical ────────────────────────────
    R, t = rigid_procrustes(lm_photon, lm_canon_ref)
    det  = np.linalg.det(R)
    print(f"Rigid transform computed  (R det={det:.4f}, should be +1.0)")
    if abs(det - 1.0) > 0.01:
        print("  Warning: det far from 1 — check landmark correspondence")

    lm_canon = apply_transform(lm_photon, R, t)
    bp_canon = apply_transform(bp_photon, R, t)

    # ── Normalize to training distribution ───────────────────────────────
    lm_norm = (lm_canon - gc) / gs
    bp_norm = (bp_canon - gc) / gs

    bp_t = torch.from_numpy(rsample(bp_norm, cfg.N_BOUND)).unsqueeze(0).to(dev)
    lm_t = torch.from_numpy(lm_norm).unsqueeze(0).to(dev)

    # ── Generate (canonical + normalized space) ───────────────────────────
    print(f"Generating {a.npts} points with {a.steps} Heun steps ...")
    with torch.no_grad():
        g = model.generate(bp_t, lm_t, n_pts=a.npts, steps=a.steps)

    # ── Denormalize → canonical space ─────────────────────────────────────
    pts_canon = g[0].cpu().numpy() * gs + gc

    # ── Inverse transform → photon space ──────────────────────────────────
    pts_photon = apply_inv_transform(pts_canon, R, t)

    save_ply(a.out, pts_photon)
    print(f"Saved {len(pts_photon)} points → {a.out}  (photon space)")


if __name__ == "__main__":
    main()
