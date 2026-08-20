"""
register_and_infer.py

Full pipeline in one script:
  1. Load patient (phantom) landmarks + condylar boundary  [phantom space]
  2. Load canonical reference landmarks (training-data mean OR a supplied CSV)
  3. Compute rigid Kabsch transform: phantom → canonical   (rotation + translation, NO scaling)
  4. Transform boundary + landmarks into canonical space, then normalize to training distribution
  5. Run OT-Flow-Matching inference → femur point cloud   [canonical-normalized space]
  6. Denormalize → canonical mm space
  7. Apply exact inverse transform → phantom mm space
  8. Save output PLY in phantom coordinate space

Transform math:
  Forward:  p_canon   = p_phantom @ R.T + t
  Inverse:  p_phantom = (p_canon - t) @ R
  R is orthogonal (det = +1), so R.T = R^{-1}.

Usage
-----
  cd femur_fm_9L

  python register_and_infer.py ^
    --lm    phantom_landmarks.csv ^
    --bound phantom_condylar_boundary.csv ^
    --ckpt  ..\9_landmarks_output\checkpoints\best_asd.pt ^
    --out   ..\9_landmarks_output\generated_femur.ply

  # Optional: supply a specific canonical reference instead of training mean
  python register_and_infer.py ... --ref_lm canonical_ref_landmarks.csv

  # Optional: point directly to norm_stats.json
  python register_and_infer.py ... --norm_stats ..\output\norm_stats.json
"""

import argparse
import os

import numpy as np
import torch

import cfg
from data import clean_pts, load_lm, load_norm_stats, load_pts, rsample
from net import FlowMatch


# ── PLY writer ──────────────────────────────────────────────────────────────

def save_ply(path, pts):
    pts = np.asarray(pts, dtype=np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for v in pts:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")


# ── Rigid registration (Kabsch algorithm) ────────────────────────────────────

def kabsch(src, tgt):
    """
    Find R, t (rigid, no scale) such that:  tgt ≈ src @ R.T + t

    Uses float64 throughout for numerical accuracy.
    Applies det-correction so R is always a proper rotation (det = +1, no reflection).

    Args:
        src : (N, 3) float64 — source points (phantom/scanner space)
        tgt : (N, 3) float64 — target points (canonical training space)

    Returns:
        R : (3, 3) float64 — rotation matrix, R.T = R^{-1}
        t : (3,)   float64 — translation vector
    """
    src = np.asarray(src, dtype=np.float64)
    tgt = np.asarray(tgt, dtype=np.float64)

    src_c = src.mean(0)          # centroid of source
    tgt_c = tgt.mean(0)          # centroid of target

    A = src - src_c              # centred source  (N, 3)
    B = tgt - tgt_c              # centred target  (N, 3)

    H = A.T @ B                  # cross-covariance  (3, 3)
    U, _, Vt = np.linalg.svd(H)

    # det-correction: prevents reflection when points are collinear or mirrored
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T   # (3, 3)

    t = tgt_c - src_c @ R.T     # translation

    return R, t


def apply_fwd(pts, R, t):
    """phantom → canonical:  pts_canon = pts_phantom @ R.T + t"""
    return pts.astype(np.float64) @ R.T + t


def apply_inv(pts, R, t):
    """canonical → phantom:  pts_phantom = (pts_canon - t) @ R"""
    return (pts.astype(np.float64) - t) @ R


def registration_rmse(src, tgt, R, t):
    """Per-landmark RMSE after applying the transform (diagnostic only)."""
    pred = apply_fwd(src, R, t)
    return float(np.sqrt(np.mean(np.sum((pred - tgt) ** 2, axis=1))))


# ── Canonical reference landmarks ────────────────────────────────────────────

def compute_mean_canonical_landmarks():
    """
    Average the position of each landmark across all training landmark CSVs.
    Used when --ref_lm is not supplied.
    Returns dict {landmark_name: np.array([x,y,z], float64)}.
    """
    accum = {n: [] for n in cfg.LM_NAMES}
    count = 0
    for d in [cfg.LEFT_LM, cfg.RIGHT_LM]:
        if not os.path.isdir(d):
            print(f"  [warn] Training landmark dir not found: {d}")
            continue
        for fname in sorted(os.listdir(d)):
            if not fname.lower().endswith('.csv'):
                continue
            lm = load_lm(os.path.join(d, fname))
            for n in cfg.LM_NAMES:
                if n in lm:
                    accum[n].append(lm[n].astype(np.float64))
            count += 1

    if count == 0:
        raise RuntimeError(
            "No training landmark CSVs found.\n"
            "Supply --ref_lm with a single canonical reference landmark CSV."
        )

    ref = {}
    for n in cfg.LM_NAMES:
        if accum[n]:
            ref[n] = np.mean(accum[n], axis=0)
        else:
            print(f"  [warn] No data for landmark '{n}' in training set")

    print(f"  Canonical reference: averaged over {count} training landmark files "
          f"({len(ref)}/{cfg.N_LM} landmarks present)")
    return ref


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Phantom-space → Canonical → OT-FM inference → Phantom-space"
    )
    p.add_argument('--lm',         required=True,
                   help='Patient landmark CSV in phantom (scanner) space')
    p.add_argument('--bound',      required=True,
                   help='Condylar boundary CSV or PLY in phantom (scanner) space')
    p.add_argument('--ckpt',       required=True,
                   help='Flow-matching checkpoint (.pt)')
    p.add_argument('--out',        required=True,
                   help='Output PLY path — result will be in phantom space')
    p.add_argument('--ref_lm',     default=None,
                   help='Canonical reference landmark CSV. '
                        'If omitted, the training-data mean is used.')
    p.add_argument('--norm_stats', default=None,
                   help='Path to norm_stats.json (default: output/norm_stats.json)')
    p.add_argument('--npts',       type=int, default=cfg.N_FEMUR,
                   help=f'Output femur point count (default {cfg.N_FEMUR})')
    p.add_argument('--steps',      type=int, default=cfg.INF_STEPS,
                   help=f'Heun ODE integration steps (default {cfg.INF_STEPS})')
    a = p.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f" register_and_infer.py")
    print(f"{'='*60}")
    print(f" Device        : {dev}")
    if dev.type == 'cuda':
        print(f" GPU           : {torch.cuda.get_device_name(0)}")
        print(f" VRAM          : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    # ── 1. Load model ──────────────────────────────────────────────────────
    print(f"\n[1/6] Loading model from {a.ckpt}")
    model = FlowMatch().to(dev)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(ck['model'])
    model.eval()
    best = ck.get('best_train_loss', ck.get('best_loss', float('nan')))
    print(f"      Epoch {ck['epoch']} | best_train_loss = {best:.4f}")

    # ── 2. Load norm stats ─────────────────────────────────────────────────
    print(f"\n[2/6] Loading norm stats")
    gc, gs = load_norm_stats(a.norm_stats)
    print(f"      centroid = {gc}")
    print(f"      std      = {gs:.4f} mm")

    # ── 3. Load phantom inputs ─────────────────────────────────────────────
    print(f"\n[3/6] Loading phantom-space inputs")
    lm_phantom_dict = load_lm(a.lm)
    print(f"      Landmarks loaded   : {len(lm_phantom_dict)} / {cfg.N_LM}")

    bp_phantom = clean_pts(load_pts(a.bound))
    if len(bp_phantom) == 0:
        raise ValueError(f"No valid boundary points in: {a.bound}")
    print(f"      Boundary points    : {len(bp_phantom)}")

    # ── 4. Rigid registration: phantom → canonical ─────────────────────────
    print(f"\n[4/6] Computing rigid registration  (phantom → canonical)")

    if a.ref_lm:
        lm_canon_dict = load_lm(a.ref_lm)
        print(f"      Reference source   : {a.ref_lm}")
    else:
        print(f"      Reference source   : training-data mean landmarks")
        lm_canon_dict = compute_mean_canonical_landmarks()

    # Use ONLY landmarks present in BOTH phantom and reference
    common = [n for n in cfg.LM_NAMES
              if n in lm_phantom_dict and n in lm_canon_dict]
    if len(common) < 3:
        raise ValueError(
            f"Need ≥3 common landmarks for registration; found {len(common)}: {common}\n"
            "Check that landmark names in your CSV match cfg.LM_NAMES exactly "
            "(note: LM 1-8 ALL CAPS, LM 9 'Lateral Anterior Cortex' Mixed Case)."
        )

    src = np.array([lm_phantom_dict[n] for n in common], dtype=np.float64)
    tgt = np.array([lm_canon_dict[n]   for n in common], dtype=np.float64)

    R, t = kabsch(src, tgt)
    rmse = registration_rmse(src, tgt, R, t)

    det = np.linalg.det(R)
    print(f"      Landmarks used     : {len(common)} / {cfg.N_LM}")
    print(f"      Unused landmarks   : {[n for n in cfg.LM_NAMES if n not in common]}")
    print(f"      R  det             : {det:+.6f}  (should be +1.0)")
    print(f"      Registration RMSE  : {rmse:.4f} mm")
    if rmse > 5.0:
        print(f"  [WARN] RMSE > 5 mm — check landmark CSV names and coordinate frame")
    if abs(det - 1.0) > 0.01:
        print(f"  [WARN] det(R) ≠ 1 — reflection detected; check landmark pairing")

    # Transform ALL landmarks to canonical space (use R,t from common subset)
    lm_canon_arr = np.zeros((cfg.N_LM, 3), dtype=np.float64)
    for i, n in enumerate(cfg.LM_NAMES):
        if n in lm_phantom_dict:
            lm_canon_arr[i] = apply_fwd(lm_phantom_dict[n][None], R, t)[0]
        else:
            print(f"  [warn] Landmark '{n}' missing from phantom file — zero-filled")

    # Transform boundary to canonical space
    bp_canon = apply_fwd(bp_phantom, R, t)

    # ── 5. Normalize to training distribution ──────────────────────────────
    print(f"\n[5/6] Normalizing to training distribution")
    lm_norm = ((lm_canon_arr - gc) / gs).astype(np.float32)
    bp_norm = ((bp_canon    - gc) / gs).astype(np.float32)

    bp_t = torch.from_numpy(rsample(bp_norm, cfg.N_BOUND)).unsqueeze(0).to(dev)
    lm_t = torch.from_numpy(lm_norm).unsqueeze(0).to(dev)

    print(f"      Boundary resampled : {cfg.N_BOUND} pts")
    print(f"      Landmarks          : {cfg.N_LM} pts")

    # ── 6. Generate ────────────────────────────────────────────────────────
    print(f"\n[6/6] Generating femur  ({a.npts} pts, {a.steps} Heun steps)")
    with torch.no_grad():
        g = model.generate(bp_t, lm_t, n_pts=a.npts, steps=a.steps)

    # Denormalize → canonical mm
    pts_canon = g[0].cpu().numpy().astype(np.float64) * gs + gc

    # Exact inverse transform → phantom space
    pts_phantom = apply_inv(pts_canon, R, t).astype(np.float32)

    # ── Save ───────────────────────────────────────────────────────────────
    save_ply(a.out, pts_phantom)
    print(f"\n Done.  {len(pts_phantom)} points saved → {a.out}")
    print(f"        Output is in phantom (scanner) coordinate space.\n")


if __name__ == "__main__":
    main()
