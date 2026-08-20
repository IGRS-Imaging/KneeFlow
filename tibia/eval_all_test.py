"""
Tibia OT-FM — Batch evaluation on all test bones.
Loads test bone IDs from output_Tibia/test_bone_ids.txt (written by train.py).
Compares generated vs GT (from output_Tibia/test_bones_gt/).
Prints ASD, HD95, NSD@2mm per bone + summary stats.
Saves results to CSV.

Usage:
  python3 eval_all_test.py                        # uses best_asd.pt
  python3 eval_all_test.py --ckpt checkpoints/ep600.pt
"""
import os, sys, argparse, csv
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cfg
from net import FlowMatch
from data import (load_lm, load_pts_ply, clean_pts, rsample,
                  load_norm_stats, load_test_ids)

GT_DIR    = cfg.TEST_GT_DIR
OUT_DIR   = os.path.join(cfg.OUTPUT, "test_infer")
NORM_PATH = os.path.join(cfg.OUTPUT, "norm_stats.json")
TEST_IDS_PATH = os.path.join(cfg.OUTPUT, "test_bone_ids.txt")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics (same formula as femur eval_all_test.py)
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(pred, gt, nsd_tau=cfg.NSD_TAU, chunk=512):
    """
    ASD  = (mean d_p2g + mean d_g2p) / 2   symmetric
    HD95 = max(p95(d_p2g), p95(d_g2p))
    NSD  = 100 × (frac_p2g + frac_g2p) / 2  as %
    """
    P = torch.from_numpy(pred).float().cuda()
    G = torch.from_numpy(gt  ).float().cuda()

    d_p2g = torch.zeros(P.shape[0], device='cuda')
    for s in range(0, P.shape[0], chunk):
        e = min(s + chunk, P.shape[0])
        d_p2g[s:e] = torch.cdist(P[s:e], G).min(1)[0]

    d_g2p = torch.zeros(G.shape[0], device='cuda')
    for s in range(0, G.shape[0], chunk):
        e = min(s + chunk, G.shape[0])
        d_g2p[s:e] = torch.cdist(G[s:e], P).min(1)[0]

    d_p2g = d_p2g.cpu().numpy()
    d_g2p = d_g2p.cpu().numpy()

    asd  = float((d_p2g.mean() + d_g2p.mean()) / 2.0)
    hd   = float(max(d_p2g.max(), d_g2p.max()))
    hd95 = float(max(np.percentile(d_p2g, 95), np.percentile(d_g2p, 95)))
    nsd  = 100.0 * float(
        (np.mean(d_p2g < nsd_tau) + np.mean(d_g2p < nsd_tau)) / 2.0)
    return asd, hd, hd95, nsd


def load_ply_pts(path):
    pts = []
    with open(path, 'r') as f:
        in_header = True
        for line in f:
            line = line.strip()
            if in_header:
                if line == 'end_header': in_header = False
                continue
            vals = line.split()
            if len(vals) >= 3:
                try: pts.append([float(vals[0]), float(vals[1]), float(vals[2])])
                except ValueError: continue
    return np.array(pts, dtype=np.float32)


def save_ply(path, pts):
    pts = np.asarray(pts)
    with open(path, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in pts:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',    default=None)
    ap.add_argument('--steps',   type=int,   default=cfg.INF_STEPS)
    ap.add_argument('--solver',  default='heun', choices=['heun', 'euler'])
    ap.add_argument('--nsd_tau', type=float, default=cfg.NSD_TAU)
    ap.add_argument('--seed',    type=int,   default=42)
    ap.add_argument('--no_save', action='store_true')
    args = ap.parse_args()

    ckpt_path = args.ckpt or os.path.join(cfg.CKPT_DIR, 'best_asd.pt')
    os.makedirs(OUT_DIR, exist_ok=True)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device     : {dev}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Solver     : {args.solver.upper()}  steps={args.steps}  NFEs={args.steps if args.solver=='euler' else args.steps*2}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    model = FlowMatch().to(dev)
    ck    = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model.load_state_dict(ck['model'])
    model.eval()
    epoch = ck.get('epoch', '?')
    print(f"Loaded epoch {epoch}\n")

    gc, gs = load_norm_stats(NORM_PATH)

    # ── Load test bone IDs ────────────────────────────────────────────────────
    if not os.path.exists(TEST_IDS_PATH):
        raise FileNotFoundError(
            f"Test IDs not found: {TEST_IDS_PATH}\nRun train.py first.")
    test_ids = load_test_ids(TEST_IDS_PATH)
    print(f"Test bones : {len(test_ids)}\n")

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"{'Bone':<30} {'ASD(mm)':>8} {'HD95(mm)':>9} {'NSD@2mm%':>9}")
    print("─" * 60)

    results, failed = [], []

    for bone_id in sorted(test_ids):
        # bone_id format: R_37_Tibia_R or L_37_Tibia_L
        parts = bone_id.split('_', 1)
        if len(parts) < 2:
            failed.append(bone_id); continue
        side = parts[0]                  # L or R
        stem = parts[1]                  # 37_Tibia_R (without _rmesh)

        bound_dir = cfg.LEFT_BOUND  if side == 'L' else cfg.RIGHT_BOUND
        lm_dir    = cfg.LEFT_LM     if side == 'L' else cfg.RIGHT_LM

        # bone_id = e.g. R_109_Tibia_R  →  stem = 109_Tibia_R
        # Actual files: 109_Tibia_R_rmesh_landmarks.csv
        #               109_Tibia_R_rmesh_boundary.ply

        # Find boundary PLY
        bound_p = None
        for suffix in ['_rmesh_boundary.ply', '_boundary.ply',
                       '_bound.ply', '.ply']:
            c = os.path.join(bound_dir, f"{stem}{suffix}")
            if os.path.exists(c): bound_p = c; break

        # Find landmark CSV
        lm_p = None
        for suffix in ['_rmesh_landmarks.csv', '_landmarks.csv',
                       '_rmesh_landmarks.txt', '_landmarks.txt', '.csv']:
            t = os.path.join(lm_dir, f"{stem}{suffix}")
            if os.path.exists(t): lm_p = t; break

        gt_p  = os.path.join(GT_DIR,  f"{bone_id}_gt.ply")
        out_p = os.path.join(OUT_DIR, f"{bone_id}_generated.ply")

        miss = []
        if not bound_p:              miss.append('boundary')
        if not lm_p:                 miss.append('landmarks')
        if not os.path.exists(gt_p): miss.append('GT')
        if miss:
            print(f"  ✗ {bone_id:<27} missing: {', '.join(miss)}")
            failed.append(bone_id); continue

        try:
            bp = clean_pts(load_pts_ply(bound_p))
            lm = load_lm(lm_p)
            la = np.zeros((cfg.N_LM, 3), dtype=np.float32)
            for i, name in enumerate(cfg.LM_NAMES):
                if name in lm: la[i] = lm[name]

            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)

            bp_norm = (bp - gc) / gs
            la_norm = (la - gc) / gs

            # Mirror right bones to left-side space before inference
            # (same as done in load_all_raw during training)
            is_right = (side == 'R')
            if is_right:
                bp_norm = bp_norm.copy(); bp_norm[:, 0] *= -1
                la_norm = la_norm.copy(); la_norm[:, 0] *= -1

            bp_t = torch.from_numpy(rsample(bp_norm, cfg.N_BOUND)).unsqueeze(0).to(dev)
            la_t = torch.from_numpy(la_norm).unsqueeze(0).to(dev)

            with torch.no_grad():
                gen = model.generate(bp_t, la_t, n_pts=cfg.N_TIBIA,
                                     steps=args.steps, solver=args.solver)

            pred_mm = gen[0].cpu().numpy() * gs + gc

            # Flip output back to right-side space
            if is_right:
                pred_mm[:, 0] *= -1

            if not args.no_save:
                save_ply(out_p, pred_mm)

            gt_mm = load_ply_pts(gt_p)
            asd, hd, hd95, nsd = compute_metrics(pred_mm, gt_mm, args.nsd_tau)

            results.append((bone_id, asd, hd, hd95, nsd))
            print(f"{bone_id:<30} {asd:8.3f} {hd95:9.3f} {nsd:9.1f}")

        except Exception as e:
            import traceback
            print(f"  ✗ ERROR {bone_id}: {e}")
            traceback.print_exc()
            failed.append(bone_id)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("─" * 60)
    if not results:
        print("No results."); return

    asds  = np.array([r[1] for r in results])
    hds   = np.array([r[2] for r in results])
    hd95s = np.array([r[3] for r in results])
    nsds  = np.array([r[4] for r in results])

    def row(tag, fn):
        print(f"{tag:<30} {fn(asds):8.3f} {fn(hd95s):9.3f} {fn(nsds):9.1f}")

    row("MEAN",   np.mean)
    row("STD",    np.std)
    row("MEDIAN", np.median)
    row("MIN",    np.min)
    row("MAX",    np.max)

    print(f"\n  Evaluated : {len(results)} / {len(test_ids)} bones")
    if failed: print(f"  Failed    : {failed}")

    # ── ASD threshold breakdown ────────────────────────────────────────────────
    print("\n── ASD threshold breakdown ──")
    for t in [2.0, 3.0, 4.0, 5.0]:
        n = int((asds < t).sum())
        print(f"  ASD < {t:.0f}mm : {n}/{len(results)}  ({100*n/len(results):.0f}%)")

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = os.path.join(cfg.OUTPUT, f"eval_ep{epoch}_{args.solver}{args.steps}_tibia.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Bone', 'ASD_mm', 'HD95_mm', f'NSD@{args.nsd_tau}mm_%'])
        for r in results:
            w.writerow([r[0], f"{r[1]:.3f}", f"{r[3]:.3f}", f"{r[4]:.1f}"])
        w.writerow([])
        w.writerow(['MEAN', f"{asds.mean():.3f}",
                    f"{hd95s.mean():.3f}", f"{nsds.mean():.1f}"])
        w.writerow(['STD',  f"{asds.std():.3f}",
                    f"{hd95s.std():.3f}",  f"{nsds.std():.1f}"])
    print(f"\n  CSV saved → {csv_path}")


if __name__ == '__main__':
    main()
