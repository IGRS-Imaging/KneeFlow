"""
Batch inference + evaluation on all 29 test bones.
- Runs generation for each test bone using boundary + landmarks only
- Compares generated vs GT (from test_bones_gt folder)
- Prints ASD, CD, HD, HD95, NSD@2mm per bone + summary stats
- Saves results to CSV for paper tables
- Saves generated PLY files to test_infer/

Usage:
  python eval_all_test.py                           # uses ep1050.pt (default)
  python eval_all_test.py --ckpt checkpoints/ep1100.pt
"""
import os, sys, argparse
import numpy as np
import torch
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cfg
from net import FlowMatch
from data import load_pts, load_lm, clean_pts, rsample, load_norm_stats

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE      = "/home/imaging/Desktop/Ragavan"
GT_DIR    = os.path.join(BASE, "9L_working_output", "test_bones_gt")
OUT_DIR   = os.path.join(BASE, "9L_working_output", "test_infer")
NORM      = os.path.join(BASE, "9L_working_output", "norm_stats.json")

LEFT_BOUND  = os.path.join(BASE, "left",  "left_condyaler_points")
LEFT_LM     = os.path.join(BASE, "left",  "registered_landmarks")
RIGHT_BOUND = os.path.join(BASE, "right", "right_condyaler_points")
RIGHT_LM    = os.path.join(BASE, "right", "registered_landmarks")

# ── All 29 test bones (selected_29_dice78_82) ──────────────────────────────────
TEST_BONES = [
    ('L', '222_Femur_L'), ('L', '225_Femur_L'), ('L', '232_Femur_L'),
    ('L', '237_Femur_L'), ('L', '245_Femur_L'), ('L', '285_Femur_L'),
    ('L', '298_Femur_L'), ('L', '308_Femur_L'), ('L', '328_Femur_L'),
    ('L', '330_Femur_L'), ('L', '335_Femur_L'), ('L', '343_Femur_L'),
    ('L', '350_Femur_L'), ('L', '354_Femur_L'), ('L', '370_Femur_L'),
    ('L', '380_Femur_L'), ('L', '381_Femur_L'), ('L', '392_Femur_L'),
    ('R', '43_Femur_R'),  ('R', '62_Femur_R'),  ('R', '99_Femur_R'),
    ('R', '101_Femur_R'), ('R', '114_Femur_R'), ('R', '163_Femur_R'),
    ('R', '197_Femur_R'), ('R', '198_Femur_R'), ('R', '203_Femur_R'),
    ('R', '210_Femur_R'), ('R', '215_Femur_R'),
]


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(pred, gt, nsd_tau=cfg.NSD_TAU, chunk=2048):
    """
    Compute ASD, HD, HD95, NSD@tau between two (N,3) numpy arrays (mm).

    ASD  = (mean d_pred→GT + mean d_GT→pred) / 2  — symmetric, matches m_cd.py
    HD   = max(max d_p2g, max d_g2p)  — full Hausdorff
    HD95 = max(p95(d_p2g), p95(d_g2p)) — matches m_cd.py and standard literature
    NSD  = 100 × (frac_p2g + frac_g2p) / 2  — Reinke et al. 2021, reported as %
    """
    P = torch.from_numpy(pred).float().cuda()
    G = torch.from_numpy(gt).float().cuda()

    # P → G distances
    d_p2g = torch.zeros(P.shape[0], device='cuda')
    for s in range(0, P.shape[0], chunk):
        e = min(s + chunk, P.shape[0])
        d_p2g[s:e] = torch.cdist(P[s:e], G).min(1)[0]

    # G → P distances
    d_g2p = torch.zeros(G.shape[0], device='cuda')
    for s in range(0, G.shape[0], chunk):
        e = min(s + chunk, G.shape[0])
        d_g2p[s:e] = torch.cdist(G[s:e], P).min(1)[0]

    d_p2g = d_p2g.cpu().numpy()
    d_g2p = d_g2p.cpu().numpy()

    # ASD — symmetric average, matches m_cd.py formula exactly
    asd  = float((d_p2g.mean() + d_g2p.mean()) / 2.0)
    hd   = float(max(d_p2g.max(), d_g2p.max()))
    # HD95 — max of each direction's p95; matches m_cd.py and standard literature
    hd95 = float(max(np.percentile(d_p2g, 95), np.percentile(d_g2p, 95)))
    # NSD@tau — standard Reinke et al. 2021 definition, reported as % (0–100)
    nsd  = 100.0 * float((np.mean(d_p2g < nsd_tau) + np.mean(d_g2p < nsd_tau)) / 2.0)

    return asd, hd, hd95, nsd


def load_ply_pts(path):
    """Load point-cloud PLY → (N,3) numpy array."""
    pts = []
    with open(path, 'r') as f:
        in_header = True
        for line in f:
            line = line.strip()
            if in_header:
                if line == 'end_header':
                    in_header = False
                continue
            vals = line.split()
            if len(vals) >= 3:
                try:
                    pts.append([float(vals[0]), float(vals[1]), float(vals[2])])
                except ValueError:
                    continue
    return np.array(pts, dtype=np.float32)


def save_ply(path, pts):
    pts = np.asarray(pts)
    with open(path, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in pts:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=None,
                    help='Checkpoint .pt (default: best_asd.pt)')
    ap.add_argument('--steps', type=int, default=cfg.INF_STEPS,
                    help=f'ODE steps (default {cfg.INF_STEPS})')
    ap.add_argument('--nsd_tau', type=float, default=cfg.NSD_TAU,
                    help=f'NSD threshold mm (default {cfg.NSD_TAU})')
    ap.add_argument('--no_save', action='store_true',
                    help='Skip saving generated PLY files (faster)')
    args = ap.parse_args()

    ckpt_path = args.ckpt or os.path.join(
        BASE, "9L_working_output", "checkpoints", "ep1050.pt")

    os.makedirs(OUT_DIR, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device  : {dev}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"NSD tau : {args.nsd_tau} mm\n")

    # ── Load model ────────────────────────────────────────────────────────────
    model = FlowMatch().to(dev)
    ck    = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model.load_state_dict(ck['model'])
    model.eval()
    epoch = ck.get('epoch', '?')
    print(f"Loaded checkpoint: epoch {epoch}\n")

    # ── Load norm stats ───────────────────────────────────────────────────────
    gc, gs = load_norm_stats(NORM)

    # ── Header ────────────────────────────────────────────────────────────────
    HDR = f"{'Bone':<26} {'ASD(mm)':>8} {'HD95(mm)':>9} {'NSD@2mm%':>9}"
    print(HDR)
    print("-" * 55)

    results = []
    failed  = []

    for side, sid in TEST_BONES:
        label = f"{side}_{sid}"

        # Input paths
        bound_dir = LEFT_BOUND  if side == 'L' else RIGHT_BOUND
        lm_dir    = LEFT_LM     if side == 'L' else RIGHT_LM

        bound_p = None
        for ext in ['.csv', '.ply']:
            c = os.path.join(bound_dir, f"{sid}_boundary{ext}")
            if os.path.exists(c):
                bound_p = c
                break
            # also try without _boundary suffix
            c2 = os.path.join(bound_dir, f"{sid}{ext}")
            if os.path.exists(c2):
                bound_p = c2
                break

        lm_p = None
        for ext in ['.csv', '.txt']:
            t = os.path.join(lm_dir, f"{sid}_landmarks{ext}")
            if os.path.exists(t):
                lm_p = t
                break
            t2 = os.path.join(lm_dir, f"{sid}{ext}")
            if os.path.exists(t2):
                lm_p = t2
                break

        gt_p  = os.path.join(GT_DIR, f"{label}_gt.ply")
        out_p = os.path.join(OUT_DIR, f"{label}_generated.ply")

        miss = []
        if not bound_p: miss.append('boundary')
        if not lm_p:    miss.append('landmarks')
        if not os.path.exists(gt_p): miss.append('GT')
        if miss:
            print(f"  ✗ {label:<24} missing: {', '.join(miss)}")
            failed.append(label)
            continue

        try:
            # Inference
            bp   = clean_pts(load_pts(bound_p))
            lm   = load_lm(lm_p)
            la   = np.zeros((cfg.N_LM, 3), dtype=np.float32)
            for i, n in enumerate(cfg.LM_NAMES):
                if n in lm:
                    la[i] = lm[n]

            # Fix seed BEFORE any randomness (rsample uses numpy, generate uses torch)
            np.random.seed(42)
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)

            bp_norm = (bp - gc) / gs
            la_norm = (la - gc) / gs
            bp_t    = torch.from_numpy(rsample(bp_norm, cfg.N_BOUND)).unsqueeze(0).to(dev)
            la_t    = torch.from_numpy(la_norm).unsqueeze(0).to(dev)

            with torch.no_grad():
                gen = model.generate(bp_t, la_t,
                                     n_pts=cfg.N_FEMUR,
                                     steps=args.steps)

            pred_mm = gen[0].cpu().numpy() * gs + gc

            if not args.no_save:
                save_ply(out_p, pred_mm)

            # Metrics
            gt_mm = load_ply_pts(gt_p)
            asd, hd, hd95, nsd = compute_metrics(pred_mm, gt_mm,
                                                   nsd_tau=args.nsd_tau)

            results.append((label, asd, hd, hd95, nsd))
            print(f"{label:<26} {asd:8.3f} {hd95:9.3f} {nsd:9.1f}")

        except Exception as e:
            import traceback
            print(f"  ✗ ERROR {label}: {e}")
            traceback.print_exc()
            failed.append(label)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("-" * 55)
    if not results:
        print("No results computed.")
        return

    asds  = np.array([r[1] for r in results])
    hds   = np.array([r[2] for r in results])
    hd95s = np.array([r[3] for r in results])
    nsds  = np.array([r[4] for r in results])

    def row(tag, fn):
        print(f"{tag:<26} {fn(asds):8.3f} {fn(hd95s):9.3f} {fn(nsds):9.1f}")

    row("MEAN", np.mean)
    row("STD",  np.std)
    row("MIN",  np.min)
    row("MAX",  np.max)
    row("MED",  np.median)

    # Excluding worst-ASD outlier (was L_320 under the old test split; that bone
    # is not in the current 29-bone test set, so this is now a no-op filter)
    mask = np.array([r[0] != 'L_320_Femur_L' for r in results])
    if mask.sum() > 0:
        print(f"\n  Excl. L_320 ({mask.sum()} bones):")
        print(f"    Mean ASD  : {asds[mask].mean():.3f} mm")
        print(f"    Mean HD95 : {hd95s[mask].mean():.3f} mm")
        print(f"    Mean NSD  : {nsds[mask].mean():.1f}%")

    # Percentile breakdown (useful for paper)
    thresholds = [2.0, 3.0, 4.0, 5.0, 6.0]
    print("\n── ASD threshold breakdown ──")
    for t in thresholds:
        n  = int((asds < t).sum())
        print(f"  ASD < {t:.0f}mm : {n}/{len(results)} bones  "
              f"({100*n/len(results):.0f}%)")

    print(f"\n  Evaluated : {len(results)} / {len(TEST_BONES)} bones")
    if failed:
        print(f"  Failed    : {failed}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(BASE, "9L_working_output",
                            f"eval_ep{epoch}_results.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Bone', 'ASD_mm', 'HD95_mm', f'NSD@{args.nsd_tau}mm_%'])
        for r in results:
            w.writerow([r[0], f"{r[1]:.3f}", f"{r[3]:.3f}", f"{r[4]:.1f}"])
        w.writerow([])
        w.writerow(['MEAN', f"{asds.mean():.3f}", f"{hd95s.mean():.3f}", f"{nsds.mean():.1f}"])
        w.writerow(['STD',  f"{asds.std():.3f}",  f"{hd95s.std():.3f}",  f"{nsds.std():.1f}"])
        if mask.sum() > 0:
            w.writerow([])
            w.writerow([f'MEAN (excl L_320)', f"{asds[mask].mean():.3f}",
                        f"{hd95s[mask].mean():.3f}", f"{nsds[mask].mean():.1f}"])

    print(f"\n  Results CSV saved → {csv_path}")


if __name__ == "__main__":
    main()
