"""
Femur OT-FM — Solver Ablation Evaluation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tests all 29 test bones with chosen solver and step count.
Measures per-bone inference time.

Usage:
  python3 eval_solver.py --solver euler --steps 100
  python3 eval_solver.py --solver euler --steps 200
  python3 eval_solver.py --solver heun  --steps 100
  python3 eval_solver.py --solver heun  --steps 200
"""
import os, sys, argparse, csv, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cfg
from net import FlowMatch
from data import load_pts, load_lm, clean_pts, rsample, load_norm_stats

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
]  # 29 bones


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_ply_pts(path):
    pts = []
    with open(path, 'r') as f:
        in_hdr = True
        for line in f:
            line = line.strip()
            if in_hdr:
                if line == 'end_header': in_hdr = False
                continue
            vals = line.split()
            if len(vals) >= 3:
                try: pts.append([float(v) for v in vals[:3]])
                except ValueError: continue
    return np.array(pts, dtype=np.float32)


def save_ply(path, pts):
    pts = np.asarray(pts, dtype=np.float32)
    with open(path, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in pts:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def compute_metrics(pred_mm, gt_mm, nsd_tau=cfg.NSD_TAU, chunk=512):
    P = torch.from_numpy(pred_mm).float().cuda()
    Q = torch.from_numpy(gt_mm  ).float().cuda()

    d_p2g = torch.zeros(P.shape[0], device='cuda')
    for s in range(0, P.shape[0], chunk):
        e = min(s + chunk, P.shape[0])
        d_p2g[s:e] = torch.cdist(P[s:e], Q).min(1)[0]

    d_g2p = torch.zeros(Q.shape[0], device='cuda')
    for s in range(0, Q.shape[0], chunk):
        e = min(s + chunk, Q.shape[0])
        d_g2p[s:e] = torch.cdist(Q[s:e], P).min(1)[0]

    d_p2g = d_p2g.cpu().numpy()
    d_g2p = d_g2p.cpu().numpy()

    asd  = float((d_p2g.mean() + d_g2p.mean()) / 2.0)
    hd   = float(max(d_p2g.max(), d_g2p.max()))
    hd95 = float(max(np.percentile(d_p2g, 95), np.percentile(d_g2p, 95)))
    nsd  = 100.0 * float(
        (np.mean(d_p2g < nsd_tau) + np.mean(d_g2p < nsd_tau)) / 2.0)
    return asd, hd, hd95, nsd


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',    default=None)
    ap.add_argument('--solver',  default='heun', choices=['heun', 'euler'])
    ap.add_argument('--steps',   type=int, default=cfg.INF_STEPS)
    ap.add_argument('--nsd_tau', type=float, default=cfg.NSD_TAU)
    ap.add_argument('--seed',    type=int, default=42)
    args = ap.parse_args()

    tag      = f"{args.solver}{args.steps}"
    nfes     = args.steps if args.solver == 'euler' else args.steps * 2
    out_dir  = os.path.join(cfg.OUTPUT, f"solver_infer_{tag}")
    gt_dir   = os.path.join(cfg.OUTPUT, "test_bones_gt")
    os.makedirs(out_dir, exist_ok=True)

    if args.ckpt is None:
        ckpt_path = os.path.join(cfg.CKPT_DIR, 'ep700.pt')
    elif os.path.isabs(args.ckpt) or os.path.exists(args.ckpt):
        ckpt_path = args.ckpt
    else:
        ckpt_path = os.path.join(cfg.CKPT_DIR, args.ckpt)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device     : {dev}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Solver     : {args.solver.upper()}  steps={args.steps}  NFEs={nfes}")
    print(f"Output     : {out_dir}\n")

    # ── Load model ─────────────────────────────────────────────────────────────
    model = FlowMatch().to(dev)
    ck    = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model.load_state_dict(ck['model'])
    model.eval()
    epoch = ck.get('epoch', '?')
    print(f"Loaded epoch {epoch}\n")

    norm_path = os.path.join(cfg.OUTPUT, "norm_stats.json")
    gc, gs    = load_norm_stats(norm_path)

    # warmup GPU (first call is always slower)
    print("Warming up GPU...")
    with torch.no_grad():
        dummy_b = torch.randn(1, cfg.N_BOUND, 3, device=dev)
        dummy_l = torch.randn(1, cfg.N_LM,    3, device=dev)
        _ = model.generate(dummy_b, dummy_l, steps=args.steps, solver=args.solver)
    if dev.type == 'cuda':
        torch.cuda.synchronize()
    print("Done.\n")

    print(f"{'Bone':<26} {'ASD':>7} {'HD95':>7} {'NSD%':>7} {'Time(s)':>9}")
    print("─" * 62)

    results, times, failed = [], [], []

    for side, sid in TEST_BONES:
        label     = f"{side}_{sid}"
        bound_dir = cfg.LEFT_BOUND  if side == 'L' else cfg.RIGHT_BOUND
        lm_dir    = cfg.LEFT_LM     if side == 'L' else cfg.RIGHT_LM

        # Find boundary file
        bound_p = None
        for ext in ['.csv', '.ply']:
            for suffix in [f'{sid}_boundary', sid]:
                c = os.path.join(bound_dir, f'{suffix}{ext}')
                if os.path.exists(c): bound_p = c; break
            if bound_p: break

        # Find landmark file
        lm_p = None
        for ext in ['.csv', '.txt']:
            for suffix in [f'{sid}_landmarks', sid]:
                t = os.path.join(lm_dir, f'{suffix}{ext}')
                if os.path.exists(t): lm_p = t; break
            if lm_p: break

        gt_p  = os.path.join(gt_dir,  f"{label}_gt.ply")
        out_p = os.path.join(out_dir, f"{label}_{tag}.ply")

        miss = []
        if not bound_p:              miss.append('boundary')
        if not lm_p:                 miss.append('landmarks')
        if not os.path.exists(gt_p): miss.append('GT')
        if miss:
            print(f"  ✗ {label:<23} missing: {', '.join(miss)}")
            failed.append(label); continue

        try:
            bp = clean_pts(load_pts(bound_p))
            lm = load_lm(lm_p)
            la = np.zeros((cfg.N_LM, 3), dtype=np.float32)
            for i, name in enumerate(cfg.LM_NAMES):
                if name in lm: la[i] = lm[name]

            # Right-bone: flip X
            if side == 'R':
                bp = bp.copy(); bp[:, 0] *= -1
                la = la.copy(); la[:, 0] *= -1

            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)

            bp_norm = (bp - gc) / gs
            la_norm = (la - gc) / gs
            bp_t = torch.from_numpy(rsample(bp_norm, cfg.N_BOUND)).unsqueeze(0).to(dev)
            la_t = torch.from_numpy(la_norm).unsqueeze(0).to(dev)

            # ── Timed inference ────────────────────────────────────────────────
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            with torch.no_grad():
                gen = model.generate(bp_t, la_t, n_pts=cfg.N_FEMUR,
                                     steps=args.steps, solver=args.solver)

            if dev.type == 'cuda':
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            infer_time = t_end - t_start

            pred_mm = gen[0].cpu().numpy() * gs + gc

            # Right-bone: flip X back
            if side == 'R':
                pred_mm = pred_mm.copy(); pred_mm[:, 0] *= -1

            save_ply(out_p, pred_mm)

            gt_mm = load_ply_pts(gt_p)
            asd, hd, hd95, nsd = compute_metrics(pred_mm, gt_mm, args.nsd_tau)

            results.append((label, asd, hd, hd95, nsd, infer_time))
            times.append(infer_time)
            print(f"{label:<26} {asd:7.3f} {hd95:7.3f} {nsd:7.1f} {infer_time:9.2f}s")

        except Exception as e:
            import traceback
            print(f"  ✗ ERROR {label}: {e}")
            traceback.print_exc()
            failed.append(label)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("─" * 62)
    if not results:
        print("No results."); return

    asds  = np.array([r[1] for r in results])
    hd95s = np.array([r[3] for r in results])
    nsds  = np.array([r[4] for r in results])
    times = np.array(times)

    def row(tag_r, fn):
        print(f"{tag_r:<26} {fn(asds):7.3f} {fn(hd95s):7.3f} {fn(nsds):7.1f} {fn(times):9.2f}s")

    row("MEAN",   np.mean)
    row("STD",    np.std)
    row("MEDIAN", np.median)

    print(f"\n  Solver    : {args.solver.upper()}  steps={args.steps}  NFEs={nfes}")
    print(f"  Evaluated : {len(results)} / {len(TEST_BONES)} bones")
    if failed: print(f"  Skipped   : {failed}")
    print(f"  Avg time  : {np.mean(times):.2f}s per bone")
    print(f"  Total time: {np.sum(times):.1f}s")

    # ── Paper summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"  PAPER TABLE — OT-FM {args.solver.upper()} {args.steps} steps ({nfes} NFEs)")
    print("=" * 62)
    print(f"  ASD   : {np.mean(asds):.3f} ± {np.std(asds):.3f} mm")
    print(f"  HD95  : {np.mean(hd95s):.3f} ± {np.std(hd95s):.3f} mm")
    print(f"  NSD@2 : {np.mean(nsds):.1f} ± {np.std(nsds):.1f} %")
    print(f"  Time  : {np.mean(times):.2f} ± {np.std(times):.2f} s/bone")
    print("=" * 62)

    # ── CSV ────────────────────────────────────────────────────────────────────
    csv_path = os.path.join(cfg.OUTPUT, f"eval_ep{epoch}_{tag}_femur.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Bone', 'ASD_mm', 'HD95_mm', f'NSD@{args.nsd_tau}mm_%', 'InferTime_s'])
        for r in results:
            w.writerow([r[0], f"{r[1]:.3f}", f"{r[3]:.3f}", f"{r[4]:.1f}", f"{r[5]:.3f}"])
        w.writerow([])
        w.writerow(['MEAN', f"{asds.mean():.3f}", f"{hd95s.mean():.3f}",
                    f"{nsds.mean():.1f}", f"{times.mean():.3f}"])
        w.writerow(['STD',  f"{asds.std():.3f}",  f"{hd95s.std():.3f}",
                    f"{nsds.std():.1f}",  f"{times.std():.3f}"])
    print(f"\n  CSV → {csv_path}")


if __name__ == '__main__':
    main()
