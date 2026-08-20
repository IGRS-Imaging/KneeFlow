"""
Tibia OT Flow Matching — Training Script
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
324 bones, 90/10 split, seed=42
Saves test bone IDs to output_Tibia/test_bone_ids.txt on first run.
Saves GT 8192-pt PLYs to output_Tibia/test_bones_gt/ on first run.

Usage:
  python3 train.py                    # fresh start
  python3 train.py --resume latest.pt # resume from checkpoint
"""
import os
# max_split_size_mb REMOVED — it forces tiny 128 MB block splits which
# cause MORE fragmentation with large models.  Just use expandable segments.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = (
    'expandable_segments:True,'
    'garbage_collection_threshold:0.8'
)

import sys, time, argparse, contextlib, gc
import numpy as np
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

mp.set_sharing_strategy('file_system')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cfg
from net import FlowMatch, nparams
from data import (load_all_raw, normalise_raw, split_data,
                  compute_norm_stats, save_norm_stats, load_norm_stats,
                  save_test_ids, sample_mesh_surface, rsample,
                  TibiaDataset, TibiaTestDataset)

NORM_PATH     = os.path.join(cfg.OUTPUT, "norm_stats.json")
TEST_IDS_PATH = os.path.join(cfg.OUTPUT, "test_bone_ids.txt")


# ─────────────────────────────────────────────────────────────────────────────
# GT export (runs once on first training start)
# ─────────────────────────────────────────────────────────────────────────────

def save_ply_pts(path, pts):
    pts = np.asarray(pts, dtype=np.float32)
    with open(path, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in pts:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def export_test_gt(test_raw, gc, gs):
    """
    Save each test bone as 8192-pt PLY (denormalised, mm) and
    write test_bone_ids.txt listing all test bone IDs.
    """
    os.makedirs(cfg.TEST_GT_DIR, exist_ok=True)
    print(f"\n--- Exporting {len(test_raw)} test GT bones ---")

    for s in test_raw:
        pts    = sample_mesh_surface(s['verts'], s['faces'], cfg.N_TIBIA)
        pts_mm = pts * gs + gc       # denormalise → mm
        # Right bones were mirrored (X flip) during loading — flip back for GT
        if s['side'] == 'R':
            pts_mm[:, 0] *= -1
        out_path = os.path.join(cfg.TEST_GT_DIR, f"{s['id']}_gt.ply")
        save_ply_pts(out_path, pts_mm)

    print(f"  GT PLYs saved → {cfg.TEST_GT_DIR}")
    save_test_ids(TEST_IDS_PATH, {s['id'] for s in test_raw})


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_metrics(model, test_loader, gc_arr, gs, dev, nsd_tau=cfg.NSD_TAU):
    """
    Compute ASD, HD, HD95, NSD@2mm on test set.
    Uses only 50 ODE steps (not 200) to reduce memory pressure during eval.
    No test loss computation — avoids full forward pass memory spike.
    """
    model.eval()
    asds, hds, hd95s, nsds = [], [], [], []

    for batch in test_loader:
        bound = batch['bound'].to(dev)
        lm    = batch['lm'].to(dev)
        gt_np = batch['tibia'][0].numpy()

        # 50 steps instead of 200 — 4x less memory, good enough for monitoring
        gen     = model.generate(bound, lm, steps=50)[0].cpu().numpy()
        pred_mm = gen   * gs + gc_arr
        gt_mm   = gt_np * gs + gc_arr

        # Free GPU tensors immediately
        del bound, lm
        torch.cuda.empty_cache()

        P = torch.from_numpy(pred_mm).float().cuda()
        Q = torch.from_numpy(gt_mm  ).float().cuda()

        d_p2g = torch.zeros(P.shape[0], device='cuda')
        for s in range(0, P.shape[0], 512):
            e = min(s + 512, P.shape[0])
            d_p2g[s:e] = torch.cdist(P[s:e], Q).min(1)[0]

        d_g2p = torch.zeros(Q.shape[0], device='cuda')
        for s in range(0, Q.shape[0], 512):
            e = min(s + 512, Q.shape[0])
            d_g2p[s:e] = torch.cdist(Q[s:e], P).min(1)[0]

        d_p2g_np = d_p2g.cpu().numpy()
        d_g2p_np = d_g2p.cpu().numpy()
        del P, Q, d_p2g, d_g2p
        torch.cuda.empty_cache()

        asd  = float((d_p2g_np.mean() + d_g2p_np.mean()) / 2.0)
        hd   = float(max(d_p2g_np.max(), d_g2p_np.max()))
        hd95 = float(max(np.percentile(d_p2g_np, 95), np.percentile(d_g2p_np, 95)))
        nsd  = 100.0 * float(
            (np.mean(d_p2g_np < nsd_tau) + np.mean(d_g2p_np < nsd_tau)) / 2.0)

        asds.append(asd); hds.append(hd); hd95s.append(hd95); nsds.append(nsd)

    return (float(np.mean(asds)), float(np.mean(hds)),
            float(np.mean(hd95s)), float(np.mean(nsds)))


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_ck(path, epoch, model, opt, sched, scaler, best_asd, gc, gs):
    tmp = path + '.tmp'
    torch.save({
        'epoch'   : epoch,
        'model'   : model.state_dict(),
        'opt'     : opt.state_dict(),
        'sched'   : sched.state_dict() if sched else None,
        'scaler'  : scaler.state_dict() if scaler else None,
        'best_asd': best_asd,
        'gc'      : gc,
        'gs'      : gs,
    }, tmp)
    os.replace(tmp, path)


def load_ck(path, model, opt, sched, scaler, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    model.load_state_dict(ck['model'])
    opt.load_state_dict(ck['opt'])
    if sched  and ck.get('sched'):  sched.load_state_dict(ck['sched'])
    if scaler and ck.get('scaler'): scaler.load_state_dict(ck['scaler'])
    return ck['epoch'], ck.get('best_asd', float('inf'))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resume', default=None)
    args = ap.parse_args()

    for d in [cfg.CKPT_DIR, cfg.SAMPLE_DIR, cfg.TEST_GT_DIR]:
        os.makedirs(d, exist_ok=True)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {dev}")

    if dev.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.95)
        torch.set_float32_matmul_precision('high')
        # Disable cudnn auto-tuner — it caches kernels for each input shape and
        # can cause unexpected memory spikes when shapes vary across batches.
        torch.backends.cudnn.benchmark = False

    # ── Load + split data ─────────────────────────────────────────────────────
    print("\n--- Loading data ---")
    raw = load_all_raw()
    if not raw:
        raise RuntimeError("No samples loaded — check paths in cfg.py")

    train_raw, test_raw, test_ids = split_data(raw)

    # ── Norm stats (compute once, reuse on resume) ────────────────────────────
    if os.path.exists(NORM_PATH):
        gc_arr, gs = load_norm_stats(NORM_PATH)
        print(f"  Norm stats loaded from {NORM_PATH}")
    else:
        gc_arr, gs = compute_norm_stats(train_raw)
        save_norm_stats(NORM_PATH, gc_arr, gs)
    print(f"  centroid={np.round(gc_arr,1)}  std={gs:.4f}")

    train_norm = normalise_raw(train_raw, gc_arr, gs)
    test_norm  = normalise_raw(test_raw,  gc_arr, gs)

    # ── Export test GT PLYs (once) ────────────────────────────────────────────
    if not os.path.exists(TEST_IDS_PATH):
        export_test_gt(test_norm, gc_arr, gs)
    else:
        print(f"  Test GT already exported (found {TEST_IDS_PATH})")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_ds = TibiaDataset(train_norm)
    test_ds  = TibiaTestDataset(test_norm)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH, shuffle=True,
                              num_workers=0, pin_memory=False, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = FlowMatch().to(dev)
    print(f"Model params: {nparams(model)/1e6:.2f} M")

    opt   = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WD)
    total = cfg.EPOCHS * len(train_loader)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=1e-6)
    scaler = GradScaler('cuda') if (cfg.AMP and dev.type == 'cuda') else None

    start_epoch = 0
    best_asd    = float('inf')

    resume = args.resume or os.path.join(cfg.CKPT_DIR, 'latest.pt')
    if os.path.exists(resume):
        start_epoch, best_asd = load_ck(resume, model, opt, sched, scaler, dev)
        print(f"Resumed: {resume}  (ep {start_epoch}, best_asd={best_asd:.4f})")
    else:
        print("Starting fresh.")

    print(f"\n{'─'*60}")
    print(f"Epochs {start_epoch+1} → {cfg.EPOCHS}  |  Batch {cfg.BATCH}  |  AMP {cfg.AMP}")
    print(f"{'─'*60}\n")

    for epoch in range(start_epoch + 1, cfg.EPOCHS + 1):
        model.train()
        losses = []
        t0     = time.time()

        for step, batch in enumerate(train_loader):
            tibia = batch['tibia'].to(dev)
            bound = batch['bound'].to(dev)
            lm    = batch['lm'].to(dev)

            loss_val = None
            try:
                opt.zero_grad(set_to_none=True)

                if scaler:
                    with autocast('cuda'):
                        loss, extras = model(tibia, bound, lm)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss, extras = model(tibia, bound, lm)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                    opt.step()

                sched.step()
                loss_val = loss.item()

                # ── Explicit cleanup every step — prevents CUDA OOM segfault ──
                del loss, extras

            except (SystemError, RuntimeError) as e:
                # CUDA memory corruption — skip this step, clear memory, continue
                print(f"\n  [SKIP] ep{epoch} step{step}: {type(e).__name__}: {e}")
                opt.zero_grad(set_to_none=True)
                if scaler:
                    scaler.update()   # reset scaler state
            finally:
                # Always clean up input tensors
                for t in [tibia, bound, lm]:
                    del t
                tibia = bound = lm = None

            if loss_val is not None:
                losses.append(loss_val)

            if step % 5 == 0 and dev.type == 'cuda':
                torch.cuda.empty_cache()

        n_steps    = len(train_loader)
        n_ok       = len(losses)
        train_loss = float(np.mean(losses)) if losses else float('nan')
        lr_now     = opt.param_groups[0]['lr']
        elapsed    = time.time() - t0
        skip_str   = f"  [{n_ok}/{n_steps} steps OK]" if n_ok < n_steps else ""
        print(f"[ep {epoch:04d}/{cfg.EPOCHS}]  "
              f"train={train_loss:.5f}  lr={lr_now:.2e}  "
              f"time={elapsed:.0f}s{skip_str}")

        # ── Aggressive memory cleanup after every epoch ────────────────────────
        gc.collect()
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # ── Save latest + periodic checkpoint ─────────────────────────────────
        save_ck(os.path.join(cfg.CKPT_DIR, 'latest.pt'),
                epoch, model, opt, sched, scaler, best_asd, gc_arr, gs)

        if epoch % cfg.SAVE_EVERY == 0:
            save_ck(os.path.join(cfg.CKPT_DIR, f'ep{epoch}.pt'),
                    epoch, model, opt, sched, scaler, best_asd, gc_arr, gs)

        # ── Full metrics every METRIC_EVERY epochs ────────────────────────────
        if epoch % cfg.METRIC_EVERY == 0:
            asd, hd, hd95, nsd = eval_metrics(
                model, test_loader, gc_arr, gs, dev)

            # Cleanup after eval
            gc.collect()
            if dev.type == 'cuda':
                torch.cuda.empty_cache()

            tag = '  ★ NEW BEST' if asd < best_asd else ''
            print(f"  ├─ ASD      : {asd:.3f} mm{tag}")
            print(f"  ├─ HD       : {hd:.3f} mm")
            print(f"  ├─ HD95     : {hd95:.3f} mm")
            print(f"  └─ NSD@2mm  : {nsd:.1f} %")

            # Append to log CSV
            log_path = os.path.join(cfg.OUTPUT, "train_log.csv")
            write_header = not os.path.exists(log_path)
            with open(log_path, 'a', newline='') as f:
                import csv
                w = csv.writer(f)
                if write_header:
                    w.writerow(['epoch', 'train_loss',
                                'ASD_mm', 'HD_mm', 'HD95_mm', 'NSD@2mm_%'])
                w.writerow([epoch, f"{train_loss:.5f}",
                            f"{asd:.3f}", f"{hd:.3f}",
                            f"{hd95:.3f}", f"{nsd:.1f}"])

            if asd < best_asd:
                best_asd = asd
                save_ck(os.path.join(cfg.CKPT_DIR, 'best_asd.pt'),
                        epoch, model, opt, sched, scaler, best_asd, gc_arr, gs)
            model.train()

        # ── Sample PLY every SAMPLE_EVERY epochs ──────────────────────────────
        if epoch % cfg.SAMPLE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    gen = model.generate(
                        batch['bound'].to(dev),
                        batch['lm'].to(dev))[0].cpu().numpy()
                    save_ply_pts(
                        os.path.join(cfg.SAMPLE_DIR, f'ep{epoch}_sample.ply'),
                        gen * gs + gc_arr)
                    break
            model.train()

    print(f"\n  Training finished.  Best ASD = {best_asd:.4f} mm")
    print(f"  Best checkpoint   : {cfg.CKPT_DIR}/best_asd.pt")
    print(f"  Training log      : {cfg.OUTPUT}/train_log.csv")


if __name__ == '__main__':
    main()
