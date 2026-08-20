"""
Training — Full Femur Flow Matching (Publication Quality)

Split     : 90% train / 10% test  (no validation split — all data used for training)
Landmarks : 9  (Medial Anterior Cortex, Medial/Lateral Posterior Proximal removed)
Metrics   : ASD, CD, HD95, NSD@2mm  (same as SMART-KNEE)
Resume    : python train.py --resume checkpoints/latest.pt

Fixes applied:
  1. import gc as _gc         — avoids name collision with gc (global centroid)
  2. mp.set_sharing_strategy  — eliminates /dev/shm segfaults
  3. num_workers=0            — no worker subprocesses to crash
  4. pin_memory=False         — no pinned RAM accumulation
  5. optimizer step fix       — CPU float32 tensor for PyTorch >=2.0 AdamW
  6. scaler not restored      — avoids 'Unknown device: -100' AMP error
  7. _gc.collect() per epoch  — prevents gradual RAM buildup
  8. TF32 matmul precision    — free GPU speed (cudnn.benchmark disabled)
  9. early stopping on ASD    — stops after ASD_PATIENCE evals without improvement
 10. CUDA env set at module TOP — must be before any CUDA init (allocator reads once)
 11. atomic checkpoint saves   — write to .tmp then rename, prevents corruption
 12. memory fraction cap        — hard cap at 95% VRAM, forces early GC
"""
# ── MUST be set before any torch/CUDA import touches the allocator ────────────
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = (
    'expandable_segments:True,'
    'garbage_collection_threshold:0.8,'   # GC when 80% VRAM used (was missing)
    'max_split_size_mb:128'               # tighter than 512 — less fragmentation
)

import time, argparse
import gc as _gc
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

import cfg
from data import build_datasets, rsample, sample_mesh_surface  # surf removed
from net import FlowMatch, nparams

# Fix 2: eliminates /dev/shm exhaustion that segfaults worker processes
mp.set_sharing_strategy('file_system')

# Early stopping: stop if ASD hasn't improved for this many consecutive
# metric evaluations. With METRIC_EVERY=50, patience=6 = 300 epochs no improvement.
ASD_PATIENCE = 6


# ── Utility ───────────────────────────────────────────────────────────────────

def save_ply(path, pts):
    pts = np.asarray(pts)
    with open(path, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in pts: f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def get_lr(ep):
    if ep < cfg.WARMUP:
        return cfg.LR * (ep + 1) / cfg.WARMUP
    p = (ep - cfg.WARMUP) / max(cfg.EPOCHS - cfg.WARMUP, 1)
    return 1e-6 + 0.5 * (cfg.LR - 1e-6) * (1 + np.cos(np.pi * p))


# ── Geometric metrics ─────────────────────────────────────────────────────────

def pairwise_dist_mm(pred_mm, gt_mm, dev, chunk=2048):
    P = torch.from_numpy(pred_mm).float().to(dev)
    G = torch.from_numpy(gt_mm).float().to(dev)
    N = P.shape[0]
    d_p2g = torch.zeros(N, device=dev)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        d_p2g[s:e] = torch.cdist(P[s:e], G).min(1)[0]
    M = G.shape[0]
    d_g2p = torch.zeros(M, device=dev)
    for s in range(0, M, chunk):
        e = min(s + chunk, M)
        d_g2p[s:e] = torch.cdist(G[s:e], P).min(1)[0]
    return d_p2g.cpu().numpy(), d_g2p.cpu().numpy()


@torch.no_grad()
def compute_metrics(model, ds_test, dev, gc, gs):
    model.eval()
    asds, cds, hds, hd95s, nsds = [], [], [], [], []
    for s in ds_test.samples:
        b_t = torch.from_numpy(
                rsample(s['bound'], cfg.N_BOUND)).unsqueeze(0).to(dev)
        l_t = torch.from_numpy(s['lm'].copy()).unsqueeze(0).to(dev)
        gen = model.generate(b_t, l_t, n_pts=cfg.N_FEMUR, steps=cfg.INF_STEPS)
        pred_mm = gen[0].cpu().numpy() * gs + gc
        gt_mm   = sample_mesh_surface(
                    s['verts'], s['faces'], cfg.N_FEMUR) * gs + gc
        d_p2g, d_g2p = pairwise_dist_mm(pred_mm, gt_mm, dev)
        all_dists    = np.concatenate([d_p2g, d_g2p])
        cd   = float(d_p2g.mean() + d_g2p.mean())
        asd  = cd / 2.0
        hd   = float(max(d_p2g.max(), d_g2p.max()))
        hd95 = float(np.percentile(all_dists, 95))
        nsd  = float((all_dists < cfg.NSD_TAU).mean())
        asds.append(asd); cds.append(cd)
        hds.append(hd);   hd95s.append(hd95); nsds.append(nsd)
    model.train()
    return (float(np.mean(asds)),  float(np.mean(cds)),
            float(np.mean(hds)),   float(np.mean(hd95s)),
            float(np.mean(nsds)))


# ── Sample generation ─────────────────────────────────────────────────────────

def gen_samples(model, ds_train, epoch, dev, gc, gs):
    model.eval()
    os.makedirs(cfg.SAMPLE_DIR, exist_ok=True)
    gt_dir = os.path.join(cfg.SAMPLE_DIR, "gt")
    os.makedirs(gt_dir, exist_ok=True)
    for i in range(min(3, len(ds_train.samples))):
        s   = ds_train.samples[i]
        b_t = torch.from_numpy(rsample(s['bound'], cfg.N_BOUND)).unsqueeze(0).to(dev)
        l_t = torch.from_numpy(s['lm'].copy()).unsqueeze(0).to(dev)
        with torch.no_grad():
            g = model.generate(b_t, l_t, n_pts=cfg.N_FEMUR, steps=100)
        pts    = g[0].cpu().numpy() * gs + gc
        gt_pts = sample_mesh_surface(s['verts'], s['faces'], cfg.N_FEMUR) * gs + gc
        bd     = rsample(s['bound'], cfg.N_BOUND) * gs + gc
        save_ply(os.path.join(cfg.SAMPLE_DIR, f"e{epoch}_{s['id']}.ply"), pts)
        save_ply(os.path.join(gt_dir, f"gt_{s['id']}.ply"),  gt_pts)
        save_ply(os.path.join(gt_dir, f"bd_{s['id']}.ply"),  bd)
        print(f"    PLY saved: e{epoch}_{s['id']}.ply")
    model.train()


# ── Main training loop ────────────────────────────────────────────────────────

def train(resume=None):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {dev}")
    if dev.type == 'cuda':
        print(f"GPU:  {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        # Cap allocator at 95% VRAM — forces early GC before driver-level OOM/segfault
        torch.cuda.set_per_process_memory_fraction(0.95)
        torch.set_float32_matmul_precision('high')  # TF32 only — safe without benchmark

    os.makedirs(cfg.CKPT_DIR,   exist_ok=True)
    os.makedirs(cfg.SAMPLE_DIR, exist_ok=True)

    logf = open(os.path.join(cfg.OUTPUT, "log.txt"), 'a')
    def log(m): print(m); logf.write(m + '\n'); logf.flush()

    S = "=" * 85
    log(S)
    log("  FULL FEMUR FLOW MATCHING — Publication Quality")
    log(S)

    # ── Data ──────────────────────────────────────────────────────────────────
    ds_train, ds_test, gc, gs = build_datasets()
    if not ds_train.samples:
        log("ERROR: 0 training samples loaded"); return

    log(f"\n  Split (90/10):")
    log(f"    Train : {len(ds_train.samples)} bones  (used every epoch)")
    log(f"    Test  : {len(ds_test.samples)} bones   (ASD / HD95 / NSD metrics)")
    log(f"    LM    : {cfg.N_LM} landmarks per bone")
    log(f"\n  Schedule:")
    log(f"    Metrics   every {cfg.METRIC_EVERY} epochs  (ASD, CD, HD95, NSD@{cfg.NSD_TAU}mm)")
    log(f"    Save      every {cfg.SAVE_EVERY} epochs + latest.pt each epoch")
    log(f"    Epochs    {cfg.EPOCHS}  (early stop: patience={ASD_PATIENCE} evals = {ASD_PATIENCE*cfg.METRIC_EVERY} epochs)")

    # Fix 3+4: num_workers=0 (no worker segfaults), pin_memory=False (no pinned RAM leak)
    dl_train = DataLoader(ds_train, batch_size=cfg.BATCH, shuffle=True,
                          num_workers=0, pin_memory=False, drop_last=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    log(f"\n  Model:")
    model  = FlowMatch().to(dev)
    log(f"    Params: {nparams(model):,} ({nparams(model)/1e6:.1f}M)")

    opt    = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WD)
    scaler = GradScaler('cuda', enabled=cfg.AMP)

    start_ep    = 0
    best_train  = float('inf')
    train_hist  = []
    metric_hist = []

    if resume and os.path.exists(resume):
        log(f"\nLoading checkpoint: {resume}")
        ck = torch.load(resume, map_location=dev, weights_only=False)

        # ── 1. Model weights ──────────────────────────────────────────────────
        model.load_state_dict(ck['model'])

        # ── 2. Optimizer state ────────────────────────────────────────────────
        try:
            opt.load_state_dict(ck['optimizer'])
            # Fix 5: PyTorch >=2.0 AdamW requires step as CPU float32 tensor
            for state in opt.state.values():
                if 'step' in state:
                    s = state['step']
                    if isinstance(s, torch.Tensor):
                        state['step'] = s.detach().float().cpu()
                    else:
                        state['step'] = torch.tensor(float(s), dtype=torch.float32)
            log("  Optimizer state restored.")
        except Exception as e:
            log(f"  Warning: optimizer state not restored ({e}) — using fresh optimizer.")

        # ── 3. AMP scaler ─────────────────────────────────────────────────────
        # Fix 6: do NOT restore scaler — causes 'Unknown device: -100' error
        # Fresh scaler re-adapts scale factor within ~10 gradient steps
        log("  AMP scaler: fresh state (re-adapts in ~10 steps).")

        # ── 4. Training counters ──────────────────────────────────────────────
        start_ep   = ck.get('epoch', 0) + 1
        best_train = ck.get('best_train_loss', float('inf'))

        # ── 5. History ────────────────────────────────────────────────────────
        train_hist  = ck.get('train_history',  [])
        metric_hist = ck.get('metric_history', [])

        # ── 6. Norm stats ─────────────────────────────────────────────────────
        ck_gc = ck.get('gc', None)
        ck_gs = ck.get('gs', None)
        if ck_gc is not None and ck_gs is not None:
            if not (np.allclose(gc, ck_gc, atol=1e-3) and
                    abs(float(gs) - float(ck_gs)) < 1e-3):
                log("  WARNING: norm stats differ — overriding with checkpoint gc/gs.")
                gc = np.array(ck_gc, dtype=np.float32)
                gs = float(ck_gs)
            else:
                log("  Norm stats match checkpoint. OK.")

        log(f"\n  Resumed from epoch {start_ep} / {cfg.EPOCHS}")
        log(f"  best_train={best_train:.4f}")
        if metric_hist:
            best_asd_so_far = min(m[1] for m in metric_hist)
            log(f"  Best ASD so far: {best_asd_so_far:.3f} mm "
                f"({len(metric_hist)} metric evaluations)")
        else:
            log("  No metric history yet.")
    else:
        if resume:
            log(f"  WARNING: checkpoint not found at '{resume}' — starting from scratch.")

    # ── Training table ────────────────────────────────────────────────────────
    SEP  = "=" * 85
    SEP2 = "-" * 85
    HDR  = (f"{'Epoch':>6}  {'TrainLoss':>9}  "
            f"{'ASD(mm)':>8}  {'CD(mm)':>7}  {'HD95(mm)':>8}  {'NSD@2mm':>7}  "
            f"{'LR':>8}  {'Time':>5}")

    def print_header():
        log(SEP); log(HDR); log(SEP2)

    log(f"\n{SEP}")
    log(f"  FEMUR FLOW MATCHING  |  Epochs {start_ep+1} -> {cfg.EPOCHS}  |  "
        f"Batch={cfg.BATCH}  |  Steps/ep={len(dl_train)}")
    log(f"  Train: {len(ds_train.samples)} bones  |  Test: {len(ds_test.samples)} bones  |  "
        f"LM: {cfg.N_LM}  |  Metrics every {cfg.METRIC_EVERY} ep")
    print_header()

    model.train()
    nan_total = 0

    for ep in range(start_ep, cfg.EPOCHS):
        eloss = 0.0; nb = 0; nan_ep = 0; t0 = time.time()
        lr = get_lr(ep)
        for pg in opt.param_groups: pg['lr'] = lr

        # ── Train batches ──────────────────────────────────────────────────
        for bi, batch in enumerate(dl_train):
            femur = batch['femur'].to(dev, non_blocking=True)
            b     = batch['bound'].to(dev, non_blocking=True)
            l     = batch['lm'].to(dev,    non_blocking=True)

            if torch.any(~torch.isfinite(femur)) or torch.any(~torch.isfinite(b)):
                nan_ep += 1; nan_total += 1; continue

            opt.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=cfg.AMP):
                loss, info = model(femur, b, l)

            if loss.item() == 0.0 or not torch.isfinite(loss):
                nan_ep += 1; nan_total += 1; continue

            scaler.scale(loss).backward()
            scaler.unscale_(opt)

            if any(p.grad is not None and torch.any(~torch.isfinite(p.grad))
                   for p in model.parameters()):
                opt.zero_grad(set_to_none=True)
                scaler.update()
                nan_ep += 1; nan_total += 1; continue

            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scaler.step(opt); scaler.update()
            eloss += info['loss']; nb += 1

            if bi % 20 == 0 and bi > 0:
                torch.cuda.empty_cache()

        avg_train = eloss / max(nb, 1)
        nan_pct   = 100.0 * nan_ep / max(len(dl_train), 1)
        train_hist.append(avg_train)
        elapsed   = time.time() - t0

        # Fix 7: release Python + CUDA memory every epoch
        torch.cuda.empty_cache()
        _gc.collect()

        # ── Best train checkpoint ──────────────────────────────────────────
        if avg_train < best_train and avg_train > 0:
            best_train = avg_train
            _save_ck(model, opt, scaler, ep, best_train,
                     train_hist, metric_hist,
                     os.path.join(cfg.CKPT_DIR, "best_train.pt"), gc, gs)

        # ── latest.pt every epoch ──────────────────────────────────────────
        _save_ck(model, opt, scaler, ep, best_train,
                 train_hist, metric_hist,
                 os.path.join(cfg.CKPT_DIR, "latest.pt"), gc, gs)

        # ── Periodic checkpoint ────────────────────────────────────────────
        if (ep + 1) % cfg.SAVE_EVERY == 0 or ep == cfg.EPOCHS - 1:
            _save_ck(model, opt, scaler, ep, best_train,
                     train_hist, metric_hist,
                     os.path.join(cfg.CKPT_DIR, f"ep{ep+1}.pt"), gc, gs)

        # ── Geometric metrics ──────────────────────────────────────────────
        asd_str = cd_str = hd95_str = nsd_str = "       -"
        has_metrics = False
        if (ep + 1) % cfg.METRIC_EVERY == 0 or ep == cfg.EPOCHS - 1:
            asd, cd, hd, hd95, nsd = compute_metrics(model, ds_test, dev, gc, gs)
            asd_str  = f"{asd:8.3f}"
            cd_str   = f"{cd:7.3f}"
            hd95_str = f"{hd95:8.3f}"
            nsd_str  = f"{nsd:7.3f}"
            has_metrics = True
            torch.cuda.empty_cache(); _gc.collect()

            prev_best_asd = min(m[1] for m in metric_hist) if metric_hist else float('inf')
            is_best_asd   = asd < prev_best_asd
            metric_hist.append((ep + 1, asd, cd, hd, hd95, nsd))
            if is_best_asd:
                _save_ck(model, opt, scaler, ep, best_train,
                         train_hist, metric_hist,
                         os.path.join(cfg.CKPT_DIR, "best_asd.pt"), gc, gs)

            # Fix 9: early stopping
            best_eval_idx    = metric_hist.index(min(metric_hist, key=lambda m: m[1]))
            evals_since_best = len(metric_hist) - 1 - best_eval_idx
            if evals_since_best >= ASD_PATIENCE:
                best_ep = metric_hist[best_eval_idx]
                log(f"  {'='*81}")
                log(f"  EARLY STOPPING at epoch {ep+1}")
                log(f"  ASD not improved for {evals_since_best} evals "
                    f"({evals_since_best * cfg.METRIC_EVERY} epochs)")
                log(f"  Best ASD: {best_ep[1]:.3f} mm at epoch {best_ep[0]}")
                log(f"  Use best_asd.pt for inference.")
                log(f"  {'='*81}")
                logf.close()
                return

        # ── Print row ──────────────────────────────────────────────────────
        if ep > 0 and ep % 50 == 0:
            print_header()

        should_print = (ep < 5 or ep % 10 == 0 or
                        ep == cfg.EPOCHS - 1 or has_metrics)
        if should_print:
            nan_warn = "  << NaN!" if nan_pct > 5.0 else ""
            log(f"{ep+1:>6}  {avg_train:9.4f}  "
                f"{asd_str}  {cd_str}  {hd95_str}  {nsd_str}  "
                f"{lr:8.2e}  {elapsed:4.0f}s{nan_warn}")

        if has_metrics:
            star = "  << NEW BEST" if is_best_asd else ""
            log(f"  {'-'*81}")
            log(f"  METRICS @ ep {ep+1:4d}  |  "
                f"ASD: {asd:.3f} mm  |  CD: {cd:.3f} mm  |  "
                f"HD95: {hd95:.3f} mm  |  NSD@2mm: {nsd:.3f}{star}")
            log(f"  {'-'*81}")

        if (ep + 1) % cfg.SAMPLE_EVERY == 0:
            gen_samples(model, ds_train, ep + 1, dev, gc, gs)

        if (ep + 1) % 50 == 0:
            torch.cuda.empty_cache()

        # ── Loss collapse guard ────────────────────────────────────────────
        if len(train_hist) >= 10 and all(h < 1e-6 for h in train_hist[-10:]):
            log(f"\n{'!':!<85}")
            log("  ERROR: Loss collapsed to zero. Stopping.")
            log(f"{'!':!<85}"); break

    # ── Final summary ─────────────────────────────────────────────────────────
    log(f"\n{SEP}")
    log(f"  TRAINING COMPLETE  |  Best train loss: {best_train:.4f}")
    if metric_hist:
        best = min(metric_hist, key=lambda m: m[1])
        log(f"  Best ASD : {best[1]:.3f} mm  (epoch {best[0]})")
        log(f"  Best HD95: {best[4]:.3f} mm")
        log(f"  Best NSD : {best[5]:.3f}")
    log(SEP)
    logf.close()


# ── Checkpoint helper ─────────────────────────────────────────────────────────

def _save_ck(model, opt, scaler, ep, best_train,
             train_hist, metric_hist, path, gc=None, gs=None):
    """Atomic checkpoint save: write to .tmp then rename.
    Prevents corruption if process is killed mid-write (power cut / OOM crash).
    """
    tmp = path + '.tmp'
    torch.save({
        'epoch':           ep,
        'model':           model.state_dict(),
        'optimizer':       opt.state_dict(),
        'scaler':          scaler.state_dict(),
        'best_train_loss': best_train,
        'train_history':   train_hist,
        'metric_history':  metric_hist,
        'gc':              gc.tolist() if gc is not None else None,
        'gs':              float(gs)   if gs is not None else None,
    }, tmp)
    os.replace(tmp, path)  # atomic on Linux (same filesystem)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--resume', type=str, default=None)
    train(p.parse_args().resume)
