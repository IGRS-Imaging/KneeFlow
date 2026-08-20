"""
Test-Time Refinement for Femur Generation
Works with original net.py (no cross-attention)
"""
import argparse, os
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
        for p in pts: f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def chamfer(p1, p2):
    d = torch.cdist(p1, p2)
    return d.min(2)[0].mean() + d.min(1)[0].mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bound',        required=True)
    p.add_argument('--lm',           required=True)
    p.add_argument('--ckpt',         required=True)
    p.add_argument('--out',          required=True)
    p.add_argument('--npts',         type=int,   default=cfg.N_FEMUR)
    p.add_argument('--steps',        type=int,   default=100)
    p.add_argument('--refine_steps', type=int,   default=100)
    p.add_argument('--refine_lr',    type=float, default=0.01)
    p.add_argument('--reg_weight',   type=float, default=0.1)
    p.add_argument('--norm_stats',   default=None, help='Path to norm_stats.json (overrides cfg.OUTPUT)')
    a = p.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {dev}")

    # Load model — freeze all weights
    model = FlowMatch().to(dev)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(ck['model'])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print(f"Loaded: epoch={ck['epoch']}, loss={ck['best_loss']:.4f}")

    # Load norm stats
    gc, gs = load_norm_stats(a.norm_stats)
    print(f"Norm std={gs:.4f}")

    # Load inputs
    bp = clean_pts(load_pts(a.bound))
    bp_norm = (bp - gc) / gs
    bound_t = torch.from_numpy(rsample(bp_norm, cfg.N_BOUND)).unsqueeze(0).to(dev)

    lm = load_lm(a.lm)
    la = np.zeros((cfg.N_LM, 3), dtype=np.float32)
    for i, n in enumerate(cfg.LM_NAMES):
        if n in lm: la[i] = lm[n]
    la_norm = (la - gc) / gs
    lm_t = torch.from_numpy(la_norm).unsqueeze(0).to(dev)

    # Encode condition once — frozen
    cond = model.encode(bound_t, lm_t)  # (1, COND_DIM)

    # Step 1: Standard Heun generation
    print(f"Generating {a.npts} points with {a.steps} steps...")
    z0 = torch.randn(1, a.npts, 3, device=dev)
    x = z0.clone()
    dt = 1.0 / a.steps
    with torch.no_grad():
        for i in range(a.steps):
            t  = torch.full((1,), i * dt,       device=dev)
            t2 = torch.full((1,), (i+1) * dt,   device=dev)
            v1 = model.vf(x, t, cond)
            x2 = x + v1 * dt
            v2 = model.vf(x2, t2, cond)
            x  = x + 0.5 * (v1 + v2) * dt
    initial = x.clone()
    print("Initial generation done.")

    # Step 2: Optimize delta_z
    print(f"Refining with {a.refine_steps} steps (lr={a.refine_lr})...")
    delta_z = torch.zeros_like(z0, requires_grad=True)
    opt = torch.optim.Adam([delta_z], lr=a.refine_lr)
    best_loss = float('inf')
    best_x = initial.clone()

    for step in range(a.refine_steps):
        opt.zero_grad()
        z = z0 + delta_z
        x = z
        for i in range(a.steps):
            t  = torch.full((1,), i * dt,       device=dev)
            t2 = torch.full((1,), (i+1) * dt,   device=dev)
            v1 = model.vf(x, t, cond)
            x2 = (x + v1 * dt).detach()
            v2 = model.vf(x2, t2, cond)
            x  = x + 0.5 * (v1 + v2) * dt

        # Boundary alignment loss
        loss_bd  = chamfer(x, bound_t)
        # Regularization — keep z close to prior
        loss_reg = (delta_z ** 2).mean()
        # Landmark loss
        lm_pts = lm_t[0].unsqueeze(0)
        loss_lm = torch.cdist(lm_pts, x).min(2)[0].mean()

        loss = loss_bd + a.reg_weight * loss_reg + 0.5 * loss_lm
        loss.backward()
        torch.nn.utils.clip_grad_norm_([delta_z], 1.0)
        opt.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_x = x.detach().clone()

        if step % 20 == 0 or step == a.refine_steps - 1:
            print(f"  Step {step+1:3d}/{a.refine_steps} | "
                  f"loss={loss.item():.4f} "
                  f"(bd={loss_bd.item():.4f} "
                  f"reg={loss_reg.item():.4f} "
                  f"lm={loss_lm.item():.4f})")

    # Save both
    init_pts    = initial[0].cpu().numpy() * gs + gc
    refined_pts = best_x[0].cpu().numpy()  * gs + gc

    out_init = a.out.replace('.ply', '_initial.ply')
    save_ply(out_init, init_pts)
    save_ply(a.out,    refined_pts)

    print(f"\nSaved initial → {out_init}")
    print(f"Saved refined → {a.out}")
    print(f"Now measure CD/HD of both against GT.")


if __name__ == "__main__":
    main()
