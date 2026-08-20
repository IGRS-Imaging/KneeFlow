"""
Tibia OT Flow Matching — Inference
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input  : 8 tibia landmarks (CSV) + tibial boundary points (PLY)
Output : full tibia point cloud (PLY)

Usage:
  python infer.py --bound boundary.ply --lm landmarks.csv --out output.ply
  python infer.py --bound boundary.ply --lm landmarks.csv --out output.ply --side R
  python infer.py --bound boundary.ply --lm landmarks.csv --out output.ply --steps 100

Landmark CSV format expected (same as training):
  Landmark,X,Y,Z
  Tibial Knee Centre,x,y,z
  Medial Plateau,x,y,z
  Lateral Plateau,x,y,z
  Tibial Tuberosity,x,y,z
  Medial Malleolus,x,y,z
  Lateral Malleolus,x,y,z
  Ankle Centre,x,y,z
  Posterior Cruciate Ligament,x,y,z

NOTE: The model was trained with right-side bones mirrored to left-side space.
      Pass --side R if your bone is a right tibia — inputs will be X-flipped
      before inference and the output will be flipped back automatically.
"""
import os, sys, argparse, csv, json
import numpy as np
import torch

# ── Make sure imports find cfg / net / data in the same folder ───────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cfg
from net import FlowMatch, nparams

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_boundary_ply(path):
    """Load boundary PLY → (N,3) float32 array."""
    if HAS_O3D:
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points, dtype=np.float32)
        if len(pts) > 0:
            return pts
        # Fallback: try as mesh
        mesh  = o3d.io.read_triangle_mesh(path)
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        return verts
    else:
        # Manual ASCII/binary PLY parser (no open3d dependency)
        return _load_ply_manual(path)


def _load_ply_manual(path):
    """Minimal PLY parser for point clouds (fallback when open3d absent)."""
    with open(path, 'rb') as f:
        nv = 0; fmt = 'ascii'; props = []
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            if 'binary_little_endian' in line:
                fmt = 'bin'
            elif line.startswith('element vertex'):
                nv = int(line.split()[-1])
            elif line.startswith('property') and 'list' not in line:
                props.append(line.split()[1])
            elif line == 'end_header':
                break

        if fmt == 'bin':
            type_sizes  = {'float':4,'float32':4,'double':8,'float64':8,
                           'int':4,'int32':4,'uint8':1,'uchar':1,'short':2}
            type_dtypes = {'float':'<f4','float32':'<f4','double':'<f8','float64':'<f8',
                           'int':'<i4','int32':'<i4','uint8':'<u1','uchar':'<u1','short':'<i2'}
            bpv  = sum(type_sizes.get(p, 4) for p in props)
            raw  = f.read(nv * bpv)
            dt   = np.dtype([(f'f{i}', type_dtypes.get(p, '<f4')) for i, p in enumerate(props)])
            data = np.frombuffer(raw[:nv * dt.itemsize], dtype=dt)
            verts = np.column_stack([data['f0'].astype(np.float32),
                                     data['f1'].astype(np.float32),
                                     data['f2'].astype(np.float32)])
        else:
            verts = np.zeros((nv, 3), dtype=np.float32)
            for i in range(nv):
                vals = f.readline().decode('ascii', errors='ignore').strip().split()
                if len(vals) >= 3:
                    verts[i] = [float(vals[0]), float(vals[1]), float(vals[2])]
    return verts


def load_landmarks_csv(path):
    """
    Load landmark CSV → dict {name: np.array([x,y,z])}.
    Accepts headers: Landmark,X,Y,Z  or  landmark,x,y,z
    """
    lm = {}
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get('Landmark', row.get('landmark', '')).strip()
            if not name:
                continue
            try:
                x = float(row.get('X', row.get('x', 0)))
                y = float(row.get('Y', row.get('y', 0)))
                z = float(row.get('Z', row.get('z', 0)))
                lm[name] = np.array([x, y, z], dtype=np.float32)
            except (ValueError, KeyError):
                continue
    return lm


def load_norm_stats(path):
    """Load norm_stats.json saved by train.py."""
    with open(path) as f:
        d = json.load(f)
    gc = np.array(d['centroid'], dtype=np.float32)
    gs = float(d['scale'])
    return gc, gs


def clean_pts(pts):
    """Remove NaN/Inf rows."""
    mask = np.isfinite(pts).all(axis=1)
    return pts[mask].astype(np.float32)


def rsample(pts, n):
    """Random subsample or oversample to exactly n points."""
    N = len(pts)
    idx = np.random.choice(N, n, replace=(N < n))
    return pts[idx].astype(np.float32)


def save_ply(path, pts):
    """Save point cloud as ASCII PLY."""
    pts = np.asarray(pts, dtype=np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in pts:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    print(f"  Saved {len(pts)} points → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Default paths (Windows local copies)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CKPT  = r"C:\Users\hticl\Downloads\9\Tibia\output_Tibia\checkpoints\ep800.pt"
DEFAULT_NORM  = r"C:\Users\hticl\Downloads\9\Tibia\output_Tibia\norm_stats.json"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Generate a tibia point cloud from 8 landmarks + boundary points.")
    ap.add_argument('--bound',      required=True,
                    help='Tibial boundary points PLY file')
    ap.add_argument('--lm',         required=True,
                    help='Landmarks CSV file  (Landmark,X,Y,Z)')
    ap.add_argument('--out',        required=True,
                    help='Output PLY path  e.g. output/tibia_gen.ply')
    ap.add_argument('--ckpt',       default=DEFAULT_CKPT,
                    help='Checkpoint .pt file')
    ap.add_argument('--norm_stats', default=DEFAULT_NORM,
                    help='norm_stats.json path')
    ap.add_argument('--side',       default='L', choices=['L', 'R'],
                    help='L (default) or R — right bones are X-flipped automatically')
    ap.add_argument('--npts',       type=int, default=cfg.N_TIBIA,
                    help=f'Number of output points (default {cfg.N_TIBIA})')
    ap.add_argument('--steps',      type=int, default=cfg.INF_STEPS,
                    help=f'ODE solver steps (default {cfg.INF_STEPS}; use 50 for speed)')
    ap.add_argument('--solver',     default='heun', choices=['heun', 'euler'],
                    help='ODE solver: heun (2nd-order, default) or euler (faster)')
    ap.add_argument('--seed',       type=int, default=42)
    args = ap.parse_args()

    # ── Device ───────────────────────────────────────────────────────────────
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice     : {dev}")
    if dev.type == 'cuda':
        print(f"GPU        : {torch.cuda.get_device_name()}")

    # ── Reproducibility ───────────────────────────────────────────────────────
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if dev.type == 'cuda':
        torch.cuda.manual_seed(args.seed)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Checkpoint : {args.ckpt}")
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    model = FlowMatch().to(dev)
    ck    = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(ck['model'])
    model.eval()
    print(f"Loaded     : epoch {ck.get('epoch', '?')}  "
          f"| best_asd={ck.get('best_asd', float('nan')):.4f} mm")
    print(f"Params     : {nparams(model)/1e6:.2f} M")

    # ── Norm stats ────────────────────────────────────────────────────────────
    print(f"Norm stats : {args.norm_stats}")
    if not os.path.exists(args.norm_stats):
        raise FileNotFoundError(f"norm_stats.json not found: {args.norm_stats}")
    gc, gs = load_norm_stats(args.norm_stats)
    print(f"           : centroid={np.round(gc,1)}  scale={gs:.4f}")

    # ── Load boundary ─────────────────────────────────────────────────────────
    print(f"\nBoundary   : {args.bound}")
    if not os.path.exists(args.bound):
        raise FileNotFoundError(f"Boundary file not found: {args.bound}")
    bp = clean_pts(load_boundary_ply(args.bound))
    if len(bp) == 0:
        raise ValueError("No valid points found in boundary PLY.")
    print(f"           : {len(bp)} points loaded")

    # ── Load landmarks ────────────────────────────────────────────────────────
    print(f"Landmarks  : {args.lm}")
    if not os.path.exists(args.lm):
        raise FileNotFoundError(f"Landmark file not found: {args.lm}")
    lm_dict = load_landmarks_csv(args.lm)

    la = np.zeros((cfg.N_LM, 3), dtype=np.float32)
    print(f"\n  Landmark check ({cfg.N_LM} expected):")
    missing = []
    for i, name in enumerate(cfg.LM_NAMES):
        if name in lm_dict:
            la[i] = lm_dict[name]
            print(f"    [{i}] {name:<35} {la[i]}")
        else:
            missing.append(name)
            print(f"    [{i}] {name:<35} *** MISSING (set to 0,0,0) ***")

    if len(missing) > 4:
        raise ValueError(
            f"Too many landmarks missing ({len(missing)}/{cfg.N_LM}). "
            f"Check your CSV column names match exactly:\n  {cfg.LM_NAMES}")

    # ── Normalise ─────────────────────────────────────────────────────────────
    bp_norm = (bp - gc) / gs
    la_norm = (la - gc) / gs

    # Right-side bones: X-flip to match training convention, flip output back
    is_right = (args.side == 'R')
    if is_right:
        print(f"\n  Side=R → X-flipping inputs to left-side space")
        bp_norm = bp_norm.copy(); bp_norm[:, 0] *= -1
        la_norm = la_norm.copy(); la_norm[:, 0] *= -1

    # ── Build tensors ─────────────────────────────────────────────────────────
    bp_t = torch.from_numpy(rsample(bp_norm, cfg.N_BOUND)).unsqueeze(0).to(dev)
    la_t = torch.from_numpy(la_norm).unsqueeze(0).to(dev)

    # ── Generate ──────────────────────────────────────────────────────────────
    print(f"\nGenerating : {args.npts} points  "
          f"| solver={args.solver.upper()}  steps={args.steps}  "
          f"| NFEs={args.steps if args.solver == 'euler' else args.steps * 2}")

    with torch.no_grad():
        gen = model.generate(bp_t, la_t,
                             n_pts=args.npts,
                             steps=args.steps,
                             solver=args.solver)

    pts_norm = gen[0].cpu().numpy()            # (N, 3) normalised
    pts_mm   = pts_norm * gs + gc              # denormalise → mm

    # Flip output back for right-side bones
    if is_right:
        pts_mm[:, 0] *= -1
        print("  X-flipped output back to right-side space")

    # ── Save ──────────────────────────────────────────────────────────────────
    save_ply(args.out, pts_mm)
    print("\nDone.")


if __name__ == '__main__':
    main()
