"""
Visualize per-point error between generated and GT femur as colored PLY.
Colors: green (< 2mm) → yellow (2-4mm) → orange (4-6mm) → red (> 6mm)
Opens nicely in MeshLab or CloudCompare.

Usage:
  python visualize_errors.py --pred test_infer/L_244_Femur_L_generated.ply \
                              --gt   test_bones_gt/L_244_Femur_L_gt.ply \
                              --out  error_viz/L_244_error.ply

  # Batch: generate error PLYs for all evaluated bones
  python visualize_errors.py --batch
"""
import os, sys, argparse
import numpy as np
import torch

BASE      = "/home/imaging/Desktop/Ragavan"
GT_DIR    = os.path.join(BASE, "9L_working_output", "test_bones_gt")
PRED_DIR  = os.path.join(BASE, "9L_working_output", "test_infer")
VIZ_DIR   = os.path.join(BASE, "9L_working_output", "error_viz")

# All 29 test bones (selected_29_dice78_82)
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


def load_ply_pts(path):
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
                    pts.append([float(v) for v in vals[:3]])
                except ValueError:
                    continue
    return np.array(pts, dtype=np.float32)


def error_to_color(dist_mm):
    """
    Map per-point distance (mm) to RGB colour.
    < 2mm   : green (0,200,0)
    2-4mm   : yellow-green → yellow
    4-6mm   : yellow → orange
    6-10mm  : orange → red
    > 10mm  : dark red (180,0,0)
    """
    colors = np.zeros((len(dist_mm), 3), dtype=np.uint8)
    for i, d in enumerate(dist_mm):
        if d < 2.0:
            colors[i] = [0, 200, 0]
        elif d < 4.0:
            t = (d - 2.0) / 2.0  # 0→1
            colors[i] = [int(255*t), int(200 - 130*t), 0]
        elif d < 6.0:
            t = (d - 4.0) / 2.0
            colors[i] = [255, int(70 - 70*t), 0]
        elif d < 10.0:
            t = (d - 6.0) / 4.0
            colors[i] = [int(255 - 75*t), 0, 0]
        else:
            colors[i] = [180, 0, 0]
    return colors


def save_colored_ply(path, pts, colors):
    """Save PLY with per-vertex RGB color."""
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(pts, colors):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} "
                    f"{c[0]} {c[1]} {c[2]}\n")


def make_error_ply(pred_path, gt_path, out_path, chunk=2048):
    """Generate error-colored PLY for predicted points (nearest GT distance)."""
    pred = load_ply_pts(pred_path)
    gt   = load_ply_pts(gt_path)

    P = torch.from_numpy(pred).float().cuda()
    G = torch.from_numpy(gt).float().cuda()

    d_p2g = torch.zeros(P.shape[0], device='cuda')
    for s in range(0, P.shape[0], chunk):
        e = min(s + chunk, P.shape[0])
        d_p2g[s:e] = torch.cdist(P[s:e], G).min(1)[0]

    dist_mm = d_p2g.cpu().numpy()
    colors  = error_to_color(dist_mm)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    save_colored_ply(out_path, pred, colors)

    return float(dist_mm.mean()), float(np.percentile(dist_mm, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred',  default=None)
    ap.add_argument('--gt',    default=None)
    ap.add_argument('--out',   default=None)
    ap.add_argument('--batch', action='store_true',
                    help='Generate error PLYs for all test bones in test_infer/')
    args = ap.parse_args()

    os.makedirs(VIZ_DIR, exist_ok=True)

    if args.batch:
        print(f"Batch error visualization → {VIZ_DIR}\n")
        print(f"{'Bone':<26} {'mean_d2g':>10} {'p95_d2g':>10}")
        print("-" * 50)
        for side, sid in TEST_BONES:
            label   = f"{side}_{sid}"
            pred_p  = os.path.join(PRED_DIR, f"{label}_generated.ply")
            gt_p    = os.path.join(GT_DIR,   f"{label}_gt.ply")
            out_p   = os.path.join(VIZ_DIR,  f"{label}_error.ply")

            if not os.path.exists(pred_p):
                print(f"  ✗ {label:<24}  (no generated file)")
                continue
            if not os.path.exists(gt_p):
                print(f"  ✗ {label:<24}  (no GT file)")
                continue

            try:
                mean_d, p95_d = make_error_ply(pred_p, gt_p, out_p)
                print(f"{label:<26} {mean_d:10.3f} {p95_d:10.3f}")
            except Exception as e:
                print(f"  ✗ ERROR {label}: {e}")

        print(f"\nColor key: green<2mm | yellow 2-4mm | orange 4-6mm | red>6mm")
        print(f"Open files in MeshLab/CloudCompare to visualize.")

    elif args.pred and args.gt and args.out:
        mean_d, p95_d = make_error_ply(args.pred, args.gt, args.out)
        print(f"Mean d(pred→GT): {mean_d:.3f} mm   P95: {p95_d:.3f} mm")
        print(f"Saved → {args.out}")

    else:
        print("Specify --batch  OR  --pred <path> --gt <path> --out <path>")


if __name__ == "__main__":
    main()
