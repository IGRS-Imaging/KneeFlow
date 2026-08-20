"""
Export GT point clouds for all 29 test bones.
Samples 8192 points from mesh surface — same as model output format.
Saves as ASCII PLY (no faces) to test_bones_gt folder.
"""
import os
import sys
import numpy as np

# Add femur_fm_9L_work to path so we can reuse data.py functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_ply_mesh, sample_mesh_surface, clean_pts

# ── Config ─────────────────────────────────────────────────────────────────────
BASE    = "/home/imaging/Desktop/Ragavan"
OUT_DIR = os.path.join(BASE, "9L_working_output", "test_bones_gt")
N_PTS   = 8192

LEFT_PLY  = os.path.join(BASE, "left",  "registered_ply")
RIGHT_PLY = os.path.join(BASE, "right", "registered_ply")

# ── All 29 test bones (selected_29_dice78_82) ──────────────────────────────────
TEST_BONES = [
    # Left femurs
    ('L', '222_Femur_L'), ('L', '225_Femur_L'), ('L', '232_Femur_L'),
    ('L', '237_Femur_L'), ('L', '245_Femur_L'), ('L', '285_Femur_L'),
    ('L', '298_Femur_L'), ('L', '308_Femur_L'), ('L', '328_Femur_L'),
    ('L', '330_Femur_L'), ('L', '335_Femur_L'), ('L', '343_Femur_L'),
    ('L', '350_Femur_L'), ('L', '354_Femur_L'), ('L', '370_Femur_L'),
    ('L', '380_Femur_L'), ('L', '381_Femur_L'), ('L', '392_Femur_L'),
    # Right femurs
    ('R', '43_Femur_R'),  ('R', '62_Femur_R'),  ('R', '99_Femur_R'),
    ('R', '101_Femur_R'), ('R', '114_Femur_R'), ('R', '163_Femur_R'),
    ('R', '197_Femur_R'), ('R', '198_Femur_R'), ('R', '203_Femur_R'),
    ('R', '210_Femur_R'), ('R', '215_Femur_R'),
]


def save_ply(path, pts):
    """Save point cloud as ASCII PLY — no faces."""
    pts = np.asarray(pts)
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in pts:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Output folder: {OUT_DIR}")
    print(f"Sampling {N_PTS} points per bone\n")

    ok = 0
    fail = 0

    for side, sid in TEST_BONES:
        ply_dir = LEFT_PLY if side == 'L' else RIGHT_PLY

        # Try common filename variants
        found = None
        for name in [f"{sid}.ply", f"{sid}_registered.ply",
                     f"{sid}_registered_left.ply", f"{sid}_registered_right.ply",
                     f"{sid}_rmesh.ply", f"{sid}_remesh.ply"]:
            p = os.path.join(ply_dir, name)
            if os.path.exists(p):
                found = p
                break

        if found is None:
            print(f"  ✗ NOT FOUND: {side}_{sid}")
            fail += 1
            continue

        try:
            verts, faces = load_ply_mesh(found)
            verts = clean_pts(verts)

            if len(verts) < 100:
                print(f"  ✗ Too few vertices: {side}_{sid} ({len(verts)})")
                fail += 1
                continue

            # Sample 8192 points from mesh surface (same as model output)
            pts = sample_mesh_surface(verts, faces, N_PTS)

            out_path = os.path.join(OUT_DIR, f"{side}_{sid}_gt.ply")
            save_ply(out_path, pts)

            print(f"  ✓ {side}_{sid}  →  {N_PTS} pts  saved")
            ok += 1

        except Exception as e:
            print(f"  ✗ ERROR {side}_{sid}: {e}")
            fail += 1

    print(f"\nDone: {ok} saved, {fail} failed")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
