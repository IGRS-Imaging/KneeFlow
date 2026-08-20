"""
convert_test_bones.py
Converts test femur PLY meshes → 8192-point surface point clouds (no faces).
Saves to /home/imaging/Desktop/Ragavan/test_bones/

Usage:
    cd ~/Desktop/Ragavan/femur_fm_9L
    python convert_test_bones.py
"""
import os
import sys
import numpy as np

# Add project dir to path so we can reuse load_ply_mesh + sample_mesh_surface
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_ply_mesh, sample_mesh_surface

# ── Test bones ────────────────────────────────────────────────────────────────
TEST_BONES = [
    ('L', '380_Femur_L'), ('L', '381_Femur_L'), ('L', '382_Femur_L'),
    ('L', '383_Femur_L'), ('L', '385_Femur_L'), ('L', '386_Femur_L'),
    ('L', '387_Femur_L'), ('L', '388_Femur_L'), ('L', '389_Femur_L'),
    ('L', '390_Femur_L'), ('L', '391_Femur_L'), ('L', '392_Femur_L'),
    ('L', '393_Femur_L'), ('L', '394_Femur_L'), ('L', '395_Femur_L'),
    ('R', '85_Femur_R'),  ('R', '86_Femur_R'),  ('R', '87_Femur_R'),
    ('R', '88_Femur_R'),  ('R', '89_Femur_R'),  ('R', '90_Femur_R'),
    ('R', '91_Femur_R'),  ('R', '92_Femur_R'),  ('R', '93_Femur_R'),
    ('R', '94_Femur_R'),  ('R', '96_Femur_R'),  ('R', '97_Femur_R'),
    ('R', '98_Femur_R'),  ('R', '99_Femur_R'),
]

BASE      = "/home/imaging/Desktop/Ragavan"
LEFT_PLY  = os.path.join(BASE, "left",  "registered_ply")
RIGHT_PLY = os.path.join(BASE, "right", "registered_ply")
OUT_DIR   = os.path.join(BASE, "test_bones")
N_PTS     = 8192

os.makedirs(OUT_DIR, exist_ok=True)


def save_ply_pointcloud(path, pts):
    """Save Nx3 numpy array as ASCII PLY point cloud (no faces)."""
    n = len(pts)
    with open(path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in pts:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def main():
    print(f"Output dir : {OUT_DIR}")
    print(f"Points     : {N_PTS}")
    print(f"Bones      : {len(TEST_BONES)}")
    print("-" * 50)

    ok = 0; fail = 0

    for tag, bone_id in TEST_BONES:
        ply_dir  = LEFT_PLY if tag == 'L' else RIGHT_PLY
        src_path = os.path.join(ply_dir, f"{bone_id}_registered.ply")
        dst_path = os.path.join(OUT_DIR,  f"{bone_id}_registered_pc{N_PTS}.ply")

        if not os.path.exists(src_path):
            print(f"  MISSING : {src_path}")
            fail += 1
            continue

        try:
            verts, faces = load_ply_mesh(src_path)
            pts = sample_mesh_surface(verts, faces, N_PTS)
            save_ply_pointcloud(dst_path, pts)
            print(f"  OK  {bone_id:20s}  {len(verts):5d} verts → {N_PTS} pts  →  {os.path.basename(dst_path)}")
            ok += 1
        except Exception as e:
            print(f"  FAIL {bone_id}: {e}")
            fail += 1

    print("-" * 50)
    print(f"Done: {ok} saved, {fail} failed")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
