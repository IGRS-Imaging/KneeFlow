"""
PREPARE 144 ORIGINAL FEMURS + TEMPLATE REGISTRATION
=====================================================
1. Read master landmarks CSV → filter normal-sized (350-550mm)
2. Copy rmesh PLY from FEMUR_SURFACE_DATA
3. Copy surface + boundary CSVs from registered data
4. Template-register everything using Procrustes (no scaling)

Input:
  /home/imaging/Desktop/PointGAN/data_source/FEMUR_SURFACE_DATA/  (rmesh PLY)
  /home/imaging/Desktop/PointGAN/data_source/Landmarks/femur_landmarks.csv
  /home/imaging/Desktop/Ragavan/left/Surface_points/   (surface CSV)
  /home/imaging/Desktop/Ragavan/left/left_condyaler points/  (boundary CSV)
  /home/imaging/Desktop/Ragavan/right/Surface_points/
  /home/imaging/Desktop/Ragavan/right/right_condyaler_points/

Output:
  /home/imaging/Desktop/PointGAN/data_source/og/
    ├── left/
    │   ├── rmesh_ply/
    │   ├── landmarks/
    │   ├── surface_points/
    │   └── condylar_boundary/
    └── right/
        ├── rmesh_ply/
        ├── landmarks/
        ├── surface_points/
        └── condylar_boundary/
"""

import os
import csv
import shutil
import struct
import numpy as np


# ============================================================================
# PATHS
# ============================================================================
FEMUR_DATA = "/home/imaging/Desktop/PointGAN/data_source/FEMUR_SURFACE_DATA"
MASTER_LM  = "/home/imaging/Desktop/PointGAN/data_source/Landmarks/femur_landmarks.csv"

# Existing surface/boundary data
LEFT_SURF_SRC   = "/home/imaging/Desktop/Ragavan/left/Surface_points"
LEFT_BOUND_SRC  = "/home/imaging/Desktop/Ragavan/left/left_condyaler points"
RIGHT_SURF_SRC  = "/home/imaging/Desktop/Ragavan/right/Surface_points"
RIGHT_BOUND_SRC = "/home/imaging/Desktop/Ragavan/right/right_condyaler_points"

# Output
OUT_BASE = "/home/imaging/Desktop/PointGAN/data_source/og"

# Length filter
MIN_LENGTH = 350.0
MAX_LENGTH = 550.0

# Template for registration (will be auto-selected)
TEMPLATE_ID = None  # Auto-pick median-length femur

LM_NAMES = [
    'HIP CENTRE', 'FEMUR KNEE CENTRE',
    'MEDIAL EPICONDYLE', 'LATERAL EPICONDYLE',
    'MEDIAL DISTAL CONDYLE', 'LATERAL DISTAL CONDYLE',
    'MEDIAL POSTERIOR CONDYLE', 'LATERAL POSTERIOR CONDYLE',
    'Medial Anterior Cortex', 'Lateral Anterior Cortex',
    'Medial Posterior Proximal', 'Lateral Posterior Proximal',
]


# ============================================================================
# MASTER CSV PARSER
# ============================================================================

def parse_master_csv(csv_path):
    """Parse master landmarks CSV → dict of {source_id: {landmark_name: [x,y,z]}}"""
    all_data = {}
    with open(csv_path, 'r') as f:
        lines = f.readlines()

    # Skip 2 header rows
    for line in lines[2:]:
        line = line.strip()
        if not line:
            continue
        vals = line.split(',')
        if len(vals) < 37:
            continue
        source = vals[0].strip()
        if not source:
            continue
        try:
            lm = {
                'HIP CENTRE':                np.array([float(vals[1]),  float(vals[2]),  float(vals[3])]),
                'FEMUR KNEE CENTRE':         np.array([float(vals[4]),  float(vals[5]),  float(vals[6])]),
                'MEDIAL EPICONDYLE':         np.array([float(vals[7]),  float(vals[8]),  float(vals[9])]),
                'LATERAL EPICONDYLE':        np.array([float(vals[10]), float(vals[11]), float(vals[12])]),
                'MEDIAL DISTAL CONDYLE':     np.array([float(vals[13]), float(vals[14]), float(vals[15])]),
                'LATERAL DISTAL CONDYLE':    np.array([float(vals[16]), float(vals[17]), float(vals[18])]),
                'MEDIAL POSTERIOR CONDYLE':   np.array([float(vals[19]), float(vals[20]), float(vals[21])]),
                'LATERAL POSTERIOR CONDYLE':  np.array([float(vals[22]), float(vals[23]), float(vals[24])]),
                'Medial Anterior Cortex':    np.array([float(vals[25]), float(vals[26]), float(vals[27])]),
                'Lateral Anterior Cortex':   np.array([float(vals[28]), float(vals[29]), float(vals[30])]),
                'Medial Posterior Proximal':  np.array([float(vals[31]), float(vals[32]), float(vals[33])]),
                'Lateral Posterior Proximal': np.array([float(vals[34]), float(vals[35]), float(vals[36])]),
            }
            length = np.linalg.norm(lm['HIP CENTRE'] - lm['FEMUR KNEE CENTRE'])
            all_data[source] = {'landmarks': lm, 'length': length}
        except (ValueError, IndexError) as e:
            continue
    return all_data


# ============================================================================
# PLY LOADER (handles double + float)
# ============================================================================

def load_ply(path):
    """Load PLY vertices + faces."""
    with open(path, 'rb') as f:
        line = f.readline().decode('ascii', errors='ignore').strip()
        assert line == 'ply'
        nv = 0; nf = 0; props = []; fmt = 'ascii'; inv = False; inf = False
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            if 'binary_little_endian' in line: fmt = 'bin'
            elif line.startswith('element vertex'): nv = int(line.split()[-1]); inv = True; inf = False
            elif line.startswith('element face'): nf = int(line.split()[-1]); inv = False; inf = True
            elif line.startswith('element'): inv = False; inf = False
            elif line.startswith('property') and inv: props.append(line.split()[1])
            elif line == 'end_header': break

        tsz = {'float': 4, 'float32': 4, 'double': 8, 'float64': 8,
               'int': 4, 'int32': 4, 'uint8': 1, 'uchar': 1, 'short': 2, 'ushort': 2}
        tdt = {'float': '<f4', 'float32': '<f4', 'double': '<f8', 'float64': '<f8',
               'int': '<i4', 'int32': '<i4', 'uint8': '<u1', 'uchar': '<u1'}

        if fmt == 'bin':
            bpv = sum(tsz.get(p, 4) for p in props)
            raw = f.read(nv * bpv)
            nv = min(nv, len(raw) // bpv)
            if all(p == props[0] for p in props):
                dt = tdt.get(props[0], '<f4')
                sz = tsz.get(props[0], 4)
                data = np.frombuffer(raw[:nv*len(props)*sz], dtype=dt).reshape(nv, len(props))
                verts = data[:, :3].astype(np.float64)
            else:
                dtl = [(f'p{i}', tdt.get(p, '<f4')) for i, p in enumerate(props)]
                dt = np.dtype(dtl)
                st = np.frombuffer(raw[:nv*dt.itemsize], dtype=dt)
                verts = np.column_stack([st['p0'].astype(np.float64),
                                         st['p1'].astype(np.float64),
                                         st['p2'].astype(np.float64)])
            # Read faces
            faces = None
            if nf > 0:
                fl = []
                for _ in range(nf):
                    try:
                        cb = f.read(1)
                        if not cb: break
                        c = struct.unpack('<B', cb)[0]
                        fb = f.read(c * 4)
                        if len(fb) < c * 4: break
                        fl.append(struct.unpack(f'<{c}i', fb))
                    except: break
                if fl:
                    tri = []
                    for face in fl:
                        if len(face) == 3: tri.append(face)
                        elif len(face) == 4:
                            tri.append((face[0], face[1], face[2]))
                            tri.append((face[0], face[2], face[3]))
                    faces = np.array(tri, dtype=np.int32) if tri else None
        else:
            verts = np.zeros((nv, 3), dtype=np.float64)
            for i in range(nv):
                vals = f.readline().decode('ascii', errors='ignore').strip().split()
                verts[i] = [float(vals[0]), float(vals[1]), float(vals[2])]
            faces = None
    return verts, faces


def load_csv_pts(path):
    """Load CSV points — auto-detects X,Y,Z columns."""
    rows = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        hl = [h.strip().lower() for h in header]
        if 'x' in hl and 'y' in hl and 'z' in hl:
            xi, yi, zi = hl.index('x'), hl.index('y'), hl.index('z')
        else:
            xi, yi, zi = 0, 1, 2
        for row in reader:
            try: rows.append([float(row[xi]), float(row[yi]), float(row[zi])])
            except: continue
    return np.array(rows, dtype=np.float64) if rows else np.zeros((0, 3))


# ============================================================================
# SAVE FUNCTIONS
# ============================================================================

def save_ply(path, verts, faces=None):
    nv = len(verts)
    nf = 0
    if faces is not None and len(faces) > 0:
        valid = np.all((faces >= 0) & (faces < nv), axis=1)
        faces = faces[valid]
        nf = len(faces)
    with open(path, 'wb') as f:
        h = "ply\nformat binary_little_endian 1.0\n"
        h += f"element vertex {nv}\nproperty float x\nproperty float y\nproperty float z\n"
        if nf > 0:
            h += f"element face {nf}\nproperty list uchar int vertex_indices\n"
        h += "end_header\n"
        f.write(h.encode('ascii'))
        for v in verts:
            f.write(struct.pack('<fff', float(v[0]), float(v[1]), float(v[2])))
        if nf > 0:
            for face in faces:
                f.write(struct.pack('<B', 3))
                f.write(struct.pack('<iii', int(face[0]), int(face[1]), int(face[2])))


def save_csv(path, pts):
    with open(path, 'w') as f:
        f.write('x,y,z\n')
        for p in pts:
            f.write(f"{p[0]:.10f},{p[1]:.10f},{p[2]:.10f}\n")


def save_lm_csv(path, lm_dict):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Landmark', 'X', 'Y', 'Z'])
        for name in LM_NAMES:
            if name in lm_dict:
                p = lm_dict[name]
                w.writerow([name, f"{p[0]:.10f}", f"{p[1]:.10f}", f"{p[2]:.10f}"])


# ============================================================================
# PROCRUSTES (rigid, no scaling)
# ============================================================================

def rigid_procrustes(src, tgt):
    """Align src landmarks to tgt. Returns R, t."""
    mask = np.any(src != 0, axis=1) & np.any(tgt != 0, axis=1)
    s = src[mask]; t = tgt[mask]
    assert len(s) >= 3, f"Need 3+ landmarks, got {len(s)}"
    sm = s.mean(0); tm = t.mean(0)
    sc = s - sm; tc = t - tm
    H = sc.T @ tc
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, np.sign(d)]) @ U.T
    t = tm - sm @ R.T
    return R, t


def lm_to_array(lm_dict):
    arr = np.zeros((12, 3), dtype=np.float64)
    for i, name in enumerate(LM_NAMES):
        if name in lm_dict: arr[i] = lm_dict[name]
    return arr


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("  PREPARE 144 ORIGINAL FEMURS + REGISTER")
    print("=" * 60)

    # Parse master CSV
    print(f"\nParsing: {MASTER_LM}")
    all_data = parse_master_csv(MASTER_LM)
    print(f"  Total entries: {len(all_data)}")

    # Filter by length
    normal = {k: v for k, v in all_data.items()
              if MIN_LENGTH <= v['length'] <= MAX_LENGTH}
    print(f"  Normal size ({MIN_LENGTH}-{MAX_LENGTH}mm): {len(normal)}")

    left = {k: v for k, v in normal.items() if '_L' in k}
    right = {k: v for k, v in normal.items() if '_R' in k}
    print(f"  Left: {len(left)}, Right: {len(right)}")

    # Pick template (median length)
    lengths = [(k, v['length']) for k, v in normal.items()]
    lengths.sort(key=lambda x: x[1])
    template_id = lengths[len(lengths) // 2][0]
    template_lm = normal[template_id]['landmarks']
    template_arr = lm_to_array(template_lm)
    print(f"\n  Template: {template_id} ({normal[template_id]['length']:.1f}mm)")

    # Create output dirs
    for side in ['left', 'right']:
        for sub in ['rmesh_ply', 'landmarks', 'surface_points', 'condylar_boundary']:
            os.makedirs(os.path.join(OUT_BASE, side, sub), exist_ok=True)

    # Process each femur
    ok = 0; fail = 0; no_rmesh = 0; no_surf = 0

    for sid, info in sorted(normal.items()):
        side = 'left' if '_L' in sid else 'right'
        lm_dict = info['landmarks']

        # Find rmesh PLY
        folder = os.path.join(FEMUR_DATA, sid)
        rmesh = None
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if 'rmesh' in f.lower() and f.endswith('.ply'):
                    rmesh = os.path.join(folder, f)
                    break
        if rmesh is None:
            no_rmesh += 1
            continue

        # Find surface CSV
        if side == 'left':
            surf_src = LEFT_SURF_SRC
            bound_src = LEFT_BOUND_SRC
        else:
            surf_src = RIGHT_SURF_SRC
            bound_src = RIGHT_BOUND_SRC

        surf_path = None
        for ext in ['.csv', '.ply']:
            c = os.path.join(surf_src, f"{sid}_surface{ext}")
            if os.path.exists(c): surf_path = c; break

        bound_path = None
        for ext in ['.csv', '.ply']:
            c = os.path.join(bound_src, f"{sid}_boundary{ext}")
            if os.path.exists(c): bound_path = c; break

        if surf_path is None or bound_path is None:
            no_surf += 1
            continue

        try:
            # Compute Procrustes alignment to template
            src_arr = lm_to_array(lm_dict)
            R, t = rigid_procrustes(src_arr, template_arr)

            # Load and transform rmesh PLY
            verts, faces = load_ply(rmesh)
            reg_verts = verts @ R.T + t
            save_ply(os.path.join(OUT_BASE, side, 'rmesh_ply', f"{sid}_rmesh.ply"),
                     reg_verts, faces)

            # Transform landmarks
            reg_lm = {n: p @ R.T + t for n, p in lm_dict.items()}
            save_lm_csv(os.path.join(OUT_BASE, side, 'landmarks', f"{sid}_landmarks.csv"),
                        reg_lm)

            # Load and transform surface points
            if surf_path.endswith('.csv'):
                sp = load_csv_pts(surf_path)
            else:
                sp, _ = load_ply(surf_path)
            sp_reg = sp @ R.T + t
            save_csv(os.path.join(OUT_BASE, side, 'surface_points', f"{sid}_surface.csv"),
                     sp_reg)

            # Load and transform boundary points
            if bound_path.endswith('.csv'):
                bp = load_csv_pts(bound_path)
            else:
                bp, _ = load_ply(bound_path)
            bp_reg = bp @ R.T + t
            save_csv(os.path.join(OUT_BASE, side, 'condylar_boundary', f"{sid}_boundary.csv"),
                     bp_reg)

            ok += 1
            if ok % 20 == 0 or ok <= 3:
                print(f"  [{ok}] {sid} ({info['length']:.0f}mm) ✅")

        except Exception as e:
            fail += 1
            print(f"  {sid} ❌ {e}")

    print(f"\n{'='*60}")
    print(f"  ✅ Registered: {ok}")
    print(f"  ❌ Failed: {fail}")
    print(f"  No rmesh: {no_rmesh}")
    print(f"  No surface/boundary: {no_surf}")
    print(f"\n  Output: {OUT_BASE}")
    print(f"  Template: {template_id}")

    # Count output
    for side in ['left', 'right']:
        n = len([f for f in os.listdir(os.path.join(OUT_BASE, side, 'rmesh_ply'))
                 if f.endswith('.ply')])
        print(f"  {side}: {n} femurs")


if __name__ == "__main__":
    main()
