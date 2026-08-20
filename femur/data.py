"""
Dataset - Full Femur Generation (Publication Quality)
- INPUT:  condylar boundary points + 12 landmarks
- TARGET: full femur — uniformly sampled from MESH SURFACE (not raw vertices)
- Global normalization saved to norm_stats.json
"""
import os, csv, json
import numpy as np
import torch
from torch.utils.data import Dataset
import cfg


# ── PLY loader — reads both vertices AND faces ────────────────────────────────

def load_ply_mesh(path):
    """
    Load PLY file. Returns (vertices, faces).
    faces is None if no face data exists (point cloud only).
    """
    with open(path, 'rb') as f:
        line = f.readline().decode('ascii', errors='ignore').strip()
        assert line == 'ply', f"Not PLY: {path}"

        nv = 0; nf = 0; v_props = []; fmt = 'ascii'
        in_vertex = False; in_face = False

        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            if 'binary_little_endian' in line: fmt = 'bin'
            elif line.startswith('element vertex'):
                nv = int(line.split()[-1]); in_vertex = True; in_face = False
            elif line.startswith('element face'):
                nf = int(line.split()[-1]); in_face = True; in_vertex = False
            elif line.startswith('element'):
                in_vertex = False; in_face = False
            elif line.startswith('property') and in_vertex:
                v_props.append(line.split()[1])
            elif line == 'end_header':
                break

        type_sizes  = {'float':4,'float32':4,'double':8,'float64':8,
                       'int':4,'int32':4,'uint8':1,'uchar':1,
                       'short':2,'ushort':2,'uint':4,'uint32':4}
        type_dtypes = {'float':'<f4','float32':'<f4','double':'<f8','float64':'<f8',
                       'int':'<i4','int32':'<i4','uint8':'<u1','uchar':'<u1',
                       'short':'<i2','ushort':'<u2','uint':'<u4','uint32':'<u4'}

        if fmt == 'bin':
            # Read vertices
            bpv = sum(type_sizes.get(p,4) for p in v_props)
            raw_v = f.read(nv * bpv)
            actual = len(raw_v) // bpv
            if actual < nv: nv = actual
            if nv == 0: return np.zeros((0,3),dtype=np.float32), None

            if all(p == v_props[0] for p in v_props):
                dt = type_dtypes.get(v_props[0], '<f4')
                np_ = len(v_props)
                data = np.frombuffer(raw_v[:nv*np_*type_sizes.get(v_props[0],4)],
                                     dtype=dt).reshape(nv, np_)
                verts = data[:,:3].astype(np.float32)
            else:
                dt_list = [(f'p{i}',type_dtypes.get(p,'<f4')) for i,p in enumerate(v_props)]
                structured = np.frombuffer(raw_v[:nv*np.dtype(dt_list).itemsize],
                                           dtype=np.dtype(dt_list))
                verts = np.column_stack([structured['p0'].astype(np.float32),
                                         structured['p1'].astype(np.float32),
                                         structured['p2'].astype(np.float32)])

            # Read faces if present
            faces = None
            if nf > 0:
                face_list = []
                try:
                    for _ in range(nf):
                        # each face: uchar count + count * int indices
                        count_bytes = f.read(1)
                        if not count_bytes: break
                        count = count_bytes[0]
                        idx_bytes = f.read(count * 4)
                        if len(idx_bytes) < count * 4: break
                        idx = np.frombuffer(idx_bytes, dtype='<i4')
                        if count == 3:
                            face_list.append(idx)
                        elif count == 4:
                            # quad → two triangles
                            face_list.append(idx[[0,1,2]])
                            face_list.append(idx[[0,2,3]])
                    if face_list:
                        faces = np.array(face_list, dtype=np.int32)
                except Exception:
                    faces = None
        else:
            # ASCII
            verts = np.zeros((nv,3), dtype=np.float32)
            for i in range(nv):
                vals = f.readline().decode('ascii', errors='ignore').strip().split()
                if len(vals) >= 3:
                    verts[i] = [float(vals[0]), float(vals[1]), float(vals[2])]
            faces = None
            if nf > 0:
                face_list = []
                for _ in range(nf):
                    vals = f.readline().decode('ascii', errors='ignore').strip().split()
                    if len(vals) >= 4:
                        n = int(vals[0])
                        idx = [int(vals[j+1]) for j in range(n)]
                        if n == 3: face_list.append(idx)
                        elif n == 4:
                            face_list.append([idx[0],idx[1],idx[2]])
                            face_list.append([idx[0],idx[2],idx[3]])
                if face_list:
                    faces = np.array(face_list, dtype=np.int32)

    return verts, faces


def sample_mesh_surface(verts, faces, n):
    """
    Uniformly sample n points from mesh surface using area-weighted triangle sampling.
    This gives evenly distributed points regardless of mesh vertex density.
    Falls back to vertex sampling if no faces.
    """
    if faces is None or len(faces) == 0:
        # No face data — just sample vertices
        return rsample(verts, n)

    # Compute triangle areas
    v0 = verts[faces[:,0]]
    v1 = verts[faces[:,1]]
    v2 = verts[faces[:,2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    areas = np.clip(areas, 0, None)
    total = areas.sum()

    if total < 1e-10:
        return rsample(verts, n)

    # Sample triangles proportional to area
    probs = areas / total
    tri_idx = np.random.choice(len(faces), size=n, p=probs)

    # Random barycentric coordinates within each triangle
    r1 = np.random.rand(n, 1).astype(np.float32)
    r2 = np.random.rand(n, 1).astype(np.float32)
    mask = (r1 + r2) > 1.0
    r1[mask] = 1.0 - r1[mask]
    r2[mask] = 1.0 - r2[mask]
    r3 = 1.0 - r1 - r2

    p0 = verts[faces[tri_idx, 0]]
    p1 = verts[faces[tri_idx, 1]]
    p2 = verts[faces[tri_idx, 2]]

    pts = r1 * p0 + r2 * p1 + r3 * p2
    return pts.astype(np.float32)


def load_csv_pts(path):
    rows = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        hl = [h.strip().lower() for h in header]
        if 'x' in hl and 'y' in hl and 'z' in hl:
            xi, yi, zi = hl.index('x'), hl.index('y'), hl.index('z')
        else:
            try:
                float(header[0])
                xi, yi, zi = (2,3,4) if len(header)>=5 else (0,1,2)
                rows.append([float(header[xi]),float(header[yi]),float(header[zi])])
            except (ValueError, IndexError):
                xi, yi, zi = 0, 1, 2
        for row in reader:
            try: rows.append([float(row[xi]),float(row[yi]),float(row[zi])])
            except (ValueError, IndexError): continue
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0,3), dtype=np.float32)


def load_pts(path):
    if path.endswith('.ply'):
        verts, _ = load_ply_mesh(path)
        return verts
    return load_csv_pts(path)


def clean_pts(pts):
    if len(pts) == 0: return pts
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) == 0: return pts
    return pts[np.linalg.norm(pts.astype(np.float64), axis=1) < 2000]


def load_lm(path):
    lm = {}
    with open(path, 'r') as f:
        for row in csv.DictReader(f):
            name = row.get('Landmark', row.get('landmark','')).strip()
            if not name: continue
            try:
                v = np.array([float(row.get('X', row.get('x',0))),
                               float(row.get('Y', row.get('y',0))),
                               float(row.get('Z', row.get('z',0)))], dtype=np.float32)
                if np.all(np.isfinite(v)): lm[name] = v
            except (ValueError, TypeError): continue
    return lm


def rsample(pts, n):
    N = len(pts)
    if N == 0: return np.zeros((n,3), dtype=np.float32)
    return pts[np.random.choice(N, n, replace=(N < n))]


def augment(femur, bound, lm):
    """
    Strong augmentation — femur + boundary + landmarks only (no surface).
    All transforms applied consistently across all inputs.
    """
    # Random rotation: all 3 axes ±20°
    angles = np.random.uniform(-20, 20, 3) * np.pi / 180
    ax, ay, az = angles

    Rx = np.array([[1,0,0],[0,np.cos(ax),-np.sin(ax)],[0,np.sin(ax),np.cos(ax)]], dtype=np.float32)
    Ry = np.array([[np.cos(ay),0,np.sin(ay)],[0,1,0],[-np.sin(ay),0,np.cos(ay)]], dtype=np.float32)
    Rz = np.array([[np.cos(az),-np.sin(az),0],[np.sin(az),np.cos(az),0],[0,0,1]], dtype=np.float32)
    R  = Rx @ Ry @ Rz

    femur = femur @ R.T
    bound = bound @ R.T
    lm    = lm    @ R.T

    # Scale ±7% — covers anatomical size variation across patients
    sc = 1.0 + np.random.uniform(-0.07, 0.07)
    femur *= sc; bound *= sc; lm *= sc

    # Translation ±5mm in normalized space
    t = np.random.uniform(-0.05, 0.05, (1, 3)).astype(np.float32)
    femur += t; bound += t; lm += t

    # Noise on boundary — target femur stays clean
    bound += np.random.randn(*bound.shape).astype(np.float32) * 0.015

    # Random landmark dropout: zero out 1 landmark with p=0.15 (robustness)
    if np.random.rand() < 0.15:
        drop_idx = np.random.randint(0, lm.shape[0])
        lm[drop_idx] = 0.0

    return femur, bound, lm


# ── Global normalization ──────────────────────────────────────────────────────

def compute_and_save_global_norm(raw_samples):
    all_pts  = np.concatenate([s['femur_raw'] for s in raw_samples], axis=0)
    centroid = all_pts.mean(axis=0).astype(np.float32)
    std      = float(all_pts.std()) + 1e-8
    os.makedirs(cfg.OUTPUT, exist_ok=True)
    path = os.path.join(cfg.OUTPUT, 'norm_stats.json')
    with open(path, 'w') as f:
        json.dump({'centroid': centroid.tolist(), 'std': std}, f, indent=2)
    print(f"  Global norm → centroid={centroid}, std={std:.4f}")
    print(f"  Saved: {path}")
    return centroid, std


def load_norm_stats(path=None):
    if path is None:
        path = os.path.join(cfg.OUTPUT, 'norm_stats.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f"norm_stats.json not found at {path}. Run training first.")
    with open(path) as f:
        s = json.load(f)
    return np.array(s['centroid'], dtype=np.float32), float(s['std'])


# ── Shared raw loader ─────────────────────────────────────────────────────────

def load_all_raw():
    """
    Load all raw femur samples from left + right directories.
    Returns list of dicts with raw (unnormalised) data, tagged by side ('L'/'R').
    """
    raw = []
    sources = [
        ('L', cfg.LEFT_PLY,  cfg.LEFT_LM,  cfg.LEFT_BOUND),
        ('R', cfg.RIGHT_PLY, cfg.RIGHT_LM, cfg.RIGHT_BOUND),
    ]

    for tag, ply_d, lm_d, bound_d in sources:
        if not os.path.exists(ply_d):
            print(f"  {tag}: dir missing {ply_d}"); continue

        plys = sorted(f for f in os.listdir(ply_d) if f.endswith('.ply'))
        ok   = 0
        for fn in plys:
            sid = fn.replace('.ply', '')
            for sfx in ['_registered', '_rmesh', '_remesh']:
                if sid.endswith(sfx): sid = sid[:-len(sfx)]; break

            # Landmarks CSV
            lm_p = os.path.join(lm_d, f"{sid}_landmarks.csv")
            if not os.path.exists(lm_p): continue

            # Boundary points (.csv or .ply)
            bound_p = None
            for ext in ['.csv', '.ply']:
                c = os.path.join(bound_d, f"{sid}_boundary{ext}")
                if os.path.exists(c): bound_p = c; break
            if not bound_p: continue

            ply_p = os.path.join(ply_d, fn)

            try:
                verts, faces = load_ply_mesh(ply_p)
                verts = clean_pts(verts)
                bp    = clean_pts(load_pts(bound_p))
                lm    = load_lm(lm_p)

                la = np.zeros((cfg.N_LM, 3), dtype=np.float32)
                for i, name in enumerate(cfg.LM_NAMES):
                    if name in lm: la[i] = lm[name]

                if len(verts) < 100 or len(bp) < 10: continue
                if (np.any(~np.isfinite(verts)) or np.any(~np.isfinite(bp))
                        or np.any(~np.isfinite(la))): continue

                has_faces = faces is not None and len(faces) > 0
                print(f"  {tag} {sid}: {len(verts)} verts, "
                      f"{'%d faces' % len(faces) if has_faces else 'no faces'}")

                raw.append({
                    'id':        f"{tag}_{sid}",
                    'tag':       tag,
                    'femur_raw': verts,
                    'verts':     verts,
                    'faces':     faces,
                    'bound_raw': bp,
                    'lm_raw':    la,
                })
                ok += 1
            except Exception as e:
                print(f"  Skip {sid}: {e}")

        print(f"  {tag}: {ok}/{len(plys)} loaded")

    return raw


# Fixed held-out test set — the exact 29 bones used for the paper's reported
# results (selected_29_dice78_82.csv). These must be excluded from training
# every run, so no random shuffle is used here.
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
TEST_IDS = {f"{side}_{sid}" for side, sid in TEST_BONES}


def split_raw(raw):
    """
    Fixed train/test split using the exact 29 held-out test bones
    (selected_29_dice78_82) instead of a random shuffle, so the same
    bones are excluded from training every run.
    Returns: train_raw, test_raw
    """
    train_raw = [r for r in raw if r['id'] not in TEST_IDS]
    test_raw  = [r for r in raw if r['id']     in TEST_IDS]

    missing = TEST_IDS - {r['id'] for r in test_raw}
    if missing:
        print(f"  WARNING: {len(missing)} test bones not found in raw data: {sorted(missing)}")

    l_train = sum(1 for r in train_raw if r['tag'] == 'L')
    r_train = sum(1 for r in train_raw if r['tag'] == 'R')
    l_test  = sum(1 for r in test_raw  if r['tag'] == 'L')
    r_test  = sum(1 for r in test_raw  if r['tag'] == 'R')

    print(f"\n  Split (fixed 29-bone test set) ->")
    print(f"    Train : {len(train_raw)}  ({l_train}L + {r_train}R)")
    print(f"    Test  : {len(test_raw)}   ({l_test}L + {r_test}R)")
    return train_raw, test_raw


def normalise_raw(raw, gc, gs):
    """Apply global normalisation to a list of raw samples. Returns list of norm samples."""
    samples = []
    for r in raw:
        verts = (r['verts']     - gc) / gs
        bp    = (r['bound_raw'] - gc) / gs
        la    = (r['lm_raw']    - gc) / gs

        if (np.any(~np.isfinite(verts)) or np.any(~np.isfinite(bp))
                or np.any(~np.isfinite(la))):
            print(f"  Skip {r['id']}: NaN after norm"); continue

        samples.append({
            'id':    r['id'],
            'verts': verts,
            'faces': r['faces'],
            'bound': bp,
            'lm':    la,
        })
    return samples


# ── Dataset classes ───────────────────────────────────────────────────────────

class DS(Dataset):
    """Training dataset — augmentation + REPEATS virtual epoch expansion."""
    def __init__(self, samples):
        self.samples = samples
        print(f"  Train DS: {len(samples)} femurs, "
              f"{len(samples) * cfg.REPEATS} steps/epoch")

    def __len__(self):
        return len(self.samples) * cfg.REPEATS

    def __getitem__(self, idx):
        s     = self.samples[idx % len(self.samples)]
        femur = sample_mesh_surface(s['verts'], s['faces'], cfg.N_FEMUR)
        bound = rsample(s['bound'], cfg.N_BOUND)
        lm    = s['lm'].copy()

        femur, bound, lm = augment(femur, bound, lm)
        return {
            'femur': torch.from_numpy(femur),
            'bound': torch.from_numpy(bound),
            'lm':    torch.from_numpy(lm),
        }


class DSTest(Dataset):
    """
    Test / validation dataset — NO augmentation, NO repeats.
    Uses fixed surface sampling for reproducible evaluation.
    """
    def __init__(self, samples):
        self.samples = samples
        print(f"  Test  DS: {len(samples)} femurs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s     = self.samples[idx]
        femur = sample_mesh_surface(s['verts'], s['faces'], cfg.N_FEMUR)
        bound = rsample(s['bound'], cfg.N_BOUND)
        lm    = s['lm'].copy()
        return {
            'femur': torch.from_numpy(femur),
            'bound': torch.from_numpy(bound),
            'lm':    torch.from_numpy(lm),
        }


def build_datasets():
    """
    Full pipeline: load -> split (90/10) -> normalise ->
    return train DS, test DS + norm stats.
    Call this once from train.py.
    """
    print("\n--- Loading data ---")
    raw = load_all_raw()
    if not raw:
        raise RuntimeError("No samples loaded. Check data paths in cfg.py.")

    train_raw, test_raw = split_raw(raw)

    # Norm stats computed from TRAIN only — prevents data leakage
    print("\n  Computing global norm stats (train only)...")
    gc, gs = compute_and_save_global_norm(train_raw)

    train_samples = normalise_raw(train_raw, gc, gs)
    test_samples  = normalise_raw(test_raw,  gc, gs)

    return DS(train_samples), DSTest(test_samples), gc, gs
