"""
Tibia Dataset
━━━━━━━━━━━━━
INPUT:  tibial boundary points (PLY) + 8 landmarks (CSV)
TARGET: full tibia — 8192 points sampled from mesh surface
Split:  90% train / 10% test  (random, seed=42, stratified by side)

Boundary files  : PLY format  in left_boundary_points / right_boundary_points
Landmark files  : CSV format  in registered_landmarks
Mesh files      : PLY format  in registered_ply

Naming convention expected (flexible matching tried):
  Mesh      : {id}_rmesh.ply  or  {id}.ply
  Landmarks : {id}_rmesh_landmarks.csv  or  {id}_landmarks.csv  or  {id}.csv
  Boundary  : {id}_boundary.ply  or  {id}_rmesh_boundary.ply  or  {id}.ply
"""
import os, csv, json
import numpy as np
import torch
from torch.utils.data import Dataset
import cfg


# ─────────────────────────────────────────────────────────────────────────────
# PLY loader — uses open3d (robust, handles all binary/ascii variants)
# ─────────────────────────────────────────────────────────────────────────────

import open3d as o3d

def load_ply_mesh(path):
    """
    Load PLY mesh → (vertices, faces).
    Uses open3d which correctly handles binary PLY with extra properties
    (normals, colours, etc.) that break manual parsers.
    """
    # Try as triangle mesh first (registered PLY files have faces)
    mesh = o3d.io.read_triangle_mesh(path)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)

    if len(verts) > 0:
        return verts, (faces if len(faces) > 0 else None)

    # Fallback: treat as point cloud (boundary PLY files)
    pcd   = o3d.io.read_point_cloud(path)
    verts = np.asarray(pcd.points, dtype=np.float32)
    return verts, None


def load_pts_ply(path):
    """Load point-only PLY → (N,3) numpy array."""
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Landmark loader
# ─────────────────────────────────────────────────────────────────────────────

def load_lm(path):
    """
    Load landmark CSV → dict {name: np.array([x,y,z])}.
    Expects header: Landmark,X,Y,Z
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


# ─────────────────────────────────────────────────────────────────────────────
# Mesh surface sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_mesh_surface(verts, faces, n):
    """
    Sample n points uniformly from mesh surface.
    Robust to NaN vertices and degenerate triangles.
    Falls back to vertex sampling if areas are all invalid.
    """
    # Always work with clean vertices
    valid_mask = np.isfinite(verts).all(axis=1)
    verts = verts[valid_mask]
    if len(verts) == 0:
        raise ValueError("No valid vertices after NaN filter")

    if faces is None or len(faces) == 0:
        idx = np.random.choice(len(verts), n, replace=(len(verts) < n))
        return verts[idx].astype(np.float32)

    # Remap face indices after vertex filtering
    old_to_new = np.full(valid_mask.shape[0], -1, dtype=np.int64)
    old_to_new[valid_mask] = np.arange(valid_mask.sum())
    # Only keep faces where all 3 vertices survived the filter
    face_ok = valid_mask[faces].all(axis=1)
    faces   = old_to_new[faces[face_ok]]

    if len(faces) == 0:
        idx = np.random.choice(len(verts), n, replace=(len(verts) < n))
        return verts[idx].astype(np.float32)

    # Compute triangle areas
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = np.linalg.norm(cross, axis=1) * 0.5
    areas = np.nan_to_num(areas, nan=0.0, posinf=0.0, neginf=0.0)
    areas = np.maximum(areas, 0.0)

    total = areas.sum()
    if total < 1e-10:
        # All triangles degenerate — fall back to vertex sampling
        idx = np.random.choice(len(verts), n, replace=(len(verts) < n))
        return verts[idx].astype(np.float32)

    probs  = areas / total
    chosen = np.random.choice(len(faces), n, p=probs, replace=True)
    r1 = np.random.rand(n, 1).astype(np.float32)
    r2 = np.random.rand(n, 1).astype(np.float32)
    sq = np.sqrt(r1)
    pts = ((1 - sq) * verts[faces[chosen, 0]] +
           sq * (1 - r2) * verts[faces[chosen, 1]] +
           sq * r2        * verts[faces[chosen, 2]])
    return pts.astype(np.float32)


def rsample(pts, n):
    """Random subsample/oversample to exactly n points."""
    if len(pts) >= n:
        idx = np.random.choice(len(pts), n, replace=False)
    else:
        idx = np.random.choice(len(pts), n, replace=True)
    return pts[idx].astype(np.float32)


def clean_pts(pts, max_pts=8192):
    """Remove NaN/Inf, optionally subsample."""
    mask = np.isfinite(pts).all(axis=1)
    pts  = pts[mask]
    if len(pts) > max_pts:
        idx = np.random.choice(len(pts), max_pts, replace=False)
        pts = pts[idx]
    return pts.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Data augmentation
# ─────────────────────────────────────────────────────────────────────────────

def mirror_to_left(tibia, bound, lm):
    """
    Flip right-side tibia to left-side space by negating the X axis.
    This ensures the model sees a single consistent bone orientation
    regardless of which side the tibia comes from.

    At inference: flip right-bone inputs before generate(), flip output back.
    """
    tibia = tibia.copy(); tibia[:, 0] *= -1
    bound = bound.copy(); bound[:, 0] *= -1
    lm    = lm.copy();   lm[:, 0]    *= -1
    return tibia, bound, lm


def augment(tibia, bound, lm):
    """
    Random rotation around Z-axis (vertical axis for tibia).
    All three inputs share the same rotation.
    """
    angle = np.random.uniform(0, 2 * np.pi)
    c, s  = np.cos(angle), np.sin(angle)
    R     = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

    tibia = tibia @ R.T
    bound = bound @ R.T
    lm    = lm    @ R.T
    return tibia, bound, lm


# ─────────────────────────────────────────────────────────────────────────────
# Scan directories and build raw sample list
# ─────────────────────────────────────────────────────────────────────────────

def _find_file(directory, stem, suffixes):
    """Try multiple filename patterns and return first match."""
    for suffix in suffixes:
        p = os.path.join(directory, f"{stem}{suffix}")
        if os.path.exists(p):
            return p
    return None


def _extract_stem(filename):
    """
    Strip trailing suffixes to get base bone ID.

    Actual naming on GPU machine:
      PLY      : 100_Tibia_L_rmesh_registered.ply  → stem = 100_Tibia_L
      LM       : 100_Tibia_L_rmesh_landmarks.csv
      Boundary : 100_Tibia_L_rmesh_boundary.ply
    """
    name = os.path.splitext(filename)[0]
    # Order matters — strip longest suffixes first
    for suffix in ['_rmesh_registered', '_rmesh_reg', '_rmesh', '_mesh',
                   '_registered', '_reg']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break   # only strip one suffix (the first match)
    return name


def load_all_raw():
    """
    Scan left + right directories and return list of raw sample dicts.
    Each dict has: id, side, verts, faces, bound, lm
    """
    samples = []

    for side, ply_dir, lm_dir, bound_dir in [
        ('L', cfg.LEFT_PLY,  cfg.LEFT_LM,  cfg.LEFT_BOUND),
        ('R', cfg.RIGHT_PLY, cfg.RIGHT_LM, cfg.RIGHT_BOUND),
    ]:
        if not os.path.isdir(ply_dir):
            print(f"  WARNING: {ply_dir} not found — skipping {side} side")
            continue

        ply_files = sorted(f for f in os.listdir(ply_dir) if f.endswith('.ply'))
        print(f"{side}: {len(ply_files)}/{len(ply_files)} loaded", end='')

        count = 0
        for fname in ply_files:
            ply_path = os.path.join(ply_dir, fname)
            stem     = _extract_stem(fname)         # e.g. 37_Tibia_R
            bone_id  = f"{side}_{stem}"             # e.g. R_37_Tibia_R

            # ── Find landmark file ────────────────────────────────────────────
            # Actual: 100_Tibia_L_rmesh_landmarks.csv  (stem = 100_Tibia_L)
            lm_path = _find_file(lm_dir, stem, [
                '_rmesh_landmarks.csv',     # ← actual format on GPU machine
                '_landmarks.csv',
                '_rmesh_landmarks.txt',
                '_landmarks.txt',
                '.csv'])
            if lm_path is None:
                continue

            # ── Find boundary file ────────────────────────────────────────────
            # Actual: 100_Tibia_L_rmesh_boundary.ply  (stem = 100_Tibia_L)
            bound_path = _find_file(bound_dir, stem, [
                '_rmesh_boundary.ply',      # ← actual format on GPU machine
                '_boundary.ply',
                '_bound.ply',
                '.ply'])
            if bound_path is None:
                continue   # skip if no boundary

            # ── Load everything ───────────────────────────────────────────────
            try:
                verts, faces = load_ply_mesh(ply_path)
                if len(verts) == 0:
                    continue

                # ── Clean NaN/Inf vertices immediately after loading ──────────
                # Binary PLY parsers can produce Inf/NaN on misaligned reads.
                # Must clean BEFORE mirroring (Inf * -1 = -Inf, still bad).
                valid = np.isfinite(verts).all(axis=1)
                if not valid.all():
                    # Remap faces to new vertex indices
                    if faces is not None and len(faces) > 0:
                        old_to_new        = np.full(len(verts), -1, dtype=np.int64)
                        old_to_new[valid] = np.arange(valid.sum())
                        face_ok           = valid[faces].all(axis=1)
                        faces             = old_to_new[faces[face_ok]]
                        if len(faces) == 0:
                            faces = None
                    verts = verts[valid]

                if len(verts) < 100:   # skip near-empty meshes
                    continue

                lm_dict = load_lm(lm_path)
                la = np.zeros((cfg.N_LM, 3), dtype=np.float32)
                found = 0
                for i, name in enumerate(cfg.LM_NAMES):
                    if name in lm_dict:
                        la[i] = lm_dict[name]
                        found += 1
                if found < cfg.N_LM // 2:
                    continue

                bound = load_pts_ply(bound_path)
                bound = clean_pts(bound)
                if len(bound) == 0:
                    continue

                # ── Mirror right bones to left-side space ─────────────────────
                if side == 'R':
                    verts = verts.copy(); verts[:, 0] *= -1
                    bound = bound.copy(); bound[:, 0] *= -1
                    la    = la.copy();    la[:, 0]    *= -1

                samples.append({
                    'id'   : bone_id,
                    'side' : side,
                    'verts': verts,
                    'faces': faces,
                    'bound': bound,
                    'lm'   : la,
                })
                count += 1

            except Exception as e:
                print(f"\n  WARN: skipping {fname}: {e}")
                continue

        print(f"\r{side}: {count}/{len(ply_files)} loaded")

    print(f"Total loaded: {len(samples)} bones")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────────────

def compute_norm_stats(samples):
    """
    Compute global centroid and scale from ALL vertices in training set.
    Uses nanmean/nanstd to ignore any residual NaN vertices in raw meshes.
    """
    all_pts = np.concatenate([s['verts'] for s in samples], axis=0)
    # Filter out any rows that still have NaN/Inf
    finite_mask = np.isfinite(all_pts).all(axis=1)
    all_pts = all_pts[finite_mask]
    if len(all_pts) == 0:
        raise ValueError("All vertices are NaN — check PLY files.")
    gc = all_pts.mean(axis=0).astype(np.float32)
    gs = float(all_pts.std())
    if gs < 1e-6:
        gs = 1.0
    return gc, gs


def save_norm_stats(path, gc, gs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'centroid': gc.tolist(), 'scale': float(gs)}, f)
    print(f"  Norm stats saved → {path}")


def load_norm_stats(path):
    with open(path) as f:
        d = json.load(f)
    gc = np.array(d['centroid'], dtype=np.float32)
    gs = float(d['scale'])
    return gc, gs


def normalise_raw(samples, gc, gs):
    """Normalise all geometry to zero-mean / unit-scale in-place (copy)."""
    out = []
    for s in samples:
        s2 = dict(s)
        s2['verts'] = (s['verts'] - gc) / gs
        s2['bound'] = (s['bound'] - gc) / gs
        s2['lm']    = (s['lm']    - gc) / gs
        out.append(s2)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Train / Test split
# ─────────────────────────────────────────────────────────────────────────────

def split_data(samples, train_frac=cfg.TRAIN_FRAC, seed=cfg.SPLIT_SEED):
    """
    Random 90/10 split, stratified by side (L/R) so both sides are
    equally represented in train and test.
    Returns (train_raw, test_raw, test_ids_set).
    """
    rng = np.random.default_rng(seed)

    left  = [s for s in samples if s['side'] == 'L']
    right = [s for s in samples if s['side'] == 'R']

    def _split_side(lst):
        idx   = rng.permutation(len(lst))
        n_tr  = max(1, int(len(lst) * train_frac))
        return [lst[i] for i in idx[:n_tr]], [lst[i] for i in idx[n_tr:]]

    l_tr, l_te = _split_side(left)
    r_tr, r_te = _split_side(right)

    train_raw = l_tr + r_tr
    test_raw  = l_te + r_te
    test_ids  = {s['id'] for s in test_raw}

    print(f"  Split : train={len(train_raw)}  test={len(test_raw)}")
    print(f"    L train={len(l_tr)}  L test={len(l_te)}")
    print(f"    R train={len(r_tr)}  R test={len(r_te)}")

    return train_raw, test_raw, test_ids


def save_test_ids(path, test_ids):
    """Save sorted test bone IDs to a text file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for tid in sorted(test_ids):
            f.write(tid + '\n')
    print(f"  Test IDs saved → {path}  ({len(test_ids)} bones)")


def load_test_ids(path):
    """Load test bone IDs from text file → set."""
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TibiaDataset(Dataset):
    """Training dataset with augmentation and virtual epoch expansion."""
    def __init__(self, samples, repeats=cfg.REPEATS):
        self.samples = samples
        self.repeats = repeats
        print(f"  TibiaDataset: {len(samples)} bones × {repeats} "
              f"= {len(samples)*repeats} samples/epoch")

    def __len__(self):
        return len(self.samples) * self.repeats

    def __getitem__(self, idx):
        s     = self.samples[idx % len(self.samples)]
        tibia = sample_mesh_surface(s['verts'], s['faces'], cfg.N_TIBIA)
        bound = rsample(s['bound'], cfg.N_BOUND)
        lm    = s['lm'].copy()
        tibia, bound, lm = augment(tibia, bound, lm)
        return {
            'tibia': torch.from_numpy(tibia).float(),
            'bound': torch.from_numpy(bound).float(),
            'lm'   : torch.from_numpy(lm).float(),
        }


class TibiaTestDataset(Dataset):
    """Test dataset — no augmentation, no repeats."""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s     = self.samples[idx]
        tibia = sample_mesh_surface(s['verts'], s['faces'], cfg.N_TIBIA)
        bound = rsample(s['bound'], cfg.N_BOUND)
        lm    = s['lm'].copy()
        return {
            'id'   : s['id'],
            'tibia': torch.from_numpy(tibia).float(),
            'bound': torch.from_numpy(bound).float(),
            'lm'   : torch.from_numpy(lm).float(),
        }
