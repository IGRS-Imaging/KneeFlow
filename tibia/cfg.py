"""
Tibia OT Flow Matching — Configuration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input  : 8 tibia landmarks + tibial boundary points (PLY)
Output : full tibia point cloud (8192 pts)
Dataset: 324 bones — 90% train / 10% test (seed=42)
"""
import os

BASE = "/home/imaging/Desktop/Ragavan_Tibia"

LEFT_PLY    = os.path.join(BASE, "left",  "registered_ply")
LEFT_LM     = os.path.join(BASE, "left",  "registered_landmarks")
LEFT_BOUND  = os.path.join(BASE, "left",  "left_boundary_points")

RIGHT_PLY   = os.path.join(BASE, "right", "registered_ply")
RIGHT_LM    = os.path.join(BASE, "right", "registered_landmarks")
RIGHT_BOUND = os.path.join(BASE, "right", "right_boundary_points")

OUTPUT      = os.path.join(BASE, "output_Tibia")
CKPT_DIR    = os.path.join(OUTPUT, "checkpoints")
SAMPLE_DIR  = os.path.join(OUTPUT, "samples")
TEST_GT_DIR = os.path.join(OUTPUT, "test_bones_gt")  # GT 8192-pt PLYs saved here

# ── Data ───────────────────────────────────────────────────────────────────────
N_TIBIA  = 8192   # output point cloud size
N_BOUND  = 1024   # boundary points sampled
N_LM     = 8      # tibia landmarks

# 8 landmarks collectible during imageless TKA surgery
LM_NAMES = [
    'Tibial Knee Centre',
    'Medial Plateau',
    'Lateral Plateau',
    'Tibial Tuberosity',
    'Medial Malleolus',
    'Lateral Malleolus',
    'Ankle Centre',
    'Posterior Cruciate Ligament',
]

# ── Architecture (same as femur OT-FM) ────────────────────────────────────────
ENC_DIM    = 384
ENC_DEPTH  = 5
ENC_HEADS  = 6
ENC_K      = 20
ENC_FPS    = 512

LM_DIM     = 192
LM_DEPTH   = 4

COND_DIM   = 512

VF_DIM     = 384
VF_DEPTH   = 8
VF_HEADS   = 6
VF_K       = 20
VF_FPS     = 2048

FFN_EXP    = 2

# ── Training ──────────────────────────────────────────────────────────────────
BATCH        = 6     # ~18-20GB on 24GB GPU
LR           = 1e-4
WD           = 5e-5
EPOCHS       = 1000
REPEATS      = 15    # reduced from 20 — 25% less memory churn per epoch (~375s vs ~500s)
SIGMA_MIN    = 1e-4
AMP          = True
GRAD_CLIP    = 0.5
WARMUP       = 50    # shorter warmup at larger batch
SAVE_EVERY   = 50
SAMPLE_EVERY = 100

# ── Split ──────────────────────────────────────────────────────────────────────
TRAIN_FRAC   = 0.90
TEST_FRAC    = 0.10
SPLIT_SEED   = 42

# ── Evaluation ─────────────────────────────────────────────────────────────────
METRIC_EVERY = 50
NSD_TAU      = 2.0
INF_STEPS    = 200
