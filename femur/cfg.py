"""
Full Femur Flow Matching - Publication Quality Config
Target: ASD < 1.0mm, HD95 < 5mm  (matches SMART-KNEE evaluation protocol)
Split : fixed 29-bone held-out test set (selected_29_dice78_82), not random
Landmarks: 9  (collectible during imageless TKA surgery)

Input (train & inference): 9 landmarks + condylar boundary points
  — no surface encoder, no train/test mismatch
  — model forced to use landmark distances for shaft length
  — boundary encodes condylar width at distal end
"""
import os

BASE = "/home/imaging/Desktop/Ragavan"

LEFT_PLY     = os.path.join(BASE, "left",  "registered_ply")
LEFT_LM      = os.path.join(BASE, "left",  "registered_landmarks")
LEFT_BOUND   = os.path.join(BASE, "left",  "left_condyaler_points")

RIGHT_PLY    = os.path.join(BASE, "right", "registered_ply")
RIGHT_LM     = os.path.join(BASE, "right", "registered_landmarks")
RIGHT_BOUND  = os.path.join(BASE, "right", "right_condyaler_points")

OUTPUT       = os.path.join(BASE, "9L_working_output")
CKPT_DIR     = os.path.join(OUTPUT, "checkpoints")
SAMPLE_DIR   = os.path.join(OUTPUT, "samples")

# ── Data ───────────────────────────────────────────────────────────────────────
N_FEMUR = 8192   # use full mesh resolution (registered PLY has 8193 verts)
N_BOUND = 1024
N_LM    = 9

# Landmark names — 9 landmarks collectible during imageless TKA
LM_NAMES = [
    'HIP CENTRE', 'FEMUR KNEE CENTRE',
    'MEDIAL EPICONDYLE', 'LATERAL EPICONDYLE',
    'MEDIAL DISTAL CONDYLE', 'LATERAL DISTAL CONDYLE',
    'MEDIAL POSTERIOR CONDYLE', 'LATERAL POSTERIOR CONDYLE',
    'Lateral Anterior Cortex',
]

# ── Architecture ───────────────────────────────────────────────────────────────
ENC_DIM    = 384
ENC_DEPTH  = 5
ENC_HEADS  = 6
ENC_K      = 20
ENC_FPS    = 512

# SurfEnc removed — training and inference both use boundary + landmarks only

LM_DIM     = 192
LM_DEPTH   = 4

COND_DIM   = 512

VF_DIM     = 384
VF_DEPTH   = 8
VF_HEADS   = 6
VF_K       = 20
VF_FPS     = 2048

FFN_EXP    = 2

# ── Training ───────────────────────────────────────────────────────────────────
BATCH        = 4     # reduced from 6 — RTX 3090 21GB fragments after 700 epochs at batch 6
LR           = 1e-4
WD           = 5e-5   # increased from 1e-5 — stronger regularisation
EPOCHS       = 1200
REPEATS      = 20
SIGMA_MIN    = 1e-4
AMP          = True
GRAD_CLIP    = 0.5
WARMUP       = 100
SAVE_EVERY   = 50
SAMPLE_EVERY = 100

# ── Split ──────────────────────────────────────────────────────────────────────
# Unused: split is now the fixed 29-bone test set in data.py (TEST_BONES),
# not a random train/test fraction.
TRAIN_FRAC   = 0.90
TEST_FRAC    = 0.10

# ── Evaluation ─────────────────────────────────────────────────────────────────
METRIC_EVERY = 50     # increased from 100 — catch ASD peak within 50 epochs
NSD_TAU      = 2.0
INF_STEPS    = 200
