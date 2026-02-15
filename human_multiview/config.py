"""Constants and configuration."""

from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).parent.parent
REPOS_DIR = REPO_ROOT / "repos"
RESULTS_DIR = REPO_ROOT / "results"

# VGGT architecture
PATCH_SIZE = 14
IMAGE_SIZE = 518
PATCH_GRID = 37  # 518 // 14
N_LAYERS = 24
N_HEADS = 16
SPECIAL_TOKENS = 5  # camera + register tokens
N_PATCHES = PATCH_GRID * PATCH_GRID  # 1369
TOKENS_PER_IMAGE = SPECIAL_TOKENS + N_PATCHES  # 1374

# HuggingFace model IDs
MOCHI_DATASET = "tzler/MOCHI"
VGGT_MODEL = "facebook/VGGT-1B"
DUST3R_MODEL = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
MAST3R_MODEL = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
PI3_MODEL = "yyfz233/Pi3"

# 3R family preprocessing
THREE_R_IMAGE_SIZE = 512
THREE_R_PATCH_SIZE = 16

# Pi3 preprocessing
PI3_IMAGE_SIZE = 560

# Evaluation
CONFIDENCE_METRICS = ["mean", "median", "max"]
MASKING_OPTIONS = [True, False]
SIMILARITY_METRICS = ["mean", "max", "global_pool"]
