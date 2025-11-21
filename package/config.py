import logging
import sys
from pathlib import Path
from rich.console import Console

# ─────────────────────────────────────────────────────────────────────────────
# Globals & Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("xfedformer")

DEVICE = "cuda" if Path("/opt/conda/bin/nvcc").exists() else "cpu" # Check for CUDA more robustly
# Model dims
D_MODEL = 128  # Reduced for faster example, original: 256
N_HEADS = 4   # Reduced, original: 8
N_LAYERS = 2  # Reduced, original: 4
# Sequence
SEQ_LEN = 24  # Reduced, original: 60
HORIZON = 12
# Training
BATCH_SIZE = 32  # Renamed from BATCH, original: 64
LR = 1e-4       # Original: 3e-4
LOCAL_EPOCHS = 1  # Original: 2
PROX_MU = 0.01
DP_ENABLED = False  # Set to False for quicker debugging, can be True
NOISE_MULTIPLIER = 1.0
MAX_GRAD_NORM = 1.0

# Directories
DATA_DIR = Path("data_cache")
DATA_DIR.mkdir(exist_ok=True)
CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)
RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

# Holiday definition (example for KZ)
KZ_HOLIDAYS = {(3, 21), (3, 22), (3, 23), (12, 16)}  # Nauryz, Independence Day