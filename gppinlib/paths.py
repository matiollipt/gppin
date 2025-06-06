# %% [markdown]
# Lightweight project-wide path registry.

# %%
from pathlib import Path

# ────────────────────────────────────────────────
# Directories
# ────────────────────────────────────────────────
PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMB_PER_PROTEIN_DIR = INTERIM_DATA_DIR / "per_protein"
EMB_PER_RESIDUE_DIR = INTERIM_DATA_DIR / "per_residue"
MODELS_DIR = PROJ_ROOT / "models"
LOG_DIR = PROJ_ROOT / "logs"

# ────────────────────────────────────────────────
# Canonical CSV splits
# ────────────────────────────────────────────────
TRAIN_FILE = INTERIM_DATA_DIR / "train.csv"
VAL_FILE = INTERIM_DATA_DIR / "val.csv"
TEST_FILE = INTERIM_DATA_DIR / "test.csv"

# Ensure folders exist
for d in [
    RAW_DATA_DIR,
    EXTERNAL_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    LOG_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)
