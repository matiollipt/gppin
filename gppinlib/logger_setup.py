# %% [markdown]
# Shared logger (file + stdout).

# %%
import sys, logging
from pathlib import Path
from .paths import LOG_DIR

LOG_FILE = LOG_DIR / "g_ppin.log"
LOG_FILE.touch(exist_ok=True)

_fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
_date = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger("g-ppin")
logger.setLevel(logging.INFO)

if not logger.handlers:
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    ch = logging.StreamHandler(sys.stdout)
    for h in (fh, ch):
        h.setFormatter(logging.Formatter(_fmt, datefmt=_date))
        logger.addHandler(h)


def get_logger():  # convenience
    return logger
