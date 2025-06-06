# %% [markdown]
# # Fetch & clean PPI splits

# %% [markdown]
# ## Imports
# %%
from gppinlib.logger_setup import get_logger
from gppinlib.data_downloader import DataDownloader
from gppinlib.paths import EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR

log = get_logger()

# %% [markdown]
# ## Download Bernett et al. 2024 PPI splits

# %%
log.info("Downloading Bernett et al. 2024 PPI splits")
dd = DataDownloader()
dd.download_gs_ppi()
log.info("Data downloaded")

# %% [markdown]
# ## Clean Bernett et al. 2024 PPI splits

# %%
log.info("Cleaning Bernett et al. 2024 PPI splits")
dd.load_split(
    "Intra1",
    remove_obsolete=True,
    remove_dupes=True,
    remove_self_loops=True,
    outdir=PROCESSED_DATA_DIR,
    tag="train",
)
dd.load_split(
    "Intra0",
    remove_obsolete=True,
    remove_dupes=True,
    remove_self_loops=True,
    outdir=PROCESSED_DATA_DIR,
    tag="val",
)
dd.load_split(
    "Intra2",
    remove_obsolete=True,
    remove_dupes=True,
    remove_self_loops=True,
    outdir=PROCESSED_DATA_DIR,
    tag="test",
)
log.info("Data cleaned")
# %%
