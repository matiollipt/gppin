# %% [markdown]
# Preprocess the Bernett et al. 2024 gold-standard PPI dataset.
# %%
from gppinlib.logger_setup import get_logger
from gppinlib.data_downloader import DataDownloader
from gppinlib.paths import PROCESSED_DATA_DIR

log = get_logger()

# %%

def main():
    dd = DataDownloader()
    log.info("Downloading gold-standard PPI splits")
    dd.download_gs_ppi()
    log.info("Cleaning splits")
    for prefix, tag in [("Intra1", "train"), ("Intra0", "val"), ("Intra2", "test")]:
        dd.load_split(prefix, outdir=PROCESSED_DATA_DIR, tag=tag)

# %%
if __name__ == "__main__":
    main()
