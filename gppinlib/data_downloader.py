# %% [markdown]
# Gold-standard PPI downloader & pre-processor.

# %%
import re, ftplib, requests
from pathlib import Path
import pandas as pd
from Bio import SeqIO
from collections import defaultdict
from .paths import EXTERNAL_DATA_DIR
from .logger_setup import get_logger

log = get_logger()


class DataDownloader:
    """
    Handles download of the Bernett et al. 2024 gold-standard PPI splits
    and lightweight cleaning (obsolete accessions, duplicates, self-loops,
    SwissProt sequence mapping).
    """

    BASE_URL = "https://api.figshare.com/v2"

    def __init__(self, outdir: Path = EXTERNAL_DATA_DIR):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.data = defaultdict(dict)

    # --------------------------------------------------
    # 1. Figshare download
    # --------------------------------------------------
    def download_gs_ppi(self, article_id: int = 21591618, version: int = 3):
        meta = requests.get(
            f"{self.BASE_URL}/articles/{article_id}/versions/{version}"
        ).json()
        for f in meta["files"]:
            fp = self.outdir / f["name"]
            if fp.exists():
                log.info(f"{fp.name} already present – skip")
                continue
            log.info(f"→ Download {fp.name}")
            r = requests.get(f["download_url"], stream=True)
            r.raise_for_status()
            with open(fp, "wb") as w:
                for chunk in r.iter_content(8192):
                    w.write(chunk)

    # --------------------------------------------------
    # 2. Split loader / cleaner
    # --------------------------------------------------
    def load_split(
        self,
        split_prefix: str,
        remove_obsolete: bool = True,
        remove_dupes: bool = True,
        remove_self_loops: bool = True,
        add_sequences: bool = True,
        outdir: Path = None,
        tag: str = None,
    ) -> pd.DataFrame:
        pos = pd.read_csv(
            self.outdir / f"{split_prefix}_pos_rr.txt",
            sep=r"\s+",
            header=None,
            names=["InteractorA", "InteractorB"],
        )
        pos["Label"] = 1
        neg = pd.read_csv(
            self.outdir / f"{split_prefix}_neg_rr.txt",
            sep=r"\s+",
            header=None,
            names=["InteractorA", "InteractorB"],
        )
        neg["Label"] = 0
        df = pd.concat([pos, neg], ignore_index=True)

        if remove_self_loops:
            df = df[df.InteractorA != df.InteractorB]

        if remove_obsolete:
            df = self._drop_obsolete(df)

        if remove_dupes:
            df = self._drop_duplicates(df)

        if add_sequences:
            df = self._add_swissprot_sequences(df)

        if tag is not None:
            df.reset_index(drop=True).to_csv(outdir / f"{tag}.csv", index=False)
        else:
            df.reset_index(drop=True).to_csv(
                outdir / f"{split_prefix}.csv", index=False
            )

        return df

    # ------------------ helpers ------------------
    def _drop_obsolete(self, df: pd.DataFrame) -> pd.DataFrame:
        obsolete_file = self.outdir / "delac_sp.txt"
        if not obsolete_file.exists():
            log.info("Fetching obsolete accession list (SwissProt)")
            with ftplib.FTP("ftp.uniprot.org") as ftp:
                ftp.login()
                ftp.cwd(
                    "/pub/databases/uniprot/current_release/knowledgebase/complete/docs/"
                )
                with open(obsolete_file, "wb") as w:
                    ftp.retrbinary("RETR delac_sp.txt", w.write)

        obsolete = {
            line.strip() for line in open(obsolete_file) if re.match(r"^[A-Z]\d", line)
        }
        keep = ~df.InteractorA.isin(obsolete) & ~df.InteractorB.isin(obsolete)
        return df[keep]

    def _drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        key = df.apply(
            lambda r: "_".join(sorted([r.InteractorA, r.InteractorB])), axis=1
        )
        return df[~key.duplicated()].copy()

    def _add_swissprot_sequences(
        self,
        df: pd.DataFrame,
        fasta_file: Path = EXTERNAL_DATA_DIR / "human_swissprot_oneliner.fasta",
    ):
        if not fasta_file.exists():
            raise FileNotFoundError(
                "SwissProt FASTA missing – download separately or adjust path."
            )
        seqs = {rec.id: str(rec.seq) for rec in SeqIO.parse(fasta_file, "fasta")}
        df["SequenceA"] = df.InteractorA.map(seqs)
        df["SequenceB"] = df.InteractorB.map(seqs)
        df.dropna(subset=["SequenceA", "SequenceB"], inplace=True)
        return df
