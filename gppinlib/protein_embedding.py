# %% [markdown]
# ProtBERT per-protein (and optional per-residue) embeddings.

# %%
import re, torch
import pandas as pd
from pathlib import Path
from transformers import BertTokenizer, BertModel
from tqdm.auto import tqdm
from .paths import EMB_PER_PROTEIN_DIR, EMB_PER_RESIDUE_DIR
from .logger_setup import get_logger

log = get_logger()


class ProteinEmbedding:
    def __init__(
        self,
        per_protein_dir: Path = EMB_PER_PROTEIN_DIR,
        per_residue_dir: Path = EMB_PER_RESIDUE_DIR,
    ):
        self.per_protein_dir = Path(per_protein_dir)
        self.per_residue_dir = Path(per_residue_dir)
        self.per_protein_dir.mkdir(parents=True, exist_ok=True)
        self.per_residue_dir.mkdir(parents=True, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = BertTokenizer.from_pretrained(
            "Rostlab/prot_bert", do_lower_case=False
        )
        self.model = (
            BertModel.from_pretrained("Rostlab/prot_bert").to(self.device).eval()
        )
        if self.device != "cpu":  # half-precision on GPU
            self.model = self.model.half()

    # --------------------------------------------------
    # Main API
    # --------------------------------------------------
    def embed_dataframe(
        self,
        df: pd.DataFrame,
        batch_size: int = 8,
        min_len: int = 50,
        max_len: int = 10_000,
        per_residue: bool = False,
    ):
        interactor_seq = self._collect_sequences(df)
        valid = {
            pid: seq
            for pid, seq in interactor_seq.items()
            if min_len <= len(seq) < max_len
        }
        excluded = set(interactor_seq) - set(valid)
        if excluded:
            log.info(f"Excluded {len(excluded)} proteins (length filter)")

        # Sort long→short for efficiency
        seq_items = sorted(valid.items(), key=lambda x: len(x[1]), reverse=True)

        for i in tqdm(range(0, len(seq_items), batch_size), desc="Embedding"):
            ids, seqs = zip(*seq_items[i : i + batch_size])
            toks = self._tokenize(seqs).to(self.device)
            with torch.no_grad():
                h = self.model(**toks).last_hidden_state  # (B, L, H)
            for j, pid in enumerate(ids):
                L = len(seqs[j])
                resid = h[j, :L].cpu()
                # per-protein (mean-pooled)
                torch.save(resid.mean(0), self.per_protein_dir / f"{pid}.pt")
                if per_residue:
                    torch.save(resid, self.per_residue_dir / f"{pid}.pt")

    # ------------------ helpers ------------------
    def _collect_sequences(self, df: pd.DataFrame):
        seqs = {}
        for a, b, sa, sb in zip(
            df.InteractorA, df.InteractorB, df.SequenceA, df.SequenceB
        ):
            if a not in seqs:
                seqs[a] = sa
            if b not in seqs:
                seqs[b] = sb
        return {pid: re.sub(r"[UZOB]", "X", seq) for pid, seq in seqs.items()}

    def _tokenize(self, seqs):
        spaced = [" ".join(s) for s in seqs]
        return self.tokenizer(
            spaced,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
        )
