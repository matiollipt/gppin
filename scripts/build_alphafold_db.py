# %% [markdown]
# Build a local AlphaFold database and derive residue graphs.
# %%
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm

from gppinlib.paths import PROCESSED_DATA_DIR, ALPHAFOLD_DIR, GRAPHLET_DIR
from gppinlib.pdb_utils import download_alphafold, pdb_to_graph
from gppinlib.protein_embedding import ProteinEmbedding

MAX_LEN = 1500

# %%

def main(max_len: int = MAX_LEN):
    dfs = [pd.read_csv(PROCESSED_DATA_DIR / f"{s}.csv") for s in ["train", "val", "test"]]
    all_df = pd.concat(dfs, ignore_index=True)
    ids = set(all_df.InteractorA) | set(all_df.InteractorB)
    keep = set()
    pdb_dir = ALPHAFOLD_DIR / "pdbs"
    for pid in tqdm(sorted(ids), desc="PDB"):
        try:
            fp = download_alphafold(pid, pdb_dir)
            g = pdb_to_graph(fp)
            if g.num_nodes > max_len:
                continue
            torch.save(g, GRAPHLET_DIR / f"{pid}.pt")
            keep.add(pid)
        except Exception:
            continue
    for name, df in zip(["train", "val", "test"], dfs):
        df = df[df.InteractorA.isin(keep) & df.InteractorB.isin(keep)]
        df.to_csv(PROCESSED_DATA_DIR / f"{name}_filtered.csv", index=False)
    emb = ProteinEmbedding()
    emb.embed_dataframe(pd.concat(dfs), per_residue=False)

# %%
if __name__ == "__main__":
    main()
