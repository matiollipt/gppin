# %% [markdown]
# Thin PyTorch utils.Dataset wrapper for PPI edge-pairs.

# %%
from torch.utils.data import Dataset

class GraphDataset(Dataset):
    """
    Returns (proteinA_id, proteinB_id, label) triples
    and can be further wrapped by a downstream collate_fn
    that loads embeddings/graphlets on-the-fly.
    """
    def __init__(self, dataframe,
                 col_a: str = "InteractorA",
                 col_b: str = "InteractorB",
                 col_y: str = "Label"):
        self.df = dataframe
        self.a  = col_a; self.b = col_b; self.y = col_y

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row[self.a], row[self.b], row[self.y]
