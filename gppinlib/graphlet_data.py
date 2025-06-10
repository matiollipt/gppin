# ruff: noqa: E402
# %%
"""Graphlet data loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from .paths import GRAPHLET_DIR


class GraphletLoader:
    """Load per-protein graphlet graphs from disk."""

    def __init__(self, graph_dir: Path = GRAPHLET_DIR):
        self.graph_dir = Path(graph_dir)

    def __call__(self, pid: str) -> Data:
        fp = self.graph_dir / f"{pid}.pt"
        if not fp.exists():
            raise FileNotFoundError(f"Graphlet file missing: {fp}")
        return torch.load(fp)


class GraphletPairDataset(Dataset):
    """Return pairs of graphlet graphs for link prediction."""

    def __init__(self, dataframe, graph_dir: Path = GRAPHLET_DIR):
        self.df = dataframe.reset_index(drop=True)
        self.loader = GraphletLoader(graph_dir)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Data, Data, float]:
        row = self.df.iloc[idx]
        g_a = self.loader(row.InteractorA)
        g_b = self.loader(row.InteractorB)
        return g_a, g_b, float(row.Label)


def collate_graphlet_pairs(batch):
    g1, g2, y = zip(*batch)
    return (
        Batch.from_data_list(g1),
        Batch.from_data_list(g2),
        torch.tensor(y, dtype=torch.float32),
    )
