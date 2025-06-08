# ruff: noqa: E402
# %%
"""Training entry point for graphlet-based PPI prediction."""


import pandas as pd
import torch
from torch.utils.data import DataLoader

from gppinlib.graphlet_data import GraphletPairDataset, collate_graphlet_pairs
from gppinlib.graphlet_model import GraphletLinkPredictor, train_epoch, evaluate
from gppinlib.paths import GRAPHLET_DIR, PROCESSED_DATA_DIR, MODELS_DIR


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")

    train_ds = GraphletPairDataset(train_df, GRAPHLET_DIR)
    val_ds = GraphletPairDataset(val_df, GRAPHLET_DIR)

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_graphlet_pairs)
    val_dl = DataLoader(val_ds, batch_size=32, collate_fn=collate_graphlet_pairs)

    model = GraphletLinkPredictor(in_dim=20, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        loss = train_epoch(model, train_dl, optimizer, device)
        acc, auc, val_loss = evaluate(model, val_dl, device)
        print(f"Epoch {epoch:02d} | train_loss {loss:.4f} | val_loss {val_loss:.4f} | val_acc {acc:.3f} | val_auc {auc:.3f}")

    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), MODELS_DIR / "graphlet_model.pt")


if __name__ == "__main__":
    main()
