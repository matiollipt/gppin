# ruff: noqa: E402
# %%
"""Training entry point for graphlet-based PPI prediction."""


import pandas as pd
import torch
from torch.utils.data import DataLoader
import mlflow
from mlflow import pytorch as mlflow_torch

from gppinlib.graphlet_data import GraphletPairDataset, collate_graphlet_pairs
from gppinlib.graphlet_model import GraphletLinkPredictor, train_epoch, evaluate
from gppinlib.paths_extended import GRAPHLET_DIR, PROCESSED_DATA_DIR, MODELS_DIR


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_path = PROCESSED_DATA_DIR / "train_filtered.csv"
    val_path = PROCESSED_DATA_DIR / "val_filtered.csv"
    if not train_path.exists():
        train_path = PROCESSED_DATA_DIR / "train.csv"
        val_path = PROCESSED_DATA_DIR / "val.csv"
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    train_ds = GraphletPairDataset(train_df, GRAPHLET_DIR)
    val_ds = GraphletPairDataset(val_df, GRAPHLET_DIR)

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_graphlet_pairs)
    val_dl = DataLoader(val_ds, batch_size=32, collate_fn=collate_graphlet_pairs)

    model = GraphletLinkPredictor(in_dim=20, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    mlflow.set_experiment("graphlet_model")
    with mlflow.start_run():
        mlflow.log_params({"lr": 1e-3, "hidden_dim": 64})
        for epoch in range(10):
            loss = train_epoch(model, train_dl, optimizer, device)
            acc, auc, val_loss = evaluate(model, val_dl, device)
            mlflow.log_metrics({
                "train_loss": loss,
                "val_loss": val_loss,
                "val_acc": acc,
                "val_auc": auc,
            }, step=epoch)
            print(
                f"Epoch {epoch:02d} | train_loss {loss:.4f} | val_loss {val_loss:.4f} | val_acc {acc:.3f} | val_auc {auc:.3f}"
            )

        MODELS_DIR.mkdir(exist_ok=True, parents=True)
        mlflow_torch.save_model(model, path=str(MODELS_DIR / "graphlet_model"))


if __name__ == "__main__":
    main()
