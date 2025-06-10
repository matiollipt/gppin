# ruff: noqa: E402
# %%
from __future__ import annotations
"""Graph neural network for graphlet-based protein representation."""

from typing import Tuple
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GraphletEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = dropout

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, data.edge_index))
        return global_mean_pool(x, data.batch)


class GraphletLinkPredictor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = GraphletEncoder(in_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, g1, g2):
        h1 = self.encoder(g1)
        h2 = self.encoder(g2)
        pair = torch.cat([h1 * h2, torch.abs(h1 - h2)], dim=1)
        return self.classifier(pair).squeeze(-1)


# --------------------------------------------------
# Training helpers
# --------------------------------------------------

def train_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for g1, g2, y in loader:
        g1, g2, y = g1.to(device), g2.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(g1, g2)
        loss = F.binary_cross_entropy_with_logits(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
    return total_loss / len(loader.dataset)



@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[float, float, float]:
    model.eval()
    probs, targets = [], []
    total_loss = 0.0
    for g1, g2, y in loader:
        g1, g2 = g1.to(device), g2.to(device)
        out = torch.sigmoid(model(g1, g2))
        loss = F.binary_cross_entropy(out, y.to(device))
        total_loss += loss.item() * y.size(0)
        probs.append(out.cpu())
        targets.append(y)
    prob = torch.cat(probs)
    y_true = torch.cat(targets)
    pred = (prob >= 0.5).float()
    acc = balanced_accuracy_score(y_true, pred)
    auc = roc_auc_score(y_true, prob)
    return acc, auc, total_loss / len(loader.dataset)
