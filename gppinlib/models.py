# %% [markdown]
# Core GNN link-prediction stack (GCNConv backbone).

# %%
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


# ────────────────────────────────────────────────
# Encoders
# ────────────────────────────────────────────────
class GCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, 2 * hid_dim)
        self.bn1 = nn.BatchNorm1d(2 * hid_dim)
        self.conv2 = GCNConv(2 * hid_dim, hid_dim)
        self.bn2 = nn.BatchNorm1d(hid_dim)
        self.drop = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        return x


class LinkPredictor(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int, dropout: float = 0.5):
        super().__init__()
        self.encoder = GCNEncoder(in_dim, emb_dim, dropout)

    def forward(self, data):
        # data expected to have .x and .edge_index
        return self.encoder(data.x, data.edge_index)

    @staticmethod
    def decode(z, edge_idx):
        # dot-product decoder
        src, dst = edge_idx
        return (z[src] * z[dst]).sum(dim=1)


# ────────────────────────────────────────────────
# Training helpers
# ────────────────────────────────────────────────
def _link_loss(pos_scores, neg_scores):
    pos = -torch.log(torch.sigmoid(pos_scores) + 1e-15).mean()
    neg = -torch.log(1 - torch.sigmoid(neg_scores) + 1e-15).mean()
    return pos + neg


def train_epoch(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    z = model(data)
    pos_s = LinkPredictor.decode(z, data.pos_edge_label_index)
    neg_s = LinkPredictor.decode(z, data.neg_edge_label_index)
    loss = _link_loss(pos_s, neg_s)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    z = model(data)
    pos_s = LinkPredictor.decode(z, data.pos_edge_label_index)
    neg_s = LinkPredictor.decode(z, data.neg_edge_label_index)
    y_true = torch.cat([torch.ones_like(pos_s), torch.zeros_like(neg_s)])
    y_prob = torch.sigmoid(torch.cat([pos_s, neg_s]))
    y_pred = (y_prob >= 0.5).float()
    acc = balanced_accuracy_score(y_true.cpu(), y_pred.cpu())
    auc = roc_auc_score(y_true.cpu(), y_prob.cpu())
    return acc, auc, _link_loss(pos_s, neg_s).item()
