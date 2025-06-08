# Graphlet Model Baseline

This report summarizes the added baseline for protein-protein interaction prediction based on graphlet graphs.

## Overview
- Each protein is represented by a graph whose nodes are structural graphlets.
- Node features encode amino-acid composition of the graphlet (20 dims).
- A `GraphletLinkPredictor` learns embeddings using two GCN layers and predicts interactions via an MLP on element-wise combinations of protein embeddings.
- Training and evaluation utilities are provided to run on a single GPU.

## Usage
Run the training script after preparing graphlet files in `data/external/graphlets` and cleaned splits in `data/processed`:

```bash
python scripts/train_graphlet_model.py
```

The model checkpoints are saved under `models/`.
