# Structural data pipeline

This document outlines the preprocessing steps to integrate AlphaFold structures with the gold-standard dataset.

1. `scripts/preprocess_dataset.py` downloads and cleans Bernett et al. PPI splits.
2. `scripts/build_alphafold_db.py` fetches AlphaFold models for every protein, stores residue-level graphs and filtered splits, and generates ProtBERT embeddings.
3. `scripts/train_graphlet_model.py` trains the GCN baseline while tracking metrics with MLflow.

Graphs are written to `data/external/graphlets` and AlphaFold files under `data/external/alphafold`.
