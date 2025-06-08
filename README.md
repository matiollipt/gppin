# gppinlib: Graph-based protein interaction prediction

## Dataset

The PPI dataset from Gold-standard PPI splits (Bernett et al. 2024) has the following properties:

* Splits: Intra-1 (163,192 training points), Intra-0 (59,260 validation points), Intra-2 (52,048 test points)
* No direct data leakage between splits
* Minimized sequence similarity w.r.t. length-normalized bitscores between training, validation, test
* Redundancy-reduction with CD-HIT (no sequence similarity >40% between proteins in each split)

## Graphlet baseline

Graphlet graphs for each protein should be placed under `data/external/graphlets`. Use the training script to fit the baseline GCN model:

```bash
python scripts/train_graphlet_model.py
```

A summary of the model is provided in `.reports/report.md`.
