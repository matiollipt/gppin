from __future__ import annotations

from pathlib import Path
import requests
import numpy as np
import torch
from Bio.PDB import PDBParser, is_aa
from Bio.PDB.Polypeptide import three_to_one
from scipy.spatial.distance import cdist
from torch_geometric.data import Data

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA)}


def download_alphafold(pid: str, outdir: Path) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / f"{pid}.pdb"
    if fp.exists():
        return fp
    url = f"https://alphafold.ebi.ac.uk/files/AF-{pid}-F1-model_v4.pdb"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"PDB fetch failed for {pid}")
    fp.write_bytes(r.content)
    return fp


def pdb_to_graph(fp: Path, cutoff: float = 8.0) -> Data:
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("p", fp)
    residues = [r for r in struct.get_residues() if is_aa(r)]
    feats, coords = [], []
    for r in residues:
        if "CA" not in r:
            continue
        aa = three_to_one(r.resname)
        vec = np.zeros(20, dtype=np.float32)
        if aa in AA_TO_IDX:
            vec[AA_TO_IDX[aa]] = 1.0
        feats.append(vec)
        coords.append(r["CA"].coord)
    coords = np.stack(coords)
    dmat = cdist(coords, coords)
    edge_index = np.array(np.nonzero((dmat < cutoff) & (dmat > 0)))
    return Data(
        x=torch.tensor(np.stack(feats), dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
    )
