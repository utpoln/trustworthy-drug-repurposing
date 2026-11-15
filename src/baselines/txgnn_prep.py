"""Convert our PrimeKG split into TxGNN's input format (same train graph as RotatE).

TxGNN (Huang et al., Nature Medicine 2024) expects, per data folder:
  kg_directed.csv   every edge once: x_type, x_id, relation, y_type, y_id, x_idx, y_idx
                    (x_idx/y_idx = index of the node within its node type)
  train/valid/test.csv   edges of each split plus reverse edges
                    ('rev_<rel>' for relations between different node types,
                    the same name for relations within one type)

Our canonical triples (src/kg/split.py) already store each edge once with the
orientation TxGNN uses (e.g. gene/protein -[disease_protein]-> disease, whose
reverse is TxGNN's 'rev_disease_protein'). valid/test hold only indication
edges, so TxGNN is selected and evaluated on exactly our held-out pairs.
x_id is our node_index, which maps TxGNN outputs back to our ids.

Usage: python -m src.baselines.txgnn_prep [--train-file data/poisoned/<x>/train.tsv.gz --out ...]
"""
import os
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.provenance.log import file_md5, now_utc

ROOT = Path(__file__).resolve().parents[2]
SPLIT = Path(os.environ["KG_SPLIT"]).resolve() if os.environ.get("KG_SPLIT") else ROOT / "data" / "processed" / "primekg_v2.1_seed42"  # override with env KG_SPLIT


def to_txgnn(df: pd.DataFrame, ntype: dict, idx: dict) -> pd.DataFrame:
    out = pd.DataFrame({
        "x_type": df["head"].map(ntype), "x_id": df["head"].astype(str), "relation": df["relation"],
        "y_type": df["tail"].map(ntype), "y_id": df["tail"].astype(str),
        "x_idx": df["head"].map(idx).astype(float), "y_idx": df["tail"].map(idx).astype(float),
    })
    assert out.notna().all().all(), "unmapped node"
    return out


def with_reverse(df: pd.DataFrame) -> pd.DataFrame:
    rev = df.rename(columns={"x_type": "y_type", "x_id": "y_id", "x_idx": "y_idx",
                             "y_type": "x_type", "y_id": "x_id", "y_idx": "x_idx"})[df.columns]
    hetero = rev["x_type"] != rev["y_type"]
    rev.loc[hetero, "relation"] = "rev_" + rev.loc[hetero, "relation"]
    return pd.concat([df, rev], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", type=Path, default=SPLIT / "train.tsv.gz")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "txgnn_seed42")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    nodes = pd.read_csv(SPLIT / "nodes.tsv.gz", sep="\t", low_memory=False)
    ntype = dict(zip(nodes["node_index"], nodes["node_type"]))
    idx = {}
    for t, g in nodes.groupby("node_type"):
        idx.update(dict(zip(sorted(g["node_index"]), range(len(g)))))

    train = pd.read_csv(args.train_file, sep="\t")
    valid = pd.read_csv(SPLIT / "valid.tsv", sep="\t")
    test = pd.read_csv(SPLIT / "test.tsv", sep="\t")

    tr, va, te = (to_txgnn(d, ntype, idx) for d in (train, valid, test))
    kg = pd.concat([tr, va, te], ignore_index=True)
    # make sure every node exists in kg_directed (TxGNN sizes node types from it)
    kg.to_csv(args.out / "kg_directed.csv", index=False)
    for name, d in (("train", tr), ("valid", va), ("test", te)):
        with_reverse(d).to_csv(args.out / f"{name}.csv", index=False)
    pd.DataFrame({"node_index": list(idx), "type_idx": list(idx.values()),
                  "node_type": [ntype[n] for n in idx]}).to_csv(args.out / "node_map.csv", index=False)

    meta = {"created_utc": now_utc(), "train_file": str(args.train_file),
            "train_md5": file_md5(str(args.train_file)), "n_train": len(tr), "n_valid": len(va), "n_test": len(te),
            "n_nodes_by_type": nodes["node_type"].value_counts().to_dict()}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
