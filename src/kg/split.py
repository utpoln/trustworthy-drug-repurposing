"""Build train / valid / test splits of PrimeKG for indication link prediction.

Design
------
* PrimeKG lists every edge in both directions. We keep ONE canonical
  direction per edge:
    - drug-disease relations (indication, contraindication, off-label use)
      are always drug -> disease;
    - other relations between two different node types keep the most
      common orientation in kg.csv;
    - relations between nodes of the same type (protein_protein, drug_drug,
      *_parent-child) keep (min(index), max(index)).
* Target edges: unique (drug, disease) `indication` pairs. These are split
  at random into train / valid / test (default 80/10/10, seed 42).
* Leakage control: for every valid/test pair (d, s), we also delete any
  `off-label use` or `contraindication` edge between d and s from the
  training graph, because those edges would give the answer away.
* Training graph = all non-indication edges (minus the leakage edges above)
  + train indications. Valid/test files contain indication triples only.

Outputs (data/processed/primekg_v2.1_seed{seed}/)
-----------------------------------------------
  nodes.tsv.gz    node_index, node_id, node_type, node_name, node_source
  train.tsv.gz    head, relation, tail      (node_index integers)
  valid.tsv       head, relation, tail      (indication only)
  test.tsv        head, relation, tail      (indication only)
  split_meta.json provenance: source md5, seed, ratios, counts, removals

Usage:
    python -m src.kg.split [--seed 42] [--valid 0.1] [--test 0.1]
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KG = ROOT / "data" / "raw" / "kg.csv"
OUT_BASE = ROOT / "data" / "processed"
EXPECTED_MD5 = "aac8191d4fbc5bf09cdf8c3c78b4e75f"
DATA_VERSION = "PrimeKG v2.1 (Harvard Dataverse doi:10.7910/DVN/IXA7BM, kg.csv id 6180620)"

TARGET = "indication"
DRUG_DISEASE = {"indication", "contraindication", "off-label use"}
LEAKY = {"contraindication", "off-label use"}


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={"x_index": np.int64, "y_index": np.int64},
        low_memory=False,
    )


def build_nodes(kg: pd.DataFrame) -> pd.DataFrame:
    cols = ["index", "id", "type", "name", "source"]
    x = kg[[f"x_{c}" for c in cols]].set_axis([f"node_{c}" for c in cols], axis=1)
    y = kg[[f"y_{c}" for c in cols]].set_axis([f"node_{c}" for c in cols], axis=1)
    nodes = pd.concat([x, y], ignore_index=True).drop_duplicates("node_index")
    return nodes.sort_values("node_index").reset_index(drop=True)


def canonical_triples(kg: pd.DataFrame) -> pd.DataFrame:
    """One row per undirected edge, oriented as described in the module docstring."""
    parts = []
    for rel, g in kg.groupby("relation", sort=True):
        if rel in DRUG_DISEASE:
            g = g[(g["x_type"] == "drug") & (g["y_type"] == "disease")]
            h, t = g["x_index"].to_numpy(), g["y_index"].to_numpy()
        elif (g["x_type"] == g["y_type"]).all():
            a, b = g["x_index"].to_numpy(), g["y_index"].to_numpy()
            h, t = np.minimum(a, b), np.maximum(a, b)
        else:
            orient = (g["x_type"] + ">" + g["y_type"]).value_counts().idxmax()
            xt, yt = orient.split(">", 1)
            g = g[(g["x_type"] == xt) & (g["y_type"] == yt)]
            h, t = g["x_index"].to_numpy(), g["y_index"].to_numpy()
        parts.append(pd.DataFrame({"head": h, "relation": rel, "tail": t}))
    return pd.concat(parts, ignore_index=True).drop_duplicates().reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", type=Path, default=DEFAULT_KG)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--valid", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    args = ap.parse_args()

    md5 = md5sum(args.kg)
    if md5 != EXPECTED_MD5:
        raise SystemExit(f"kg.csv md5 {md5} != expected {EXPECTED_MD5}")

    print("Loading kg.csv ...")
    kg = load(args.kg)
    nodes = build_nodes(kg)
    triples = canonical_triples(kg)
    print(f"  {len(nodes):,} nodes, {len(triples):,} canonical triples")

    # Split indication pairs
    ind = triples[triples["relation"] == TARGET].reset_index(drop=True)
    rest = triples[triples["relation"] != TARGET]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(ind))
    n_test = int(round(args.test * len(ind)))
    n_valid = int(round(args.valid * len(ind)))
    test = ind.iloc[perm[:n_test]]
    valid = ind.iloc[perm[n_test:n_test + n_valid]]
    train_ind = ind.iloc[perm[n_test + n_valid:]]

    # Remove leaky drug-disease edges for held-out pairs
    held = set(zip(pd.concat([valid, test])["head"], pd.concat([valid, test])["tail"]))
    pair = list(zip(rest["head"], rest["tail"]))
    leak_mask = rest["relation"].isin(LEAKY).to_numpy() & np.fromiter(
        (p in held for p in pair), dtype=bool, count=len(pair)
    )
    removed = rest[leak_mask]
    train = pd.concat([rest[~leak_mask], train_ind], ignore_index=True)

    # Sanity checks
    train_pairs = set(zip(train_ind["head"], train_ind["tail"]))
    assert not (train_pairs & held), "held-out indication found in train"
    assert not set(zip(valid["head"], valid["tail"])) & set(zip(test["head"], test["tail"]))
    leftover = train[train["relation"].isin(LEAKY)]
    assert not (set(zip(leftover["head"], leftover["tail"])) & held), "leaky edge left in train"

    train_entities = set(train["head"]).union(train["tail"])
    def unseen(df):
        return int((~df["head"].isin(train_entities) | ~df["tail"].isin(train_entities)).sum())

    out = OUT_BASE / f"primekg_v2.1_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(out / "nodes.tsv.gz", sep="\t", index=False)
    train.to_csv(out / "train.tsv.gz", sep="\t", index=False)
    valid.to_csv(out / "valid.tsv", sep="\t", index=False)
    test.to_csv(out / "test.tsv", sep="\t", index=False)

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_version": DATA_VERSION,
        "source_md5": md5,
        "seed": args.seed,
        "ratios": {"train": 1 - args.valid - args.test, "valid": args.valid, "test": args.test},
        "n_nodes": len(nodes),
        "n_canonical_triples": len(triples),
        "n_relations": int(triples["relation"].nunique()),
        "indication": {
            "total": len(ind), "train": len(train_ind), "valid": len(valid), "test": len(test),
            "test_diseases": int(test["tail"].nunique()),
            "test_drugs": int(test["head"].nunique()),
        },
        "train_triples": len(train),
        "leakage_edges_removed": removed["relation"].value_counts().to_dict(),
        "heldout_with_entity_unseen_in_train": {"valid": unseen(valid), "test": unseen(test)},
    }
    (out / "split_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
