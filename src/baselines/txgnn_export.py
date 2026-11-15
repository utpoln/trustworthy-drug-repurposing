"""Turn TxGNN score matrices into the same files a KGE run produces.

  test_ranks.tsv    disease, drug, kge_rank (filtered realistic rank, as for RotatE)
  test_topk.jsonl   top-k candidates per test disease (same schema as RotatE's)
  valid_topk.jsonl  same for validation diseases
  metrics.json      MRR / Hits@k on test and valid

Filtering is identical to src/candidates/export.py: other known indications of
the disease (train/valid/test) are removed before ranking a target. The
column name `kge_rank` is kept so downstream code (run_llm, rerank) works
unchanged with TxGNN as the candidate generator.

Usage: python -m src.baselines.txgnn_export --run results/runs/txgnn/TxGNN_seed42
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.candidates.data import load_split
from src.eval.ranking import ranking_metrics, realistic_rank


def ranks_for(df, S, dis_row, drug_pos, known):
    out = []
    for d, s in zip(df["head"], df["tail"]):
        mask = np.ones(S.shape[1], dtype=bool)
        for other in known.get(int(s), ()):
            if other != int(d) and other in drug_pos:
                mask[drug_pos[other]] = False
        out.append((int(s), int(d), realistic_rank(S[dis_row[int(s)]], drug_pos[int(d)], mask)))
    return pd.DataFrame(out, columns=["disease", "drug", "kge_rank"])


def topk(df, S, dis_row, drugs, names, known, k, path):
    targets = {}
    for d, s in zip(df["head"], df["tail"]):
        targets.setdefault(int(s), set()).add(int(d))
    with open(path, "w") as f:
        for s in sorted(targets):
            sc = S[dis_row[s]].astype(float).copy()
            sc[np.isin(drugs, list(known.get(s, set()) - targets[s]))] = -np.inf
            top = np.argsort(-sc, kind="stable")[:k]
            f.write(json.dumps({
                "disease": s, "disease_name": str(names[s]), "gold_drugs": sorted(targets[s]),
                "candidates": [{"drug": int(drugs[j]), "drug_name": str(names[int(drugs[j])]),
                                "kge_score": float(sc[j]), "kge_rank": r + 1} for r, j in enumerate(top)],
            }) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--topk", type=int, default=50)
    args = ap.parse_args()
    data = load_split()
    names = data.nodes.set_index("node_index")["node_name"]
    metrics = {}
    for split, df in (("test", data.test), ("valid", data.valid)):
        z = np.load(args.run / f"scores_{split}.npz")
        S, dis, drugs = z["scores"], z["diseases"], z["drugs"]
        dis_row = {int(s): i for i, s in enumerate(dis)}
        drug_pos = {int(d): j for j, d in enumerate(drugs)}
        r = ranks_for(df, S, dis_row, drug_pos, data.known)
        metrics[split] = ranking_metrics(r["kge_rank"].to_numpy(float))
        if split == "test":
            r.to_csv(args.run / "test_ranks.tsv", sep="\t", index=False)
        topk(df, S, dis_row, drugs, names, data.known, args.topk, args.run / f"{split}_topk.jsonl")
    (args.run / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
