"""Export per-pair filtered KGE ranks and top-k candidates from a trained run.

  test_ranks.tsv   disease, drug, kge_rank (filtered, realistic) for every test pair
  test_topk.jsonl  top-k candidates per test disease (overwrites)

Usage: python -m src.candidates.export --run results/runs/kge/RotatE_seed42 [--topk 50]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.candidates.data import load_split, node_lut
from src.candidates.train_kge import export_topk, pick_device, score_drugs
from src.eval.ranking import realistic_rank


def export_ranks(model, data, df, path: Path) -> pd.DataFrame:
    _, ent = node_lut(data)
    drug_nodes = ent[data.drug_ids]
    pos = {int(n): k for k, n in enumerate(drug_nodes)}
    diseases = np.unique(df["tail"].to_numpy())
    scores = score_drugs(model, data, diseases)
    row = {int(s): k for k, s in enumerate(diseases)}
    out = []
    for d, s in zip(df["head"], df["tail"]):
        mask = np.ones(len(drug_nodes), dtype=bool)
        for other in data.known.get(int(s), ()):
            if other != int(d) and other in pos:
                mask[pos[other]] = False
        out.append((int(s), int(d), realistic_rank(scores[row[int(s)]], pos[int(d)], mask)))
    res = pd.DataFrame(out, columns=["disease", "drug", "kge_rank"])
    res.to_csv(path, sep="\t", index=False)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    data = load_split()
    model = torch.load(args.run / "model.pt", weights_only=False).to(pick_device(args.device))
    r = export_ranks(model, data, data.test, args.run / "test_ranks.tsv")
    export_topk(model, data, data.test, args.topk, args.run / "test_topk.jsonl")
    print(f"MRR={np.mean(1 / r.kge_rank):.4f}  Hits@10={np.mean(r.kge_rank <= 10):.4f}  -> {args.run}")


if __name__ == "__main__":
    main()
