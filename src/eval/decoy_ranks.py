"""Training-time poisoning: does the targeted attack promote decoy drugs in the KGE?

For every (disease, decoy) in decoys.tsv, compute the decoy's rank among all
drugs (known train/valid indications filtered) under the clean and the
poisoned KGE model.

Usage:
  python -m src.eval.decoy_ranks --poison data/poisoned/targeted_n3_seed13 \
      --clean-run results/runs/kge/RotatE_seed42 \
      --poisoned-run results/runs/kge/RotatE_seed42_poison_targeted_n3_seed13
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.candidates.data import load_split, node_lut
from src.candidates.train_kge import pick_device, score_drugs
from src.eval.ranking import realistic_rank


def decoy_ranks(run: Path, data, decoys: pd.DataFrame) -> np.ndarray:
    model = torch.load(run / "model.pt", weights_only=False).to(pick_device("auto"))
    _, ent = node_lut(data)
    drug_nodes = ent[data.drug_ids]
    pos = {int(n): k for k, n in enumerate(drug_nodes)}
    diseases = decoys["disease"].to_numpy()
    scores = score_drugs(model, data, diseases)
    ranks = []
    for i, (s, d) in enumerate(zip(decoys["disease"], decoys["decoy_drug"])):
        mask = np.ones(len(drug_nodes), dtype=bool)
        for other in data.known.get(int(s), ()):
            if other in pos:
                mask[pos[other]] = False
        mask[pos[int(d)]] = True
        ranks.append(realistic_rank(scores[i], pos[int(d)], mask))
    return np.array(ranks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poison", type=Path, required=True)
    ap.add_argument("--clean-run", type=Path, required=True)
    ap.add_argument("--poisoned-run", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    data = load_split()
    decoys = pd.read_csv(args.poison / "decoys.tsv", sep="\t")
    c = decoy_ranks(args.clean_run, data, decoys)
    p = decoy_ranks(args.poisoned_run, data, decoys)
    decoys["rank_clean"], decoys["rank_poisoned"] = c, p
    res = {
        "n": int(len(decoys)),
        "median_rank_clean": float(np.median(c)), "median_rank_poisoned": float(np.median(p)),
        "top1_clean": float(np.mean(c <= 1)), "top1_poisoned": float(np.mean(p <= 1)),
        "top10_clean": float(np.mean(c <= 10)), "top10_poisoned": float(np.mean(p <= 10)),
        "frac_promoted": float(np.mean(p < c)),
    }
    out = args.out or args.poisoned_run / "decoy_ranks.json"
    out.write_text(json.dumps(res, indent=2))
    decoys.to_csv(out.with_suffix(".tsv"), sep="\t", index=False)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
