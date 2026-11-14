"""Create poisoned copies of the training graph (robustness experiments).

Attacks
-------
targeted  For each test disease, choose a decoy drug: a non-gold candidate in
          the KGE top-k. Inject `n_paths` fake mechanistic chains
              decoy -[drug_protein: target]-> P_i -[disease_protein]-> disease
          through randomly chosen druggable proteins P_i (proteins that already
          have a drug_protein edge, so they look plausible). The decoy now has strong-looking
          but false 2-hop evidence.
random    Add fake drug_protein and disease_protein edges between random
          drug/protein and protein/disease nodes, equal to `rate` x the number
          of real edges of each relation.

Outputs (data/poisoned/<name>/):
  train.tsv.gz      poisoned training graph (same format as the clean split)
  fake_edges.tsv    every injected edge (head, relation, tail, display, attack)
  decoys.tsv        targeted only: disease, decoy drug, decoy KGE rank
  meta.json         attack settings, seed, counts, source hashes

Usage:
  python -m src.kg.poison --attack targeted --n-paths 3
  python -m src.kg.poison --attack random --rate 0.05
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.provenance.log import file_md5, now_utc

ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / "data" / "processed" / "primekg_v2.1_seed42"
KGE_RUN = ROOT / "results" / "runs" / "kge" / "RotatE_seed42"


def existing_pairs(train: pd.DataFrame, rel: str) -> set:
    t = train[train["relation"] == rel]
    return set(zip(t["head"], t["tail"]))


def targeted(train, nodes, topk_file, n_paths, k, rng):
    dp, dis_p = existing_pairs(train, "drug_protein"), existing_pairs(train, "disease_protein")
    # stealthy attacker: only use proteins that are already drug targets in the KG
    proteins = np.array(sorted({p for _, p in dp}))
    fake, decoys = [], []
    for line in open(topk_file):
        rec = json.loads(line)
        gold = set(rec["gold_drugs"])
        pool = [c for c in rec["candidates"][:k] if c["drug"] not in gold]
        if not pool:
            continue
        dec = pool[rng.integers(len(pool))]
        decoys.append((rec["disease"], dec["drug"], dec["kge_rank"]))
        for p in rng.choice(proteins, size=n_paths, replace=False):
            p = int(p)
            if (dec["drug"], p) not in dp:
                fake.append((dec["drug"], "drug_protein", p, "target"))
            # disease_protein is stored gene/protein -> disease (canonical orientation)
            if (p, rec["disease"]) not in dis_p:
                fake.append((p, "disease_protein", rec["disease"], "associated with"))
    return fake, decoys


def random_attack(train, nodes, rate, rng):
    by_type = {t: g["node_index"].to_numpy() for t, g in nodes.groupby("node_type")}
    fake = []
    specs = [("drug_protein", "drug", "gene/protein", "target"),
             ("disease_protein", "gene/protein", "disease", "associated with")]
    for rel, ht, tt, disp in specs:
        have = existing_pairs(train, rel)
        n = int(round(rate * len(have)))
        added = set()
        while len(added) < n:
            h = rng.choice(by_type[ht], size=n); t = rng.choice(by_type[tt], size=n)
            for a, b in zip(h.tolist(), t.tolist()):
                if (a, b) not in have and (a, b) not in added:
                    added.add((a, b))
                    if len(added) >= n:
                        break
        fake += [(a, rel, b, disp) for a, b in added]
    return fake


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", choices=["targeted", "random"], required=True)
    ap.add_argument("--n-paths", type=int, default=3)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--kge-run", type=Path, default=KGE_RUN)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    train = pd.read_csv(SPLIT / "train.tsv.gz", sep="\t")
    nodes = pd.read_csv(SPLIT / "nodes.tsv.gz", sep="\t", low_memory=False)

    if args.attack == "targeted":
        name = f"targeted_n{args.n_paths}_seed{args.seed}"
        fake, decoys = targeted(train, nodes, args.kge_run / "test_topk.jsonl", args.n_paths, args.k, rng)
    else:
        name = f"random_r{args.rate:g}_seed{args.seed}"
        fake, decoys = random_attack(train, nodes, args.rate, rng), []

    out = ROOT / "data" / "poisoned" / name
    out.mkdir(parents=True, exist_ok=True)
    fe = pd.DataFrame(fake, columns=["head", "relation", "tail", "display"]).drop_duplicates()
    fe["attack"] = args.attack
    poisoned = pd.concat([train, fe[["head", "relation", "tail"]]], ignore_index=True)
    poisoned.to_csv(out / "train.tsv.gz", sep="\t", index=False)
    fe.to_csv(out / "fake_edges.tsv", sep="\t", index=False)
    if decoys:
        pd.DataFrame(decoys, columns=["disease", "decoy_drug", "decoy_kge_rank"]).to_csv(
            out / "decoys.tsv", sep="\t", index=False)
    meta = {"created_utc": now_utc(), **vars(args), "kge_run": str(args.kge_run),
            "n_fake_edges": int(len(fe)), "fake_by_relation": fe["relation"].value_counts().to_dict(),
            "n_decoys": len(decoys), "clean_train_md5": file_md5(str(SPLIT / "train.tsv.gz")),
            "n_train_clean": int(len(train)), "n_train_poisoned": int(len(poisoned))}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    main()
