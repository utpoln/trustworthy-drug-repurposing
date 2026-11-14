"""Train a knowledge-graph embedding model (PyKEEN) on the PrimeKG split and
evaluate drug recommendation for held-out indication edges.

For each held-out (drug, disease), all 7,957 drugs are scored as heads of
(?, indication, disease); known indications other than the target are
filtered (see src/eval/ranking.py).

Outputs (results/runs/kge/<model>_seed<seed>/):
  model.pt          trained PyKEEN model (git-ignored)
  config.json       hyper-parameters, data version, device, timings
  history.json      training loss and validation metrics per checkpoint
  metrics.json      best-epoch valid/test metrics
  test_topk.jsonl   top-k candidate drugs for every test disease (for the LLM stage)

Usage:
  python -m src.candidates.train_kge --model RotatE --epochs 30
  python -m src.candidates.train_kge --model TransE --epochs 1 --max-triples 200000   # smoke test
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from pykeen.models import model_resolver
from pykeen.training import SLCWATrainingLoop
from pykeen.triples import TriplesFactory

from src.candidates.data import DEFAULT_SPLIT, SplitData, load_split, node_lut
from src.eval.ranking import ranking_metrics, realistic_rank

ROOT = Path(__file__).resolve().parents[2]


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def score_drugs(model, data: SplitData, diseases: np.ndarray, batch: int = 16) -> np.ndarray:
    """Scores of all drugs as heads of (?, indication, disease). Shape [len(diseases), n_drugs]."""
    model.eval()
    lut, _ = node_lut(data)
    rel = data.relation_to_id["indication"]
    drug_ids = torch.as_tensor(data.drug_ids, device=model.device)
    out = []
    for i in range(0, len(diseases), batch):
        t = torch.as_tensor(lut[diseases[i:i + batch]], device=model.device)
        rt = torch.stack([torch.full_like(t, rel), t], dim=1)
        s = model.score_h(rt, heads=drug_ids)      # [b, n_drugs]
        out.append(s.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def evaluate(model, data: SplitData, df) -> dict:
    _, ent = node_lut(data)
    drug_nodes = ent[data.drug_ids]
    pos = {int(n): k for k, n in enumerate(drug_nodes)}
    diseases = np.unique(df["tail"].to_numpy())
    scores = score_drugs(model, data, diseases)
    row = {int(s): k for k, s in enumerate(diseases)}
    ranks = []
    for d, s in zip(df["head"], df["tail"]):
        sc = scores[row[int(s)]]
        mask = np.ones(len(drug_nodes), dtype=bool)
        for other in data.known.get(int(s), ()):
            if other != int(d) and other in pos:
                mask[pos[other]] = False
        ranks.append(realistic_rank(sc, pos[int(d)], mask))
    return ranking_metrics(ranks)


def export_topk(model, data: SplitData, df, k: int, path: Path) -> None:
    """Top-k drugs per test disease, excluding drugs already known (train/valid)
    to treat it. Test targets are NOT excluded, so they can be found."""
    _, ent = node_lut(data)
    drug_nodes = ent[data.drug_ids]
    names = data.nodes.set_index("node_index")["node_name"]
    test_targets = {}
    for d, s in zip(df["head"], df["tail"]):
        test_targets.setdefault(int(s), set()).add(int(d))
    diseases = np.array(sorted(test_targets))
    scores = score_drugs(model, data, diseases)
    with open(path, "w") as f:
        for i, s in enumerate(diseases):
            sc = scores[i].copy()
            exclude = data.known.get(int(s), set()) - test_targets[int(s)]
            sc[np.isin(drug_nodes, list(exclude))] = -np.inf
            top = np.argsort(-sc)[:k]
            f.write(json.dumps({
                "disease": int(s), "disease_name": str(names[int(s)]),
                "gold_drugs": sorted(test_targets[int(s)]),
                "candidates": [
                    {"drug": int(drug_nodes[j]), "drug_name": str(names[int(drug_nodes[j])]),
                     "kge_score": float(sc[j]), "kge_rank": r + 1}
                    for r, j in enumerate(top)
                ],
            }) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="RotatE")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--negatives", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=3, help="evaluations without valid-MRR gain")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    ap.add_argument("--max-triples", type=int, default=None, help="subsample train (smoke tests only)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    run = ROOT / "results" / "runs" / "kge" / f"{args.model}_seed{args.seed}{args.tag}"
    run.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    data = load_split(args.split)
    tf = data.train
    if args.max_triples:
        g = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(tf.num_triples, generator=g)[: args.max_triples]
        tf = TriplesFactory(mapped_triples=tf.mapped_triples[idx], entity_to_id=data.entity_to_id,
                            relation_to_id=data.relation_to_id)
    print(f"Loaded split in {time.time() - t0:.0f}s: {tf.num_triples:,} train triples, "
          f"{tf.num_entities:,} entities, device={device}")

    model = model_resolver.make(args.model, triples_factory=tf, embedding_dim=args.dim,
                                random_seed=args.seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loop = SLCWATrainingLoop(model=model, triples_factory=tf, optimizer=optimizer,
                             negative_sampler_kwargs={"num_negs_per_pos": args.negatives})

    config = {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "device": str(device), "n_train_triples": tf.num_triples,
              "split_meta": json.loads((args.split / "split_meta.json").read_text()),
              "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "torch": torch.__version__}
    (run / "config.json").write_text(json.dumps(config, indent=2))

    history, best, bad, epoch = [], -1.0, 0, 0
    while epoch < args.epochs:
        n = min(args.eval_every, args.epochs - epoch)
        te = time.time()
        losses = loop.train(triples_factory=tf, num_epochs=epoch + n, batch_size=args.batch_size,
                            use_tqdm=False, continue_training=epoch > 0)
        epoch += n
        v = evaluate(model, data, data.valid)
        history.append({"epoch": epoch, "loss": float(losses[-1]), "valid": v,
                        "sec": round(time.time() - te, 1)})
        print(f"epoch {epoch:3d} loss={losses[-1]:.4f} valid MRR={v['MRR']:.4f} "
              f"H@10={v['Hits@10']:.4f} ({time.time() - te:.0f}s)", flush=True)
        (run / "history.json").write_text(json.dumps(history, indent=2))
        if v["MRR"] > best:
            best, bad = v["MRR"], 0
            torch.save(model, run / "model.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping.")
                break

    model = torch.load(run / "model.pt", weights_only=False).to(device)
    best_epoch = max(history, key=lambda h: h["valid"]["MRR"])["epoch"]
    metrics = {"model": args.model, "best_epoch": best_epoch,
               "valid": evaluate(model, data, data.valid), "test": evaluate(model, data, data.test),
               "train_minutes": round((time.time() - t0) / 60, 1)}
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2))
    export_topk(model, data, data.test, args.topk, run / "test_topk.jsonl")
    print(json.dumps(metrics, indent=2))
    print(f"Saved run to {run}")


if __name__ == "__main__":
    main()
