"""Train TxGNN on our split and score every drug for every test (and valid) disease.

Runs in the isolated env (.venv-txgnn: torch 2.2, DGL 2.2, pandas 1.5) with the
upstream code in third_party/TxGNN (commit 0e0203c), unmodified. Two shims:
  * dgl.dataloading.EdgeDataLoader (removed in DGL >= 1.0) is rebuilt from
    DataLoader + as_edge_prediction_sampler (same batches: input nodes, pos
    graph, neg graph, blocks).
  * TxData is bypassed (it downloads its own PrimeKG copy); we build the same
    object from data/processed/txgnn_seed42 (src/baselines/txgnn_prep.py).

Hyper-parameters follow the TxGNN README / paper defaults
(hidden 100, proto_num 3, all_nodes_profile similarity, rarity aggregation,
pretrain 2 epochs lr 1e-3 batch 1024, finetune 500 epochs lr 5e-4).

Output (results/runs/txgnn/<name>/):
  scores_<split>.npz   diseases (node_index), drugs (node_index), score matrix
  model/               TxGNN checkpoint;  config.json
Then, in the main env:  python -m src.baselines.txgnn_export --run results/runs/txgnn/<name>

Usage: .venv-txgnn/bin/python -m src.baselines.txgnn_run [--pretrain-epochs 2 --finetune-epochs 500]
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party" / "TxGNN"))

import dgl  # noqa: E402
import dgl.dataloading as dl  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402


MAX_PRETRAIN_STEPS = None  # timing tests only


def _edge_data_loader(g, eids, sampler, negative_sampler=None, **kw):
    loader = dl.DataLoader(g, eids, dl.as_edge_prediction_sampler(sampler, negative_sampler=negative_sampler), **kw)
    if MAX_PRETRAIN_STEPS is None:
        return loader
    import itertools
    t0 = time.time()

    def capped():
        for i, b in enumerate(itertools.islice(loader, MAX_PRETRAIN_STEPS)):
            yield b
        print(f"pretrain: {MAX_PRETRAIN_STEPS} steps in {time.time() - t0:.1f}s "
              f"({len(loader)} steps per epoch)", flush=True)
    return capped()


dl.EdgeDataLoader = _edge_data_loader

from txgnn import TxGNN  # noqa: E402
from txgnn import utils as txu  # noqa: E402
from txgnn.utils import create_dgl_graph  # noqa: E402


def _neg_keep_empty(self, graph):
    """Upstream drops edge types with no edges from the negative graph; our valid/test
    graphs contain only indications, so keep (empty) contraindication/off-label types."""
    out = {}
    empty = torch.zeros(0, dtype=torch.int64, device=self.device)
    for etype in graph.canonical_etypes:
        if graph.num_edges(etype) == 0:
            out[etype] = (empty, empty)  # sampling from an empty type gives invalid ids
            continue
        out.update(txu.construct_negative_graph_each_etype(graph, self.k, etype, self.method, self.weights, self.device))
    return dgl.heterograph(out, num_nodes_dict={nt: graph.number_of_nodes(nt) for nt in graph.ntypes})


txu.Full_Graph_NegSampler.__call__ = _neg_keep_empty


def _print_dict(x, dd_only=True):  # upstream indexes all 6 drug-disease types; ours has 2 in valid/test
    for i, j in x.items():
        print(f"{i}: {j}")


txu.print_dict = _print_dict
# the module txgnn.TxGNN imported print_dict via `from .utils import *`; note that
# `txgnn.TxGNN` as an attribute is the class (re-exported), so use sys.modules
sys.modules["txgnn.TxGNN"].print_dict = _print_dict


def load(folder: Path):
    df = pd.read_csv(folder / "kg_directed.csv")
    tr, va, te = (pd.read_csv(folder / f"{s}.csv") for s in ("train", "valid", "test"))
    G = create_dgl_graph(tr, df)
    return SimpleNamespace(G=G, df=df, df_train=tr, df_valid=va, df_test=te, data_folder=str(folder),
                           disease_eval_idx=None, split="random", no_kg=False, seed=42)


@torch.no_grad()
def score_split(tx, df_split: pd.DataFrame, n_drugs: int, chunk: int):
    pos = df_split[df_split.relation == "indication"]
    diseases = np.array(sorted(pos.y_idx.astype(int).unique()))
    drugs = np.arange(n_drugs)
    out = np.zeros((len(diseases), n_drugs), dtype=np.float32)
    et = ("drug", "indication", "disease")
    for i in range(0, len(diseases), chunk):
        ds = diseases[i:i + chunk]
        q = pd.DataFrame({"x_idx": np.tile(drugs, len(ds)), "relation": "indication",
                          "y_idx": np.repeat(ds, n_drugs)})
        s = tx.predict(q)[et].reshape(-1).cpu().numpy()
        out[i:i + len(ds)] = s.reshape(len(ds), n_drugs)
        print(f"  scored {i + len(ds)}/{len(diseases)} diseases", flush=True)
    return diseases, drugs, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "txgnn_seed42")
    ap.add_argument("--name", default="TxGNN_seed42")
    ap.add_argument("--pretrain-epochs", type=int, default=2)
    ap.add_argument("--pretrain-batch", type=int, default=1024)
    ap.add_argument("--finetune-epochs", type=int, default=500)
    ap.add_argument("--valid-per-n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--max-pretrain-steps", type=int, default=None, help="timing tests only")
    ap.add_argument("--skip-train", action="store_true", help="load model/ and only score")
    args = ap.parse_args()
    global MAX_PRETRAIN_STEPS
    MAX_PRETRAIN_STEPS = args.max_pretrain_steps
    torch.manual_seed(args.seed); np.random.seed(args.seed); dgl.seed(args.seed)

    out = ROOT / "results" / "runs" / "txgnn" / args.name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = load(args.data)
    print(f"graph built in {time.time() - t0:.0f}s: {data.G}", flush=True)
    tx = TxGNN(data=data, weight_bias_track=False, device="cpu")
    if args.skip_train:
        tx.load_pretrained(str(out / "model"))
    else:
        tx.model_initialize(n_hid=100, n_inp=100, n_out=100, proto=True, proto_num=3, attention=False,
                            sim_measure="all_nodes_profile", agg_measure="rarity", num_walks=200, path_length=2)
        t1 = time.time()
        tx.pretrain(n_epoch=args.pretrain_epochs, learning_rate=1e-3, batch_size=args.pretrain_batch, train_print_per_n=20)
        t2 = time.time()
        tx.finetune(n_epoch=args.finetune_epochs, learning_rate=5e-4, train_print_per_n=5,
                    valid_per_n=args.valid_per_n, save_name=str(out / "finetune_result.pkl"))
        t3 = time.time()
        print(f"pretrain {(t2 - t1) / 60:.1f} min, finetune {(t3 - t2) / 60:.1f} min", flush=True)
        tx.save_model(str(out / "model"))
        (out / "config.json").write_text(json.dumps({
            **{k: str(v) for k, v in vars(args).items()}, "txgnn_commit": "0e0203c",
            "model_config": tx.config, "pretrain_min": (t2 - t1) / 60, "finetune_min": (t3 - t2) / 60}, indent=2))
    tx.model = tx.best_model
    n_drugs = data.G.number_of_nodes("drug")
    nm = pd.read_csv(args.data / "node_map.csv")
    back = {t: dict(zip(g.type_idx, g.node_index)) for t, g in nm.groupby("node_type")}
    for split, df in (("test", data.df_test), ("valid", data.df_valid)):
        d, r, s = score_split(tx, df, n_drugs, args.chunk)
        np.savez_compressed(out / f"scores_{split}.npz", diseases=np.array([back["disease"][i] for i in d]),
                            drugs=np.array([back["drug"][i] for i in r]), scores=s)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
