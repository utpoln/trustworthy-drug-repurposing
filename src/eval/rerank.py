"""Accuracy and calibration of LLM re-ranking.

Methods compared for every test disease:
  kge     graph-only: KGE order
  llm     candidates in the top-k sorted by LLM confidence (ties -> KGE order)
  rrf     reciprocal-rank fusion of the KGE and LLM orders: 1/(60+r_kge) + 1/(60+r_llm)
          (Cormack et al., 2009; parameter-free, so nothing is tuned on test)

Only the top-k candidates are re-ordered. A gold drug outside the top-k keeps
its filtered KGE rank, which is > k. Within the top-k, ranks are filtered:
other gold drugs of the same disease ranked above the target are not counted.

Calibration (over every scored disease-candidate pair; label = is gold):
  ECE (10 equal-width bins), Brier score, AUROC.
"""
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.eval.ranking import ranking_metrics

RRF_K = 60


def load_llm(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            conf = r["parsed"].get("confidence") if r.get("parse_ok") else None
            try:
                conf = float(conf) if conf is not None else np.nan
            except (TypeError, ValueError):
                conf = np.nan
            rows.append({"disease": r["disease"], "drug": r["drug"], "is_gold": r["is_gold"],
                         "kge_rank_raw": r["kge_rank"], "confidence": np.clip(conf, 0, 1),
                         "error": r["llm"]["error"]})
    df = pd.DataFrame(rows)
    # keep the last successful record per pair (reruns append)
    df = df.sort_values("error", na_position="last").drop_duplicates(["disease", "drug"], keep="first")
    return df


def rerank_ranks(llm: pd.DataFrame, test_ranks: pd.DataFrame, k: int) -> Dict[str, List[float]]:
    """Filtered rank of every test pair under each method."""
    out = {"kge": [], "llm": [], "rrf": []}
    by_d = {d: g for d, g in llm.groupby("disease")}
    for d, gold_rows in test_ranks.groupby("disease"):
        golds = set(gold_rows["drug"])
        cand = by_d.get(d)
        for drug, kr in zip(gold_rows["drug"], gold_rows["kge_rank"]):
            out["kge"].append(kr)
            if cand is None:
                out["llm"].append(kr); out["rrf"].append(kr); continue
            c = cand[cand["kge_rank_raw"] <= k].copy()
            if drug not in set(c["drug"]):
                out["llm"].append(kr); out["rrf"].append(kr); continue
            conf = c["confidence"].fillna(0.0).to_numpy()  # unparsed -> 0 (reported separately)
            kge_pos = c["kge_rank_raw"].to_numpy()
            llm_order = np.lexsort((kge_pos, -conf))          # by conf desc, then KGE rank
            llm_pos = np.empty(len(c)); llm_pos[llm_order] = np.arange(1, len(c) + 1)
            rrf = 1 / (RRF_K + kge_pos) + 1 / (RRF_K + llm_pos)
            rrf_order = np.lexsort((kge_pos, -rrf))
            drugs = c["drug"].to_numpy()
            for name, order in (("llm", llm_order), ("rrf", rrf_order)):
                ranked = [x for x in drugs[order] if x == drug or x not in golds]  # filter other golds
                out[name].append(float(ranked.index(drug) + 1))
    return out


def ece(conf: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(conf[m].mean() - y[m].mean())
    return float(total)


def reliability(conf: np.ndarray, y: np.ndarray, bins: int = 10) -> List[dict]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
    return [{"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int((idx == b).sum()),
             "mean_conf": float(conf[idx == b].mean()) if (idx == b).any() else None,
             "frac_gold": float(y[idx == b].mean()) if (idx == b).any() else None}
            for b in range(bins)]


def auroc(score: np.ndarray, y: np.ndarray) -> float:
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def calibration(llm: pd.DataFrame) -> dict:
    ok = llm.dropna(subset=["confidence"])
    c, y = ok["confidence"].to_numpy(float), ok["is_gold"].to_numpy(int)
    return {"n_pairs": int(len(llm)), "parse_rate": float(len(ok) / max(len(llm), 1)),
            "base_rate": float(y.mean()) if len(y) else None,
            "ECE": ece(c, y), "Brier": float(np.mean((c - y) ** 2)), "AUROC": auroc(c, y),
            "mean_conf": float(c.mean()), "reliability": reliability(c, y)}


def evaluate_run(llm_file: Path, test_ranks_file: Path, k: int = 20) -> dict:
    llm = load_llm(llm_file)
    tr = pd.read_csv(test_ranks_file, sep="\t")
    ranks = rerank_ranks(llm, tr, k)
    return {"file": str(llm_file), "k": k,
            "accuracy": {m: ranking_metrics(r) for m, r in ranks.items()},
            "calibration": calibration(llm),
            "api_errors": int(llm["error"].notna().sum())}
