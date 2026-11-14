"""Ranking metrics for drug recommendation.

Protocol (used for every method in the paper):
  For each held-out indication (drug d, disease s), rank ALL drugs for
  disease s by score. Other known indications of s (from train and valid,
  and the other test pairs) are filtered out, so only the target d competes
  against drugs that are not known treatments. Rank is 1-based; ties use the
  mean ("realistic") rank.
"""
from typing import Dict, Iterable

import numpy as np


def realistic_rank(scores: np.ndarray, target: int, mask: np.ndarray) -> float:
    """Rank of `target` among candidates where `mask` is True (target included)."""
    s = scores[target]
    cand = scores[mask]
    higher = np.sum(cand > s)
    equal = np.sum(cand == s)  # includes target itself
    return float(higher + (equal + 1) / 2.0)


def ranking_metrics(ranks: Iterable[float], ks=(1, 3, 10)) -> Dict[str, float]:
    r = np.asarray(list(ranks), dtype=float)
    out = {"n": int(len(r)), "MRR": float(np.mean(1.0 / r))}
    for k in ks:
        out[f"Hits@{k}"] = float(np.mean(r <= k))
    out["mean_rank"] = float(np.mean(r))
    out["median_rank"] = float(np.median(r))
    return out
