"""Post-hoc recalibration of LLM confidence, fitted on the validation split.

Maps are fitted on (confidence, is_gold) pairs from validation diseases and
applied to test pairs. All maps are monotone, so they leave every per-disease
ranking (and therefore MRR / Hits@k) unchanged, except that isotonic
regression can create ties.

  temperature  p' = sigmoid(logit(p) / T)            1 parameter
  platt        p' = sigmoid(a * logit(p) + b)        2 parameters (also fixes base-rate shift)
  isotonic     non-parametric monotone step function

Note: plain temperature scaling cannot move a confidence of 0.5, so it only
partly fixes a base-rate mismatch (here ~3.5% positives among top-20
candidates); Platt scaling includes a bias term for that.
"""
from typing import Callable, Dict

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-4


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def _nll(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperature(c: np.ndarray, y: np.ndarray) -> Callable:
    z = _logit(c)
    T = minimize_scalar(lambda t: _nll(_sigmoid(z / t), y), bounds=(0.05, 20), method="bounded").x
    f = lambda p: _sigmoid(_logit(p) / T)
    f.params = {"T": float(T)}
    return f


def fit_platt(c: np.ndarray, y: np.ndarray) -> Callable:
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(c)[:, None], y)
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    f = lambda p: _sigmoid(a * _logit(p) + b)
    f.params = {"a": a, "b": b}
    return f


def fit_isotonic(c: np.ndarray, y: np.ndarray) -> Callable:
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(c, y)
    f = lambda p: iso.predict(p)
    f.params = {"n_steps": int(len(iso.X_thresholds_))}
    return f


FITTERS: Dict[str, Callable] = {"temperature": fit_temperature, "platt": fit_platt, "isotonic": fit_isotonic}
