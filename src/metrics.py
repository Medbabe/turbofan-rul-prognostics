"""Evaluation metrics. d = predicted - true (d > 0 = late prediction = dangerous)."""
from __future__ import annotations

import numpy as np


def rmse(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((p - y) ** 2)))


def cmapss_score_terms(y, p) -> np.ndarray:
    d = np.asarray(p, float) - np.asarray(y, float)
    return np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)


def cmapss_score(y, p, reduce: str = "sum") -> float:
    t = cmapss_score_terms(y, p)
    return float(t.sum() if reduce == "sum" else t.mean())


def coverage(y, lo, hi) -> float:
    y = np.asarray(y, float)
    return float(np.mean((y >= lo) & (y <= hi)))


def width(lo, hi) -> float:
    return float(np.mean(np.asarray(hi, float) - np.asarray(lo, float)))


def alpha_lambda_hits(y, p, alpha: float = 0.2) -> np.ndarray:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return np.abs(p - y) <= alpha * y
