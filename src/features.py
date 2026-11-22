"""Strictly causal, per-unit feature engineering.

Every quantity at cycle t uses only cycles <= t of the SAME unit (trailing windows).
Nothing here is fitted, so it can be computed once on any trajectory; `check_causality`
verifies that truncating a trajectory never changes the features of the remaining cycles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _window_sums(v: np.ndarray, w: int):
    """For each index i, sum of v over the trailing window [max(0, i-w+1), i] and its length."""
    c = np.concatenate([[0.0], np.cumsum(v)])
    i = np.arange(len(v))
    s = np.maximum(0, i - w + 1)
    return c[i + 1] - c[s], (i - s + 1).astype(float)


def rolling_ols(y: np.ndarray, x: np.ndarray, w: int, min_n: int = 3) -> dict[str, np.ndarray]:
    """Trailing OLS y ~ a + b x over the last w points (fewer at the start of a trajectory).

    Returns slope b, its standard error, the fitted value at the current x and its standard
    error, and the number of points used. NaN when fewer than min_n points.
    """
    x0 = x - x[0]                        # shift for numerical stability (constant, causal)
    sx, n = _window_sums(x0, w)
    sy, _ = _window_sums(y, w)
    sxx, _ = _window_sums(x0 * x0, w)
    sxy, _ = _window_sums(x0 * y, w)
    syy, _ = _window_sums(y * y, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        sxx_c = sxx - sx * sx / n
        b = (sxy - sx * sy / n) / sxx_c
        a = (sy - b * sx) / n
        sse = syy - 2 * a * sy - 2 * b * sxy + n * a * a + 2 * a * b * sx + b * b * sxx
        s2 = np.maximum(sse, 0) / (n - 2)
        se_b = np.sqrt(s2 / sxx_c)
        xbar = sx / n
        fit = a + b * x0
        se_fit = np.sqrt(s2 * (1 / n + (x0 - xbar) ** 2 / sxx_c))
    bad = n < min_n
    out = {"slope": b, "se_slope": se_b, "fit": fit, "se_fit": se_fit, "n": n}
    for k in ("slope", "se_slope", "fit", "se_fit"):
        out[k] = np.where(bad, np.nan, out[k])
    return out


def _unit_features(u: pd.DataFrame, sensors: list[str], windows: list[int], ewm_span: int) -> dict:
    x = u["cycle"].to_numpy(float)
    feats: dict[str, np.ndarray] = {}
    for s in sensors:
        y = u[s].to_numpy(float)
        yc = y - y[0]                    # causal centring for numerical stability
        feats[s] = y
        for w in windows:
            sm, n = _window_sums(yc, w)
            sq, _ = _window_sums(yc * yc, w)
            mean = sm / n
            with np.errstate(invalid="ignore"):
                var = (sq - n * mean * mean) / (n - 1)
            feats[f"{s}_mean{w}"] = mean + y[0]
            feats[f"{s}_std{w}"] = np.where(n > 1, np.sqrt(np.maximum(var, 0)), np.nan)
            feats[f"{s}_slope{w}"] = rolling_ols(y, x, w, min_n=3)["slope"]
        # adjust=False is the recursive (causal) filter: e_t = a*y_t + (1-a)*e_{t-1}
        feats[f"{s}_ewm{ewm_span}"] = pd.Series(y).ewm(span=ewm_span, adjust=False).mean().to_numpy()
    return feats


def build_features(df: pd.DataFrame, sensors: list[str], windows=(5, 10, 20, 30),
                   ewm_span: int = 10) -> pd.DataFrame:
    """Return df[unit, cycle] + causal features for `sensors`, same row order as df."""
    df = df.sort_values(["unit", "cycle"])
    parts = []
    for _, u in df.groupby("unit", sort=True):
        f = pd.DataFrame(_unit_features(u, sensors, list(windows), ewm_span), index=u.index)
        parts.append(f)
    feats = pd.concat(parts)
    out = pd.concat([df[["unit", "cycle"]], feats], axis=1)
    return out


def feature_groups(sensors: list[str], windows=(5, 10, 20, 30), ewm_span: int = 10) -> dict[str, list[str]]:
    raw = list(sensors)
    means = [f"{s}_mean{w}" for s in sensors for w in windows] + [f"{s}_ewm{ewm_span}" for s in sensors]
    other = [f"{s}_{k}{w}" for s in sensors for w in windows for k in ("std", "slope")]
    return {"raw": raw + ["cycle"], "raw+mean": raw + means + ["cycle"],
            "full": raw + means + other + ["cycle"]}


def check_causality(df: pd.DataFrame, sensors: list[str], n_cuts: int = 3, seed: int = 0) -> float:
    """Max abs difference between features computed on full vs truncated trajectories."""
    rng = np.random.default_rng(seed)
    full = build_features(df, sensors).set_index(["unit", "cycle"])
    worst = 0.0
    for unit, u in df.groupby("unit"):
        L = int(u["cycle"].max())
        for cut in rng.integers(2, L, size=n_cuts):
            tr = build_features(u[u["cycle"] <= cut], sensors).set_index(["unit", "cycle"])
            ref = full.loc[tr.index, tr.columns]
            d = np.nanmax(np.abs(tr.to_numpy() - ref.to_numpy()))
            nan_mismatch = (tr.isna().to_numpy() != ref.isna().to_numpy()).any()
            if nan_mismatch:
                return np.inf
            worst = max(worst, float(d))
    return worst
