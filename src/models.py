"""B0 (population prior), B1 (health indicator + linear degradation), M1 (gradient boosting).

Every model has fit(train_df) on run-to-failure training units and
predict(df) -> DataFrame[q05, q50, q95] for ALL rows of df (features are causal, so the row
for cycle t only depends on cycles <= t). Everything that is fitted is fitted in fit().
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.data import SENSORS
from src.features import build_features, feature_groups, rolling_ols

SEED = 42


# --------------------------------------------------------------------------- selection
def select_sensors(train: pd.DataFrame, min_rho: float = 0.5, min_share: float = 0.9) -> list[str]:
    """Monotone-trend sensors, decided on TRAINING units only (same rule as the diagnostic)."""
    keep = []
    for s in SENSORS:
        if train[s].nunique() <= 2 or train[s].std() < 1e-3:
            continue
        rhos = np.array([spearmanr(u["cycle"], u[s])[0] for _, u in train.groupby("unit")
                         if u[s].nunique() > 1])
        if len(rhos) == 0:
            continue
        med = np.median(rhos)
        if abs(med) >= min_rho and np.mean(np.sign(rhos) == np.sign(med)) >= min_share:
            keep.append(s)
    return keep


def non_flat_sensors(train: pd.DataFrame) -> list[str]:
    return [s for s in SENSORS if train[s].nunique() > 2 and train[s].std() >= 1e-3]


def _lifetimes(train: pd.DataFrame) -> np.ndarray:
    return train.groupby("unit")["cycle"].max().to_numpy()


# --------------------------------------------------------------------------- B0
class PopulationPrior:
    """RUL(t) = distribution of (L - t) over training units with lifetime L > t. No sensors."""
    name = "B0 prior"

    def fit(self, train: pd.DataFrame):
        self.L = np.sort(_lifetimes(train))
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        t = df["cycle"].to_numpy()
        out = np.zeros((len(t), 3))
        for tv in np.unique(t):
            r = self.L[self.L > tv] - tv
            q = np.quantile(r, [.05, .5, .95]) if len(r) else np.zeros(3)
            out[t == tv] = q
        return pd.DataFrame(out, columns=["q05", "q50", "q95"], index=df.index)


# --------------------------------------------------------------------------- B1
class HIBaseline:
    """Health indicator (linear fusion of selected sensors) + trailing linear degradation."""

    def __init__(self, W: int = 30, r_max: float = 125, n_mc: int = 500, edge: int = 30,
                 min_n: int = 5, z_signif: float = 1.645, intervals: str = "mc",
                 calib_frac: float = 0.2, n_bins: int = 4, seed: int = SEED):
        self.W, self.r_max, self.n_mc, self.edge = W, r_max, n_mc, edge
        self.min_n, self.z = min_n, z_signif
        self.intervals, self.calib_frac, self.n_bins, self.seed = intervals, calib_frac, n_bins, seed
        self.rng = np.random.default_rng(seed)
        self.name = f"B1 HI (W={W}, Rmax={r_max}, int={intervals})"

    def _hi(self, df: pd.DataFrame) -> np.ndarray:
        return self.reg.predict(self.scaler.transform(df[self.sensors].to_numpy()))

    def _trend(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["unit", "cycle"])
        hi = pd.Series(self._hi(df), index=df.index)
        rows = []
        for _, u in df.groupby("unit", sort=True):
            r = rolling_ols(hi.loc[u.index].to_numpy(), u["cycle"].to_numpy(float), self.W, self.min_n)
            rows.append(pd.DataFrame(r, index=u.index))
        out = pd.concat(rows)
        out["hi"] = hi
        return out

    def fit(self, train: pd.DataFrame, sensors: list[str] | None = None):
        if self.intervals == "conformal":
            # split conformal by unit: fit on 80% of training units, measure errors on the rest
            rng = np.random.default_rng(self.seed)
            units = train["unit"].unique()
            cal_u = rng.choice(units, size=max(1, int(round(self.calib_frac * len(units)))), replace=False)
            cal = train[train["unit"].isin(cal_u)]
            self._fit_core(train[~train["unit"].isin(cal_u)], sensors)
            point = self._point(cal)
            y = np.minimum(cal.groupby("unit")["cycle"].transform("max") - cal["cycle"], self.r_max).to_numpy()
            e = y - point                                    # >0: prediction too early (safe side)
            edges = np.unique(np.quantile(point, np.linspace(0, 1, self.n_bins + 1)))
            self.bin_edges = edges
            b = np.clip(np.searchsorted(edges, point, side="right") - 1, 0, len(edges) - 2)
            self.bin_q = []
            for k in range(len(edges) - 1):
                ek = np.sort(e[b == k]) if (b == k).any() else np.sort(e)
                n = len(ek)
                lo = ek[max(0, int(np.floor((n + 1) * 0.05)) - 1)]
                hi = ek[min(n - 1, int(np.ceil((n + 1) * 0.95)) - 1)]
                self.bin_q.append((lo, hi))
            return self
        return self._fit_core(train, sensors)

    def _point(self, df: pd.DataFrame) -> np.ndarray:
        tt = self._trend(df).loc[df.index]
        fit, b, se_b = tt["fit"].to_numpy(), tt["slope"].to_numpy(), tt["se_slope"].to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            point = np.where((b / se_b) < -self.z, (fit - self.thr_med) / (-b), self.r_max)
        return np.clip(np.nan_to_num(point, nan=self.r_max), 0, self.r_max)

    def _fit_core(self, train: pd.DataFrame, sensors: list[str] | None):
        self.sensors = sensors if sensors is not None else select_sensors(train)
        tr = train.copy()
        tr["RUL"] = tr.groupby("unit")["cycle"].transform("max") - tr["cycle"]
        self.scaler = StandardScaler().fit(tr[self.sensors].to_numpy())
        healthy = tr["cycle"] <= self.edge
        failed = tr["RUL"] < self.edge
        m = healthy | failed
        X = self.scaler.transform(tr.loc[m, self.sensors].to_numpy())
        self.reg = LinearRegression().fit(X, healthy[m].astype(float).to_numpy())
        tt = self._trend(tr)
        last = tr.groupby("unit")["cycle"].idxmax()
        self.thresholds = tt.loc[last, "fit"].to_numpy()
        self.thr_med = float(np.median(self.thresholds))
        return self

    def predict(self, df: pd.DataFrame, return_trend: bool = False):
        if self.intervals == "conformal":
            point = self._point(df)
            e = self.bin_edges
            b = np.clip(np.searchsorted(e, point, side="right") - 1, 0, len(e) - 2)
            q = np.array(self.bin_q)[b]
            out = pd.DataFrame({"q05": np.clip(point + q[:, 0], 0, self.r_max), "q50": point,
                                "q95": np.clip(point + q[:, 1], 0, self.r_max)}, index=df.index)
            return out
        tt = self._trend(df).loc[df.index]
        fit, se_fit = tt["fit"].to_numpy(), tt["se_fit"].to_numpy()
        b, se_b = tt["slope"].to_numpy(), tt["se_slope"].to_numpy()
        R = self.r_max
        # point: extrapolation to the median threshold; R_max if no significant degradation
        with np.errstate(invalid="ignore", divide="ignore"):
            signif = (b / se_b) < -self.z
            point = np.where(signif, (fit - self.thr_med) / (-b), R)
        point = np.clip(np.nan_to_num(point, nan=R), 0, R)
        # interval: Monte Carlo over slope, current level and threshold
        n = len(df)
        q = np.full((n, 3), R, dtype=float)
        ok = np.isfinite(b) & np.isfinite(se_b) & np.isfinite(se_fit)
        idx = np.flatnonzero(ok)
        for chunk in np.array_split(idx, max(1, len(idx) // 2000)):
            k = len(chunk)
            bs = b[chunk, None] + se_b[chunk, None] * self.rng.standard_normal((k, self.n_mc))
            fs = fit[chunk, None] + se_fit[chunk, None] * self.rng.standard_normal((k, self.n_mc))
            ths = self.rng.choice(self.thresholds, size=(k, self.n_mc))
            with np.errstate(invalid="ignore", divide="ignore"):
                rs = np.where(bs < 0, (fs - ths) / (-bs), np.inf)
            rs = np.clip(rs, 0, R)
            q[chunk] = np.quantile(rs, [.05, .5, .95], axis=1).T
        out = pd.DataFrame({"q05": np.minimum(q[:, 0], point), "q50": point,
                            "q95": np.maximum(q[:, 2], point)}, index=df.index)
        return (out, tt) if return_trend else out


# --------------------------------------------------------------------------- M1
LGB_PARAMS = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=50,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED,
                  verbose=-1, n_jobs=-1)


class GBMQuantile:
    """LightGBM direct RUL regression on causal features, quantile objectives.

    cap: None (raw RUL target) or R_max. sensors: "selected" | "all". features: "raw" |
    "raw+mean" | "full". intervals: None (median only) | "quantile" | "cqr" (split conformal,
    calibration on 20% of the training units, against the capped target).
    """

    def __init__(self, cap=125, sensors="selected", features="full", intervals="quantile",
                 calib_frac: float = 0.2, alpha: float = 0.10, seed: int = SEED):
        self.cap, self.sensor_mode, self.feat_mode = cap, sensors, features
        self.intervals, self.calib_frac, self.alpha, self.seed = intervals, calib_frac, alpha, seed
        self.name = f"M1 GBM (cap={cap}, sensors={sensors}, feats={features}, int={intervals})"

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        f = build_features(df, self.sensors).loc[df.index]
        return f[self.cols]

    def _target(self, train: pd.DataFrame) -> np.ndarray:
        rul = (train.groupby("unit")["cycle"].transform("max") - train["cycle"]).to_numpy(float)
        return np.minimum(rul, self.cap) if self.cap is not None else rul

    def _fit_q(self, X, y, a):
        import lightgbm as lgb
        # boost_from_average=False: with a capped target, >50% of rows can sit exactly at the cap.
        # Starting from the median (= cap) gives every row the same quantile gradient, so no split
        # has any gain and the model stays constant. Starting from 0 avoids this degenerate case.
        return lgb.LGBMRegressor(objective="quantile", alpha=a, boost_from_average=False,
                                 **LGB_PARAMS).fit(X, y)

    def fit(self, train: pd.DataFrame):
        self.sensors = select_sensors(train) if self.sensor_mode == "selected" else non_flat_sensors(train)
        self.cols = feature_groups(self.sensors)[self.feat_mode]
        y = self._target(train)
        X = self._X(train)
        self.q_hat = 0.0
        if self.intervals == "cqr":
            rng = np.random.default_rng(self.seed)
            units = train["unit"].unique()
            cal_units = rng.choice(units, size=max(1, int(round(self.calib_frac * len(units)))), replace=False)
            cal = train["unit"].isin(cal_units).to_numpy()
            prop = ~cal
            self.m = {a: self._fit_q(X[prop], y[prop], a) for a in (.05, .5, .95)}
            lo, hi = self.m[.05].predict(X[cal]), self.m[.95].predict(X[cal])
            lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
            e = np.maximum(lo - y[cal], y[cal] - hi)
            n = len(e)
            self.q_hat = float(np.quantile(e, min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)))
        else:
            qs = (.05, .5, .95) if self.intervals == "quantile" else (.5,)
            self.m = {a: self._fit_q(X, y, a) for a in qs}
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._X(df)
        upper = self.cap if self.cap is not None else np.inf
        q50 = np.clip(self.m[.5].predict(X), 0, upper)
        if .05 in self.m:
            lo, hi = self.m[.05].predict(X), self.m[.95].predict(X)
            lo, hi = np.minimum(lo, hi) - self.q_hat, np.maximum(lo, hi) + self.q_hat
            lo, hi = np.clip(np.minimum(lo, q50), 0, upper), np.clip(np.maximum(hi, q50), 0, upper)
        else:
            lo = hi = q50
        return pd.DataFrame({"q05": lo, "q50": q50, "q95": hi}, index=df.index)
