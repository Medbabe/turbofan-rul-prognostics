"""Phase 2 - Leak-free cross-validation of B0, B1 and M1 (+ ablation) on training engines.

* 5-fold GroupKFold by unit (shuffled, seed 42): an engine is never in both train and validation.
* Every model is fitted inside the fold on training engines only (selection, scaler, HI,
  threshold, GBM, conformal calibration).
* Validation engines are scored at 5 fixed random cut points per engine (uniform over [31, L]),
  identical for every model -> paired comparison. Predictions for all cycles are kept for the
  alpha-lambda diagram and the decision simulation (Phase 3).
* The NASA test set is NOT touched here.

Outputs: reports/cv_results.md, reports/cv_predictions.csv.gz, reports/figures/1x_*.png
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import ROOT, add_train_rul, load_train  # noqa: E402
from src.features import check_causality  # noqa: E402
from src.metrics import cmapss_score, coverage, rmse, width  # noqa: E402
from src.models import GBMQuantile, HIBaseline, PopulationPrior, non_flat_sensors, select_sensors  # noqa: E402

SEED = 42
N_FOLDS = 5
N_CUTS = 5
MIN_HIST = 31          # shortest observed test history
ZONE = 150             # decision zone: true RUL <= 150 (fixed before results, see HYPOTHESIS.md)
REPORT = ROOT / "reports"
FIG = REPORT / "figures"


def configs() -> list[tuple[str, str, callable]]:
    """(group, key, factory). Factories so every fold gets a fresh model."""
    c = [("baseline", "B0", lambda: PopulationPrior())]
    for W in (20, 30, 50):
        for R in (100, 125, 150):
            c.append(("B1 grid", f"B1_W{W}_R{R}", lambda W=W, R=R: HIBaseline(W=W, r_max=R)))
    c.append(("B1 intervals", "B1_W20_R100_conformal",
              lambda: HIBaseline(W=20, r_max=100, intervals="conformal")))
    c.append(("B1 intervals", "B1_W20_R125_conformal",
              lambda: HIBaseline(W=20, r_max=125, intervals="conformal")))
    for cap in (None, 100, 125, 150):
        c.append(("M1 cap", f"M1_cap{cap}", lambda cap=cap: GBMQuantile(cap=cap, intervals="quantile")))
    c.append(("M1 ablation", "M1_cap125_allsensors",
              lambda: GBMQuantile(cap=125, sensors="all", intervals=None)))
    c.append(("M1 ablation", "M1_cap125_raw", lambda: GBMQuantile(cap=125, features="raw", intervals=None)))
    c.append(("M1 ablation", "M1_cap125_rawmean",
              lambda: GBMQuantile(cap=125, features="raw+mean", intervals=None)))
    c.append(("M1 intervals", "M1_cap125_cqr", lambda: GBMQuantile(cap=125, intervals="cqr")))
    # configurations to be frozen for the test set (best point config of each family + calibrated intervals)
    c.append(("frozen", "B1_W30_R100_conformal", lambda: HIBaseline(W=30, r_max=100, intervals="conformal")))
    c.append(("frozen", "M1_cap125_allsensors_cqr",
              lambda: GBMQuantile(cap=125, sensors="all", intervals="cqr")))
    return c


def cut_points(train: pd.DataFrame) -> pd.Series:
    rng = np.random.default_rng(SEED)
    rows = []
    for unit, u in train.groupby("unit", sort=True):
        L = int(u["cycle"].max())
        for c in rng.integers(MIN_HIST, L + 1, size=N_CUTS):
            rows.append((unit, int(c)))
    return pd.MultiIndex.from_tuples(rows, names=["unit", "cycle"])


def rmax_of(key: str) -> float:
    m = re.search(r"_R(\d+)", key) or re.search(r"cap(\d+)", key)
    return float(m.group(1)) if m else np.inf


def fold_metrics(ev: pd.DataFrame, has_int: bool, rmax: float = np.inf) -> dict:
    out = {}
    z = ev[ev["RUL"] <= ZONE]
    c = ev[ev["RUL"] <= rmax]
    # coverage where the capped interval CAN contain the truth, and validity of the lower bound
    out["cov_reach"] = coverage(c["RUL"], c["q05"], c["q95"]) if has_int else np.nan
    out["P(RUL>=q05)_zone"] = float((z["RUL"] >= z["q05"]).mean()) if has_int else np.nan
    for tag, m in (("all", np.ones(len(ev), bool)), ("zone", ev["RUL"].to_numpy() <= ZONE)):
        e = ev[m]
        out[f"RMSE_{tag}"] = rmse(e["RUL"], e["q50"])
        out[f"score_{tag}"] = cmapss_score(e["RUL"], e["q50"], reduce="mean")
        out[f"cov_{tag}"] = coverage(e["RUL"], e["q05"], e["q95"]) if has_int else np.nan
        out[f"width_{tag}"] = width(e["q05"], e["q95"]) if has_int else np.nan
    return out


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines)


def main() -> None:
    t0 = time.time()
    FIG.mkdir(parents=True, exist_ok=True)
    log: list[str] = []
    train = add_train_rul(load_train())
    log.append(f"- Loaded train_FD001: {len(train)} rows, {train['unit'].nunique()} units.")

    sens = non_flat_sensors(train)
    caus = check_causality(train[train["unit"] <= 20], sens)
    log.append(f"- Causality check (features on truncated vs full trajectories, 20 units x 3 cuts): "
               f"max abs difference = {caus:g} -> {'OK' if caus == 0 else 'FAILED'}.")
    if caus != 0:
        raise SystemExit("Causality check failed")

    cuts = cut_points(train)
    ev_mask = pd.MultiIndex.from_frame(train[["unit", "cycle"]]).isin(cuts)
    log.append(f"- Evaluation points: {N_CUTS} cut points per unit, uniform over [{MIN_HIST}, L], seed {SEED} "
               f"-> {ev_mask.sum()} distinct points ({(train.loc[ev_mask, 'RUL'] <= ZONE).sum()} in the "
               f"decision zone RUL <= {ZONE}).")

    gkf = GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cfgs = configs()
    preds, metr = [], []
    log.append("")
    log.append("| fold | train units | val units | train rows | val rows | selected sensors | HI threshold (median) |")
    log.append("|---|---|---|---|---|---|---|")
    for k, (tr_idx, va_idx) in enumerate(gkf.split(train, groups=train["unit"])):
        tr, va = train.iloc[tr_idx], train.iloc[va_idx]
        assert not set(tr["unit"]) & set(va["unit"]), "unit leakage"
        sel = select_sensors(tr)
        hi_probe = HIBaseline(W=30).fit(tr, sensors=sel)
        log.append(f"| {k} | {tr['unit'].nunique()} | {va['unit'].nunique()} | {len(tr)} | {len(va)} | "
                   f"{', '.join(sel)} | {hi_probe.thr_med:.3f} |")
        va_ev = ev_mask[va_idx]
        for group, key, make in cfgs:
            ts = time.time()
            model = make().fit(tr.drop(columns="RUL"))
            p = model.predict(va.drop(columns="RUL"))
            p = p.assign(unit=va["unit"].values, cycle=va["cycle"].values, RUL=va["RUL"].values,
                         fold=k, config=key, eval=va_ev)
            preds.append(p)
            has_int = not (p["q05"] == p["q95"]).all()
            m = fold_metrics(p[p["eval"]], has_int, rmax_of(key))
            m.update(fold=k, config=key, group=group)
            metr.append(m)
            print(f"fold {k} {key:24s} RMSE_zone={m['RMSE_zone']:6.2f} cov_zone={m['cov_zone']:.2f} "
                  f"({time.time() - ts:.1f}s)", flush=True)

    P = pd.concat(preds, ignore_index=True)
    P.to_csv(REPORT / "cv_predictions.csv.gz", index=False, float_format="%.3f")
    M = pd.DataFrame(metr)
    M.to_csv(REPORT / "cv_fold_metrics.csv", index=False)

    # ---- aggregate: mean +- std over folds
    num = [c for c in M.columns if c not in ("fold", "config", "group")]
    agg_m, agg_s = M.groupby("config")[num].mean(), M.groupby("config")[num].std()
    order = [k for _, k, _ in cfgs]

    def table(keys, cols):
        rows = []
        for key in keys:
            r = {"config": key}
            for c in cols:
                mu, sd = agg_m.loc[key, c], agg_s.loc[key, c]
                if np.isnan(mu):
                    r[c] = "-"
                elif c.startswith("cov") or c.startswith("P("):
                    r[c] = f"{mu:.0%} ± {sd:.0%}"
                else:
                    r[c] = f"{mu:.1f} ± {sd:.1f}" if not c.startswith("score") else f"{mu:.2f} ± {sd:.2f}"
            rows.append(r)
        return md(pd.DataFrame(rows))

    point_cols = ["RMSE_zone", "score_zone", "RMSE_all", "score_all"]
    int_cols = ["cov_zone", "cov_reach", "P(RUL>=q05)_zone", "width_zone", "cov_all"]
    b1_keys = [k for k in order if k.startswith("B1") and not k.endswith("conformal")]
    m1_keys = [k for k in order if k.startswith("M1") and not k.endswith("cqr")]
    best_b1 = agg_m.loc[b1_keys, "RMSE_zone"].idxmin()
    best_m1 = agg_m.loc[m1_keys, "RMSE_zone"].idxmin()

    # paired comparison on identical evaluation points
    def paired(a, b):
        d = M.set_index(["config", "fold"])["RMSE_zone"]
        diff = d.loc[a].values - d.loc[b].values
        return diff.mean(), diff.std(ddof=1), int((diff > 0).sum())

    out = ["# Phase 2 - Cross-validation results (training engines only)", "",
           "_Generated by `python -m src.cv`. 5-fold GroupKFold by engine, everything fitted inside the "
           "fold. Values: mean ± std over the 5 folds. Zone = evaluation points with true RUL <= "
           f"{ZONE} (model selection metric: RMSE_zone). Score = mean C-MAPSS score per point. The NASA "
           "test set is not used here._", "",
           "## 1. Preparation log", ""] + log + ["",
           "## 2. Baselines", "",
           table(["B0"] + b1_keys, point_cols + int_cols), "",
           f"Best B1 (RMSE_zone): **{best_b1}**.", "",
           "### B1 intervals: Monte Carlo (slope + level + threshold) vs split conformal", "",
           table(["B1_W20_R100", "B1_W20_R100_conformal", "B1_W20_R125", "B1_W20_R125_conformal",
                  "B1_W30_R100", "B1_W30_R100_conformal"], point_cols + int_cols), "",
           "_cov_reach = coverage on points with true RUL <= R_max of the model (where a capped interval "
           "can contain the truth). P(RUL>=q05)_zone = validity of the lower bound used by the decision "
           "rule (target 95 %)._", "",
           "## 3. Gradient boosting - target cap", "",
           table([k for k in m1_keys if k.startswith("M1_cap") and "_" not in k[4:]], point_cols + int_cols), "",
           "## 4. Gradient boosting - ablation (one change at a time from cap=125, selected sensors, full features)",
           "", table(["M1_cap125", "M1_cap125_allsensors", "M1_cap125_rawmean", "M1_cap125_raw"], point_cols), "",
           "## 5. Intervals - raw quantile regression vs split conformal (CQR)", "",
           table(["M1_cap125", "M1_cap125_cqr"], point_cols + int_cols), "",
           f"Best M1 (RMSE_zone): **{best_m1}**.", "",
           "## 5b. Configurations frozen for the single test evaluation", "",
           "Point configuration = best RMSE_zone of each family; intervals = calibrated version "
           "(raw intervals under-cover, see above).", "",
           table(["B0", "B1_W30_R100_conformal", "M1_cap125_allsensors_cqr"], point_cols + int_cols), "",
           "## 6. Paired comparisons (difference of RMSE_zone per fold, same evaluation points)", ""]
    rows = []
    for a, b in (("B0", best_b1), (best_b1, best_m1), ("M1_capNone", "M1_cap125")):
        mu, sd, n = paired(a, b)
        rows.append({"A": a, "B": b, "RMSE_zone(A) - RMSE_zone(B)": f"{mu:.1f} ± {sd:.1f}",
                     "folds where B is better": f"{n}/{N_FOLDS}"})
    out += [md(pd.DataFrame(rows)), ""]

    # ---- error by true-RUL bin (H5)
    bins = [0, 25, 50, 75, 100, 125, 150, 200, 400]
    E = P[P["eval"] & P["config"].isin(["B0", best_b1, best_m1])].copy()
    E["bin"] = pd.cut(E["RUL"], bins, right=False)
    eb = E.groupby(["config", "bin"], observed=True).apply(
        lambda g: pd.Series({"RMSE": rmse(g["RUL"], g["q50"]), "bias": (g["q50"] - g["RUL"]).mean(),
                             "n": len(g)}), include_groups=False).reset_index()
    pv = eb.pivot(index="bin", columns="config", values="RMSE").round(1)
    pb = eb.pivot(index="bin", columns="config", values="bias").round(1)
    pv.insert(0, "n points", eb.groupby("bin", observed=True)["n"].first().astype(int))
    pb.columns = [f"bias {c}" for c in pb.columns]
    out += ["## 7. Error by true-RUL range (pooled over folds; bias = mean(pred - true), > 0 = late)", "",
            md(pd.concat([pv, pb], axis=1).reset_index().astype(str)), "",
            f"Share of {best_b1} point predictions sitting at its fallback value R_max (no significant "
            "HI slope yet): " + ", ".join(
                f"{b}: {s:.0%}" for b, s in E[E["config"] == best_b1].groupby("bin", observed=True).apply(
                    lambda g: float((g["q50"] >= rmax_of(best_b1) - 1e-6).mean()), include_groups=False).items()) + ".",
            "",
            "Figures: `reports/figures/10_rmse_by_rul.png`, `11_example_units.png`.", "",
            f"_Runtime: {time.time() - t0:.0f} s._"]
    (REPORT / "cv_results.md").write_text("\n".join(out) + "\n", encoding="utf-8")

    # ---- figures
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for key in ["B0", best_b1, best_m1]:
        s = pv[key]
        ax.plot([b.left + (b.right - b.left) / 2 for b in s.index], s.values, marker="o", label=key)
    ax.axvline(ZONE, color="gray", ls=":", lw=1)
    ax.set_xlabel("true RUL (cycles)"); ax.set_ylabel("RMSE (cycles)")
    ax.set_title("CV error by true RUL (decision zone left of dotted line)"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "10_rmse_by_rul.png", dpi=150); plt.close(fig)

    units = sorted(P["unit"].unique())[::25][:4]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), sharey=True)
    for ax, u in zip(axes, units):
        for key, col in ((best_b1, "tab:orange"), (best_m1, "tab:green")):
            d = P[(P["unit"] == u) & (P["config"] == key)].sort_values("cycle")
            ax.fill_between(d["cycle"], d["q05"], d["q95"], color=col, alpha=.18)
            ax.plot(d["cycle"], d["q50"], color=col, lw=1.2, label=key)
        ax.plot(d["cycle"], d["RUL"], "k--", lw=1, label="true RUL")
        ax.set_title(f"unit {u} (validation)", fontsize=9); ax.set_xlabel("cycle")
    axes[0].set_ylabel("RUL"); axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "11_example_units.png", dpi=150); plt.close(fig)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
