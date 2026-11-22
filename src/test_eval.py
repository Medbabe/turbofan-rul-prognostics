"""Phase 3b - SINGLE evaluation on the NASA test set (test_FD001 + RUL_FD001).

* The three frozen configurations (src/frozen.py) are refitted on all 100 training engines and
  predict the RUL at the LAST observed cycle of each test engine.
* Test-set lock: the first run writes reports/TEST_LOCK with the hash of the frozen configuration.
  Re-running with the same configuration (reproducibility, `make all`) is allowed and gives identical
  numbers; running with a DIFFERENT configuration is refused. The test set therefore can't be used
  to choose between configurations.
* Uncertainty on test metrics: bootstrap over engines (2,000 resamples, seed 42).
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import ROOT, load_test, load_test_rul, load_train  # noqa: E402
from src.frozen import FROZEN, FROZEN_HASH, LEAD_TIME, make  # noqa: E402
from src.metrics import alpha_lambda_hits, cmapss_score, coverage, rmse, width  # noqa: E402

REPORT = ROOT / "reports"
FIG = REPORT / "figures"
LOCK = REPORT / "TEST_LOCK"
SEED = 42
N_BOOT = 2000


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in r.values) + " |" for _, r in df.iterrows()]
    return "\n".join(out)


def check_lock() -> None:
    if LOCK.exists():
        info = json.loads(LOCK.read_text())
        if info["frozen_hash"] != FROZEN_HASH:
            raise SystemExit(
                f"REFUSED: the test set was already used with configuration {info['frozen_hash']} "
                f"({info['first_run']}). Current configuration {FROZEN_HASH} differs. The test set is used once.")
        print(f"Test lock found, same configuration {FROZEN_HASH}: reproducing the original evaluation.")
    else:
        try:
            commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                    cwd=ROOT).stdout.strip()
        except OSError:
            commit = "unknown"
        LOCK.write_text(json.dumps({"frozen_hash": FROZEN_HASH, "frozen": FROZEN, "lead_time": LEAD_TIME,
                                    "first_run": dt.datetime.now().isoformat(timespec="seconds"),
                                    "git_commit": commit}, indent=2) + "\n")
        print(f"First use of the test set: lock written ({FROZEN_HASH}).")


def boot_ci(y, preds: dict[str, np.ndarray]) -> dict:
    rng = np.random.default_rng(SEED)
    n = len(y)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    res = {}
    for m, p in preds.items():
        r = np.sqrt(((p[idx] - y[idx]) ** 2).mean(axis=1))
        s = np.array([cmapss_score(y[i], p[i]) for i in idx])
        res[m] = {"rmse": np.percentile(r, [2.5, 97.5]), "score": np.percentile(s, [2.5, 97.5]),
                  "rmse_samples": r}
    return res


def main() -> None:
    check_lock()
    FIG.mkdir(parents=True, exist_ok=True)
    train = load_train()
    test = load_test()
    y = load_test_rul().to_numpy(float)
    last = test.groupby("unit")["cycle"].idxmax()

    preds = {}
    for name in ("B0", "B1", "M1"):
        model = make(name).fit(train)
        p = model.predict(test).loc[last]
        preds[name] = p.reset_index(drop=True)
        print(f"{name}: fitted and predicted", flush=True)

    P = pd.concat({k: v for k, v in preds.items()}, axis=1)
    P.columns = [f"{a}_{b}" for a, b in P.columns]
    P.insert(0, "unit", np.arange(1, len(y) + 1))
    P.insert(1, "last_cycle", test.loc[last, "cycle"].to_numpy())
    P.insert(2, "true_RUL", y.astype(int))
    P.to_csv(REPORT / "test_predictions.csv", index=False, float_format="%.2f")

    ci = boot_ci(y, {m: preds[m]["q50"].to_numpy() for m in preds})
    rows = []
    for m in ("B0", "B1", "M1"):
        p = preds[m]
        cap = FROZEN[m][1].get("r_max", FROZEN[m][1].get("cap", np.inf))
        reach = y <= cap
        rows.append({
            "model": m,
            "RMSE [95% CI]": f"{rmse(y, p['q50']):.2f} [{ci[m]['rmse'][0]:.1f}, {ci[m]['rmse'][1]:.1f}]",
            "score (sum) [95% CI]": f"{cmapss_score(y, p['q50']):.0f} [{ci[m]['score'][0]:.0f}, {ci[m]['score'][1]:.0f}]",
            "MAE": f"{np.mean(np.abs(p['q50'] - y)):.2f}",
            "bias (pred-true)": f"{np.mean(p['q50'] - y):+.2f}",
            "late predictions": f"{int((p['q50'] > y).sum())}/100",
            "cov 90%": f"{coverage(y, p['q05'], p['q95']):.0%}",
            "cov (RUL<=R_max)": f"{coverage(y[reach], p['q05'][reach], p['q95'][reach]):.0%}",
            "P(RUL>=q05)": f"{np.mean(y >= p['q05']):.0%}",
            "width": f"{width(p['q05'], p['q95']):.1f}",
            "α-λ hits (±20%)": f"{alpha_lambda_hits(y, p['q50']).mean():.0%}",
        })
    T = pd.DataFrame(rows)

    # paired bootstrap differences
    pr = []
    for a, b in (("B0", "B1"), ("B1", "M1"), ("B0", "M1")):
        d = ci[a]["rmse_samples"] - ci[b]["rmse_samples"]
        pr.append({"comparison": f"RMSE({a}) - RMSE({b})", "mean": f"{d.mean():.2f}",
                   "95% CI": f"[{np.percentile(d, 2.5):.2f}, {np.percentile(d, 97.5):.2f}]",
                   "P(second model better)": f"{(d > 0).mean():.1%}"})

    # decision at the last observed cycle
    dec = []
    for m in ("B0", "B1", "M1"):
        for bound in ("q05", "q50"):
            act = preds[m][bound].to_numpy() <= LEAD_TIME
            urgent = y < LEAD_TIME
            dec.append({"model": m, "bound": bound,
                        "urgent engines (true RUL < L)": int(urgent.sum()),
                        "caught": int((act & urgent).sum()),
                        "missed -> failure": int((~act & urgent).sum()),
                        "intervene, not urgent": int((act & ~urgent).sum()),
                        "life wasted by those (cycles, mean)": f"{np.mean(y[act & ~urgent] - LEAD_TIME):.1f}"
                        if (act & ~urgent).any() else "-"})

    lock = json.loads(LOCK.read_text())
    out = ["# Phase 3b - Single evaluation on the NASA test set (FD001)", "",
           f"_Generated by `python -m src.test_eval`. Frozen configuration hash `{FROZEN_HASH}`, first run "
           f"{lock['first_run']} (commit {lock['git_commit']}). Models refitted on the 100 training engines, "
           "prediction at the last observed cycle of each of the 100 test engines. CIs: bootstrap over "
           f"engines ({N_BOOT} resamples)._", "",
           "## 1. Point and interval metrics", "", md(T), "",
           "Score = standard C-MAPSS sum over the 100 engines (lower is better; late predictions cost more).", "",
           "## 2. Paired bootstrap comparisons", "", md(pd.DataFrame(pr)), "",
           f"## 3. Decision at the last observed cycle (L = {LEAD_TIME} cycles)", "",
           "Intervene now if the bound <= L. *Missed* = engine whose true RUL < L, left running -> would fail "
           "before maintenance.", "", md(pd.DataFrame(dec)), "",
           "Figure: `reports/figures/30_test_predictions.png`."]
    (REPORT / "test_results.md").write_text("\n".join(out) + "\n", encoding="utf-8")

    order = np.argsort(y)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    for ax, m, col in ((axes[0], "B1", "tab:orange"), (axes[1], "M1", "tab:green")):
        p = preds[m].iloc[order]
        x = np.arange(len(y))
        ax.vlines(x, p["q05"], p["q95"], color=col, alpha=.35, lw=2, label="90 % interval")
        ax.plot(x, p["q50"], "o", ms=3, color=col, label="prediction")
        ax.plot(x, y[order], "k.", ms=4, label="true RUL")
        ax.axhline(LEAD_TIME, color="red", ls=":", lw=1, label=f"lead time L={LEAD_TIME}")
        ax.set_title(f"{m} on the test set (engines sorted by true RUL)", fontsize=9)
        ax.set_xlabel("test engine (sorted)")
    axes[0].set_ylabel("RUL (cycles)"); axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "30_test_predictions.png", dpi=150); plt.close(fig)
    print("\n".join(out))


if __name__ == "__main__":
    main()
