# Baseline, hypotheses and protocol (written before any model)

_Date: 2026-11-22. Written after the data diagnostic (`reports/diagnostic.md`) and **before** fitting
any model. Not edited after results come in; any later deviation goes in a dated "Changes" section
at the end._

## 1. Task

For an engine observed up to cycle *t* (FD001, one operating regime, one fault mode: HPC), predict
its remaining useful life RUL(t) in cycles, **with a 90 % interval [q05, q95]**, and decide whether
to schedule maintenance now.

## 2. Facts from the diagnostic that the design relies on

- Clean grid: no NaN, no duplicates, no missing cycles, single regime -> no imputation and no
  per-regime normalisation.
- 7 flat sensors (s1, s5, s6, s10, s16, s18, s19) are dropped. 12 sensors have a consistent
  monotone trend (s2, s3, s4, s7, s8, s11, s12, s13, s15, s17, s20, s21); s9 and s14 trend in
  opposite directions depending on the engine.
- Degradation is **flat for most of life, then accelerates** over roughly the last 100-150 cycles.
- Noise is high relative to drift (drift/noise ≈ 3-6): one cycle alone tells little.
- Test trajectories are **right-censored** (31-303 observed cycles); train trajectories run to failure.

## 3. Baselines (in increasing order of strength)

**B0 - population prior, no sensors.** RUL̂(t) = median over training engines of (L − t) among
engines with lifetime L > t (conditional remaining life). Interval: 5 %/95 % quantiles of the same
set. Any model that uses sensors must beat it.

**B1 - required baseline: health indicator + linear degradation model.**
1. *Sensor selection* (training units only): keep sensors whose median per-unit Spearman
   |ρ(sensor, cycle)| ≥ 0.5 with the same sign in ≥ 90 % of units.
2. *Health indicator*: standardise the selected sensors (scaler fitted on training units), then
   fit a linear regression to a target HI that is 1 in the first 30 cycles of every training engine
   and 0 in its last 30 cycles. HI(t) is the prediction, smoothed with a **trailing** mean.
3. *Failure threshold*: distribution of HI at the last cycle of training engines (keep its
   median and its 5 %/95 % quantiles).
4. *Extrapolation*: at cycle *t*, fit a straight line by least squares to the last **W** HI values
   (W ∈ {20, 30, 50}, chosen in cross-validation) and extrapolate to the threshold:
   RUL̂ = (HI(t) − threshold) / |slope|.
5. *Interval*: Monte Carlo over (slope ~ its OLS sampling distribution) × (threshold ~ its
   empirical distribution) -> q05, q50, q95.
6. If the slope isn't significantly negative (no visible degradation yet), or if the prediction
   is above R_max, predict R_max (same value as the cap of the regression target, §4). Predictions
   are clipped to [0, R_max].

The window is local, not the whole history, because degradation isn't linear over a whole life.
A straight line over the recent past is the "linear degradation" asked for, and the knee makes it
pessimistic early and accurate late.

## 4. Richer model (Phase 2, only after B0/B1 are in place)

**M1 - direct RUL regression with gradient boosting** (LightGBM), quantile objectives for q05, q50
and q95 (plus split conformal calibration by unit if coverage is off).
- Features, all **strictly causal per unit**: current values of the selected sensors, trailing
  rolling mean/std/slope over windows {5, 10, 20, 30} (`groupby("unit")`, right-aligned windows,
  `min_periods` for short histories), EWM, current cycle.
- Target capped at **R_max**: RUL_target = min(RUL, R_max), R_max ∈ {100, 125, 150}, chosen in
  cross-validation. *Justification:* the diagnostic shows no detectable trend before roughly 100-150
  cycles from failure (the course's P-F curve: an alert can only be useful between P and F). An
  uncapped linear target asks the model to tell RUL = 300 from RUL = 200 using signals that are
  identical in both cases. That is unlearnable and inflates the error.
- Ablation: cap vs no cap and the R_max value; selected sensors vs all 14; with vs without rolling
  features; window sizes; quantile vs conformal intervals.

## 5. Validation protocol (leak-free)

- **Groups = engines.** 5-fold `GroupKFold` on the 100 training engines (fixed seed). An engine is
  never split between training and validation.
- **Everything that is fitted is fitted inside the fold on training engines only:** sensor
  selection, scaler, HI regression, failure threshold, B0 quantiles, GBM, conformal calibration
  (a held-out subset of the training engines of the fold). Implemented as a per-fold pipeline.
- **Censoring reproduced:** each validation engine is evaluated (a) at 5 random cut points (fixed
  seed, uniform over its life, keeping ≥ 31 observed cycles like the shortest test history) for
  RMSE / score / coverage, and (b) at every cycle for the α-λ diagram and the decision simulation.
- **Time arrow:** features at cycle *t* use cycles ≤ *t* of the same engine only. No centred
  windows, `filtfilt`, `bfill`, Savitzky-Golay, or statistics over a whole trajectory. Engine
  lifetime and RUL never enter the features.
- **On "gap":** engines are independent, so the separation between training and evaluation is
  between engines, not between periods. The gap the course asks for corresponds here to (i) no
  cycle of a validation engine being in training and (ii) features that are strictly trailing.
- **Model selection** (W, R_max, hyperparameters) on cross-validation only.
- **NASA test set** (`test_FD001` + `RUL_FD001`): evaluated **once**, at the end, with the frozen
  configuration, for B0, B1 and M1 together. Its RUL labels are not looked at before then.

## 6. Metrics (with d = RUL̂ − RUL)

- **RMSE** in cycles.
- **C-MAPSS score** S = Σ [exp(−d/13) − 1] if d < 0, [exp(d/10) − 1] if d ≥ 0. Late predictions
  (d > 0) cost more.
- **Coverage** of [q05, q95] (target 90 %) and **mean interval width** (a wide interval covers
  trivially, so both are reported).
- **α-λ diagram**, α = 20 %: share of engines whose prediction is inside ±20 % of the true RUL at
  λ = 25 %, 50 %, 75 % of their life.
- Cross-validation results reported as mean ± standard deviation across folds.

## 7. Decision rule

- **Logistics lead time L = 20 cycles** (parts + intervention). This is an assumed value since the
  subject says only "a given lead time"; it's a parameter, with sensitivity at L ∈ {10, 20, 30}.
- **Rule:** schedule maintenance at the first cycle where **q05(RUL) ≤ L**.
- Evaluated over the full trajectories of validation engines:
  - *failure*: the engine fails less than L cycles after the alert, or no alert is ever raised;
  - *wasted life*: cycles between the intervention and the actual failure.
- Same rule with the median q50 instead of q05, to show what deciding on the lower bound buys.

## 8. Testable hypotheses (each one can be shown wrong)

| # | Hypothesis | Shown wrong if |
|---|---|---|
| H1 | B1 beats B0 on RMSE and score. | RMSE(B1) ≥ RMSE(B0) in cross-validation. |
| H2 | M1 beats B1 on RMSE and score. | No improvement beyond the fold-to-fold std. |
| H3 | The capped target beats the uncapped one for M1. | Capped RMSE ≥ uncapped RMSE. |
| H4 | The raw quantile intervals of M1 under-cover (< 90 %), and conformal fixes it. | Raw coverage already ≈ 90 %, or conformal doesn't help. |
| H5 | Errors are much larger far from failure; α-λ accuracy rises with λ. | Flat α-λ curve. |
| H6 | The q05 rule gives fewer failures than the q50 rule at L = 20, for a moderate amount of extra wasted life. | Same failure count. |

Expected order of magnitude (published FD001 results, for reference only, not a target): RMSE about
30-45 cycles for a linear HI-extrapolation baseline, about 15-20 for capped gradient boosting.

## 9. Reproducibility

Pinned environment (`requirements.txt`), seed 42 everywhere, raw files kept read-only and
identified by their SHA-256 (in `reports/diagnostic.md`), one command to rerun everything.

## Changes

**2026-11-22, Phase 2 start, before any model was fitted.** Details settled while implementing:

1. *Decision zone.* Random cut points give true RULs of up to about 360 cycles, while a capped
   model is designed to answer "≥ R_max" far from failure. Cross-validation metrics are therefore
   reported **twice**: on all evaluation points, and on the **decision zone** (true RUL ≤ 150,
   fixed for every model, equal to the largest R_max candidate so it doesn't favour any cap).
   **Model selection uses the RMSE in the decision zone.** The NASA test set is reported overall,
   in the standard way.
2. *Coverage.* B1 and M1 intervals are bounded above by R_max, so they can't cover a true RUL
   above R_max by construction. Coverage is reported on the decision zone (main figure) and
   overall. Conformal calibration (CQR) is done against the capped target.
3. *B1 implementation.* "HI smoothed with a trailing mean" is done by the trailing OLS fit itself:
   HI(t) is the endpoint at *t* of the line fitted to the last W raw HI values. This is still
   strictly causal and gives the standard error needed for the interval. The failure threshold
   uses the same estimator at the last cycle of each training engine.
4. *Score in cross-validation* = mean per evaluation point (folds have different sizes). On the
   test set: the standard sum over the 100 engines.
5. *Point B0* is not clipped at R_max (it's the sensor-free prior).
6. *GBM hyperparameters* are fixed beforehand (500 trees, learning rate 0.03, 31 leaves,
   min_child_samples 50, subsample/colsample 0.8) and not tuned. The ablation covers the
   modelling choices (cap, sensors, features, intervals).
7. *Evaluation cut points* are drawn once per engine (seed 42, uniform over [31, L]), so all
   models are compared on exactly the same points.

**2026-11-22, after the first cross-validation run (training engines only, test set untouched).**

8. *Bug fixed in M1:* with cap = 100, more than half of the training rows sit exactly at the cap.
   LightGBM's quantile objective starts from the target quantile (= the cap), so every row gets the
   same gradient, no split is ever made, and the model predicts a constant 100 (CV RMSE about 50).
   Fix: `boost_from_average=False` for every quantile model (fold 0: cap 100 goes 50.3 → 20.0;
   neutral at caps 125 and 150). The cap comparison is re-run with the fix.
9. *B1 intervals added:* the Monte Carlo interval of §3 (slope, level and threshold noise) covers
   only about 10-40 % of true RULs in the decision zone instead of 90 %. It leaves out the
   model-shape error (a straight line extrapolating a curve that bends near failure) and assumes
   independent residuals. That result is kept and reported. Added alongside it: a **split-conformal
   B1 interval** (errors of B1 on 20 % held-out training engines of the fold, binned by predicted
   RUL), the same principle as M1's CQR.
10. *Two interval metrics added (reporting only, the selection rule is unchanged):* coverage on
    points with true RUL ≤ R_max (where a capped interval can contain the truth), and
    P(RUL ≥ q05) in the decision zone, i.e. the validity of the lower bound the decision rule
    uses (target 95 %).
11. *Frozen configurations for the test set*, from the pre-registered rule (best RMSE_zone per
    family): B0; B1 W = 30, R_max = 100 with conformal intervals; M1 cap = 125, all non-flat
    sensors (16.8 ± 1.0 vs 17.3 ± 1.7 with selected sensors, a difference within fold noise),
    full features, CQR intervals. Calibrated intervals are used because the raw ones under-cover.
