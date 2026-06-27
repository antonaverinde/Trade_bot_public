"""
Parameter sweep for binary XGBoost forex classifier.

Goals: reduce overfitting, improve calibration, improve high-confidence precision.
Loads data once; sweeps XGB params + barrier params + post-hoc calibration.
Prints ranked summary table at the end.
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost

from sklearn.metrics import (
    roc_auc_score, log_loss, brier_score_loss,
    precision_score, recall_score, f1_score,
)
from sklearn.calibration import (
    calibration_curve, CalibratedClassifierCV
)

from Pipeline.pipeline import ForexDataLoader, ForexPipeline

MLFLOW_DB       = f"sqlite:///{os.path.abspath('mlflow.db')}"
EXPERIMENT_NAME = "xgboost_forex_binary_sweep"

# ── Confidence thresholds to measure precision at ───────────────────────────
CONF_THRESHOLDS = [0.55, 0.60, 0.65, 0.70]

# ── Barrier & data settings shared across all configs unless overridden ──────
BASE_DATA = dict(
    pair="EURUSD", years=[2023], timeframe="H1",
    weekends="filled", norm_method="fracdiff",
    target_type="triple_barrier", target_col="tb_label",
    lags=[1, 2, 5, 10], target_horizons=[1, 5, 15], gap_bars=50,
    scaling="none", fracdiff_d=0.3, threshold=6e-4,
    k_up=1.0, k_down=1.0, horizon_bars=10, barrier_price="hl",
    barrier_on_raw=True,
)

# ── Base XGB params ──────────────────────────────────────────────────────────
BASE_XGB = dict(
    n_estimators=1000, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=50, gamma=0.1,
    reg_alpha=0.1, reg_lambda=1.0,
    early_stopping_rounds=20,
    objective="binary:logistic", eval_metric="logloss",
    random_state=42, n_jobs=-1,
)

# ── Experiment configs ───────────────────────────────────────────────────────
# Each entry: (name, xgb_overrides, data_overrides, calibration)
# calibration: None | "platt" | "isotonic"
CONFIGS = [
    # ── Baseline ─────────────────────────────────────────────────────────────
    ("baseline",            {}, {}, None),

    # ── Depth reduction ───────────────────────────────────────────────────────
    ("depth3",              dict(max_depth=3), {}, None),
    ("depth4",              dict(max_depth=4), {}, None),

    # ── Heavier min_child_weight ─────────────────────────────────────────────
    ("mcw100",              dict(min_child_weight=100), {}, None),
    ("mcw200",              dict(min_child_weight=200), {}, None),

    # ── Gamma (min split gain) ────────────────────────────────────────────────
    ("gamma1",              dict(gamma=1.0), {}, None),
    ("gamma2",              dict(gamma=2.0), {}, None),

    # ── L2 regularization ────────────────────────────────────────────────────
    ("lambda5",             dict(reg_lambda=5.0), {}, None),
    ("lambda10",            dict(reg_lambda=10.0), {}, None),

    # ── L1 regularization ────────────────────────────────────────────────────
    ("alpha1",              dict(reg_alpha=1.0), {}, None),

    # ── Combined: moderate tightening ────────────────────────────────────────
    ("tight_moderate",      dict(max_depth=4, min_child_weight=100, gamma=0.5,
                                 reg_lambda=5.0, reg_alpha=0.5), {}, None),

    # ── Combined: strong tightening ──────────────────────────────────────────
    ("tight_strong",        dict(max_depth=3, min_child_weight=200, gamma=2.0,
                                 reg_lambda=10.0, reg_alpha=1.0,
                                 subsample=0.7, colsample_bytree=0.7), {}, None),

    # ── Subsample/colsample ───────────────────────────────────────────────────
    ("sub07_col07",         dict(subsample=0.7, colsample_bytree=0.7), {}, None),
    ("sub06_col06",         dict(subsample=0.6, colsample_bytree=0.6), {}, None),

    # ── Lower learning rate ───────────────────────────────────────────────────
    ("lr002",               dict(learning_rate=0.02, n_estimators=2000,
                                 early_stopping_rounds=40), {}, None),

    # ── Asymmetric barriers: 2:1 take-profit vs stop-loss ────────────────────
    ("kup2_kdown1",         {}, dict(k_up=2.0, k_down=1.0), None),

    # ── Longer hold horizon ───────────────────────────────────────────────────
    ("horizon15",           {}, dict(horizon_bars=15), None),

    # ── Asymmetric barriers + longer horizon ──────────────────────────────────
    ("kup2_h15",            {}, dict(k_up=2.0, k_down=1.0, horizon_bars=15), None),

    # ── Platt calibration on top of moderate-tight model ─────────────────────
    ("tight_moderate_platt",
                            dict(max_depth=4, min_child_weight=100, gamma=0.5,
                                 reg_lambda=5.0, reg_alpha=0.5), {}, "platt"),

    # ── Isotonic calibration on top of moderate-tight model ──────────────────
    ("tight_moderate_iso",
                            dict(max_depth=4, min_child_weight=100, gamma=0.5,
                                 reg_lambda=5.0, reg_alpha=0.5), {}, "isotonic"),

    # ── Best combined guess: tight + asymmetric barriers ─────────────────────
    ("tight_kup2_h15",      dict(max_depth=4, min_child_weight=100, gamma=0.5,
                                 reg_lambda=5.0, reg_alpha=0.5),
                            dict(k_up=2.0, k_down=1.0, horizon_bars=15), None),

    # ── More data: 3 years ────────────────────────────────────────────────────
    ("3yr_baseline",        {}, dict(years=[2021, 2022, 2023]), None),
    ("3yr_tight",           dict(max_depth=4, min_child_weight=100, gamma=0.5,
                                 reg_lambda=5.0, reg_alpha=0.5),
                            dict(years=[2021, 2022, 2023]), None),
    ("3yr_kup2",            {}, dict(years=[2021, 2022, 2023],
                                    k_up=2.0, k_down=1.0), None),

    # ── log_returns instead of fracdiff ───────────────────────────────────────
    ("logret_baseline",     {}, dict(norm_method="log_returns"), None),
    ("logret_tight",        dict(max_depth=4, min_child_weight=100, gamma=0.5,
                                 reg_lambda=5.0, reg_alpha=0.5),
                            dict(norm_method="log_returns"), None),
    ("logret_kup2",         {}, dict(norm_method="log_returns",
                                    k_up=2.0, k_down=1.0), None),
    ("logret_3yr_tight",    dict(max_depth=4, min_child_weight=100, gamma=0.5,
                                 reg_lambda=5.0, reg_alpha=0.5),
                            dict(norm_method="log_returns",
                                 years=[2021, 2022, 2023]), None),

    # ── Tight barriers (k=0.5) to create more decisive labels ────────────────
    ("k05_baseline",        {}, dict(k_up=0.5, k_down=0.5), None),
    ("k15_baseline",        {}, dict(k_up=1.5, k_down=1.5), None),
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def make_pipeline(data_cfg):
    return ForexPipeline(
        lags            = data_cfg["lags"],
        target_horizons = data_cfg["target_horizons"],
        gap_bars        = data_cfg["gap_bars"],
        scaling         = data_cfg["scaling"],
        norm_method     = data_cfg["norm_method"],
        fracdiff_d      = data_cfg["fracdiff_d"],
        target_type     = data_cfg["target_type"],
        k_up            = data_cfg["k_up"],
        k_down          = data_cfg["k_down"],
        horizon_bars    = data_cfg["horizon_bars"],
        threshold       = data_cfg["threshold"],
    )


def extract_xy(pipeline, results, split, target_col, feature_cols):
    X, y_raw = pipeline.get_xy(results[split], target_col, feature_cols)
    mask = y_raw != 0
    X = X[mask]
    y = (y_raw[mask] == 1).astype(int)
    return X, y


def prec_at_conf(y, proba_long, t):
    """Precision for long predictions where P(long)>=t, or NaN if empty."""
    mask = proba_long >= t
    if mask.sum() < 5:
        return np.nan
    pred_sel = np.ones(mask.sum(), dtype=int)
    return float(precision_score(y[mask], pred_sel, zero_division=0))


def prec_short_at_conf(y, proba_long, t):
    """Precision for short predictions where P(short)>=t."""
    proba_short = 1 - proba_long
    mask = proba_short >= t
    if mask.sum() < 5:
        return np.nan
    pred_sel = np.zeros(mask.sum(), dtype=int)
    return float(precision_score(y[mask], pred_sel, pos_label=0, zero_division=0))


def evaluate(model_or_calib, X, y, calibrated=False):
    proba_long = model_or_calib.predict_proba(X)[:, 1]
    auc   = roc_auc_score(y, proba_long)
    brier = brier_score_loss(y, proba_long)
    ll    = log_loss(y, np.column_stack([1 - proba_long, proba_long]))
    row = dict(auc=auc, brier=brier, logloss=ll)
    for t in CONF_THRESHOLDS:
        row[f"prec_long_{int(t*100)}"] = prec_at_conf(y, proba_long, t)
        row[f"prec_short_{int(t*100)}"] = prec_short_at_conf(y, proba_long, t)
        cov_long  = float((proba_long >= t).mean())
        cov_short = float(((1 - proba_long) >= t).mean())
        row[f"cov_long_{int(t*100)}"]  = cov_long
        row[f"cov_short_{int(t*100)}"] = cov_short
    return row, proba_long


def coverage_weighted_precision(eval_row, threshold_pct):
    """Combined score: mean long+short precision at given threshold, weighted by coverage."""
    p_long  = eval_row.get(f"prec_long_{threshold_pct}",  np.nan)
    p_short = eval_row.get(f"prec_short_{threshold_pct}", np.nan)
    c_long  = eval_row.get(f"cov_long_{threshold_pct}",   0.0)
    c_short = eval_row.get(f"cov_short_{threshold_pct}",  0.0)
    scores = []
    if np.isfinite(p_long)  and c_long  > 0: scores.append(p_long  * c_long)
    if np.isfinite(p_short) and c_short > 0: scores.append(p_short * c_short)
    return sum(scores) / max(c_long + c_short, 1e-9) if scores else np.nan


# ── Load data (shared across all configs with same data params) ──────────────

print("=" * 70)
print("Loading EURUSD 2023 M1 data …")
loader = ForexDataLoader()
df_m1_base = loader.load_and_merge(
    "histdata/", pair="EURUSD", years=[2023],
    weekends=BASE_DATA["weekends"],
)
print(f"Raw M1: {df_m1_base.shape[0]:,} bars  "
      f"{df_m1_base.index.min().date()} → {df_m1_base.index.max().date()}")
print("=" * 70)

mlflow.set_tracking_uri(MLFLOW_DB)
mlflow.set_experiment(EXPERIMENT_NAME)

rows = []
prev_data_key = None
results = None
pipeline_obj = None

for cfg_name, xgb_overrides, data_overrides, calibration in CONFIGS:
    data_cfg  = {**BASE_DATA, **data_overrides}
    xgb_cfg   = {**BASE_XGB,  **xgb_overrides}

    # Identify if data params changed so we only re-run the pipeline when needed
    data_key = (tuple(data_cfg["years"]), data_cfg["k_up"], data_cfg["k_down"],
                data_cfg["horizon_bars"], data_cfg["norm_method"])

    if data_key != prev_data_key:
        print(f"\n[data] years={data_cfg['years']} k_up={data_cfg['k_up']} "
              f"k_down={data_cfg['k_down']} horizon={data_cfg['horizon_bars']} "
              f"norm={data_cfg['norm_method']}")
        # Reload M1 if years changed
        if data_cfg["years"] != BASE_DATA["years"]:
            df_m1 = loader.load_and_merge(
                "histdata/", pair=data_cfg["pair"],
                years=data_cfg["years"], weekends=data_cfg["weekends"],
            )
        else:
            df_m1 = df_m1_base
        pipeline_obj = make_pipeline(data_cfg)
        results      = pipeline_obj.run(df_m1, timeframe=data_cfg["timeframe"])
        feature_cols = results["feature_cols"]
        target_col   = data_cfg["target_col"]
        prev_data_key = data_key

    try:
        X_train, y_train = extract_xy(pipeline_obj, results, "train", target_col, feature_cols)
        X_val,   y_val   = extract_xy(pipeline_obj, results, "val",   target_col, feature_cols)
        X_test,  y_test  = extract_xy(pipeline_obj, results, "test",  target_col, feature_cols)
    except ValueError as e:
        print(f"  [{cfg_name}] SKIP — {e}")
        continue

    print(f"\n[run] {cfg_name}  "
          f"train={len(y_train)} val={len(y_val)} test={len(y_test)}")

    # Train base XGBoost
    model = xgb.XGBClassifier(**xgb_cfg)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )
    best_iter = model.best_iteration
    print(f"  best_iter={best_iter}")

    # Optionally wrap in calibration (manual Platt/isotonic on val set)
    final_model = model
    if calibration in ("platt", "isotonic"):
        from sklearn.linear_model import LogisticRegression
        from sklearn.isotonic import IsotonicRegression

        proba_val_raw = model.predict_proba(X_val)[:, 1].reshape(-1, 1)

        class _CalibratedWrapper:
            def __init__(self, base, calibrator, method):
                self.base = base
                self.calibrator = calibrator
                self.method = method

            def predict_proba(self, X):
                raw = self.base.predict_proba(X)[:, 1].reshape(-1, 1)
                if self.method == "platt":
                    cal_p = self.calibrator.predict_proba(raw)[:, 1]
                else:
                    cal_p = self.calibrator.predict(raw.ravel()).clip(0, 1)
                return np.column_stack([1 - cal_p, cal_p])

        if calibration == "platt":
            lr = LogisticRegression()
            lr.fit(proba_val_raw, y_val)
            final_model = _CalibratedWrapper(model, lr, "platt")
        else:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(proba_val_raw.ravel(), y_val)
            final_model = _CalibratedWrapper(model, iso, "isotonic")

    # Evaluate
    tr_row, _ = evaluate(final_model, X_train, y_train)
    va_row, _ = evaluate(final_model, X_val,   y_val)
    te_row, _ = evaluate(final_model, X_test,  y_test)

    overfit_gap = va_row["auc"] - tr_row["auc"]   # negative = overfit

    # Overfitting measure on logloss (val - train, positive = overfit)
    ll_gap = va_row["logloss"] - tr_row["logloss"]

    row = {
        "config":       cfg_name,
        "calib":        calibration or "none",
        "best_iter":    best_iter,
        "k_up":         data_cfg["k_up"],
        "k_down":       data_cfg["k_down"],
        "horizon":      data_cfg["horizon_bars"],
        "max_depth":    xgb_cfg["max_depth"],
        "mcw":          xgb_cfg["min_child_weight"],
        "gamma":        xgb_cfg["gamma"],
        "lambda":       xgb_cfg["reg_lambda"],
        "alpha":        xgb_cfg["reg_alpha"],
        # AUC
        "tr_auc":       tr_row["auc"],
        "va_auc":       va_row["auc"],
        "te_auc":       te_row["auc"],
        "overfit_gap":  overfit_gap,  # negative = overfit, 0 = no overfit
        # Logloss gap (positive = overfit)
        "ll_gap":       ll_gap,
        # Brier (lower = better calibration)
        "va_brier":     va_row["brier"],
        "te_brier":     te_row["brier"],
    }

    # Per-threshold precision and coverage (val + test)
    for t in CONF_THRESHOLDS:
        tp = int(t * 100)
        for split_name, ev in [("va", va_row), ("te", te_row)]:
            row[f"{split_name}_pl{tp}"] = ev.get(f"prec_long_{tp}",  np.nan)
            row[f"{split_name}_ps{tp}"] = ev.get(f"prec_short_{tp}", np.nan)
            row[f"{split_name}_cl{tp}"] = ev.get(f"cov_long_{tp}",   np.nan)
            row[f"{split_name}_cs{tp}"] = ev.get(f"cov_short_{tp}",  np.nan)

    rows.append(row)

    # Quick console line
    va_pl60 = row.get("va_pl60", np.nan)
    va_ps60 = row.get("va_ps60", np.nan)
    print(f"  train_auc={tr_row['auc']:.4f}  val_auc={va_row['auc']:.4f}  "
          f"test_auc={te_row['auc']:.4f}  ll_gap={ll_gap:+.4f}  "
          f"va_brier={va_row['brier']:.4f}  "
          f"val_prec_long@60={va_pl60:.3f}  val_prec_short@60={va_ps60:.3f}")

    # MLflow log
    with mlflow.start_run(run_name=cfg_name):
        mlflow.log_params({
            **{f"xgb_{k}": v for k, v in xgb_cfg.items()},
            "calibration": calibration or "none",
            "k_up": data_cfg["k_up"], "k_down": data_cfg["k_down"],
            "horizon_bars": data_cfg["horizon_bars"],
        })
        mlflow.log_metrics({
            "train_auc": tr_row["auc"], "val_auc": va_row["auc"], "test_auc": te_row["auc"],
            "train_brier": tr_row["brier"], "val_brier": va_row["brier"], "test_brier": te_row["brier"],
            "overfit_ll_gap": ll_gap,
            **{f"val_{k}": v for k, v in va_row.items()},
            **{f"test_{k}": v for k, v in te_row.items()},
        })

# ── Summary table ────────────────────────────────────────────────────────────
print("\n\n" + "=" * 120)
print("PARAMETER SWEEP SUMMARY")
print("=" * 120)

df = pd.DataFrame(rows)

# Composite score: lower logloss gap + higher val AUC + higher prec@60 (long+short avg)
df["avg_prec60_va"] = (df["va_pl60"].fillna(0.5) + df["va_ps60"].fillna(0.5)) / 2
df["avg_prec65_va"] = (df["va_pl65"].fillna(0.5) + df["va_ps65"].fillna(0.5)) / 2

# Rank: (1) ll_gap < 0.05, (2) Brier, (3) avg_prec@60
df["rank_score"] = (
    - df["ll_gap"].clip(lower=0)         * 5    # penalise overfitting
    + df["va_auc"]                        * 2    # reward discrimination
    - df["va_brier"]                      * 3    # reward calibration
    + df["avg_prec60_va"]                 * 2    # reward high-conf precision
)
df_sorted = df.sort_values("rank_score", ascending=False)

display_cols = [
    "config", "calib",
    "max_depth", "mcw", "gamma", "lambda",
    "k_up", "horizon",
    "best_iter",
    "tr_auc", "va_auc", "te_auc", "ll_gap",
    "va_brier", "te_brier",
    "va_pl55", "va_ps55",
    "va_pl60", "va_ps60",
    "va_cl60", "va_cs60",
    "va_pl65", "va_ps65",
    "avg_prec60_va",
    "rank_score",
]
display_cols = [c for c in display_cols if c in df_sorted.columns]

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.4f}".format)
print(df_sorted[display_cols].to_string(index=False))

print("\n\nTOP 5 CONFIGS BY RANK SCORE:")
print("-" * 80)
for _, row in df_sorted.head(5).iterrows():
    print(f"\n  {row['config']}  (calib={row['calib']})")
    print(f"    AUC:  train={row['tr_auc']:.4f}  val={row['va_auc']:.4f}  "
          f"test={row['te_auc']:.4f}")
    print(f"    Logloss gap (val-train): {row['ll_gap']:+.4f}  "
          f"val_brier={row['va_brier']:.4f}  test_brier={row['te_brier']:.4f}")
    print(f"    Prec long @55%={row.get('va_pl55', np.nan):.4f}  "
          f"@60%={row.get('va_pl60', np.nan):.4f}  "
          f"@65%={row.get('va_pl65', np.nan):.4f}")
    print(f"    Prec short@55%={row.get('va_ps55', np.nan):.4f}  "
          f"@60%={row.get('va_ps60', np.nan):.4f}  "
          f"@65%={row.get('va_ps65', np.nan):.4f}")
    print(f"    Cov long @60%={row.get('va_cl60', np.nan):.4f}  "
          f"cov short@60%={row.get('va_cs60', np.nan):.4f}")

print("\n\nSweep done. MLflow experiment: " + EXPERIMENT_NAME)
print(f"DB: sqlite:///{os.path.abspath('mlflow.db')}")
