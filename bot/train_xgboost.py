from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from Pipeline.pipeline import ForexDataLoader, ForexPipeline
from strategy_fvg_fractals.pipeline import FVGFractalPipeline

from .config import FVG_DATA_CFG, H1_DATA_CFG


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"


BASE_XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 50,
    "gamma": 0.2,
    "reg_alpha": 0.2,
    "reg_lambda": 3.0,
    "early_stopping_rounds": 40,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "num_class": 3,
    "random_state": 42,
    "n_jobs": -1,
}


def _years(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _remap_direction(y: np.ndarray) -> np.ndarray:
    """-1,0,1 -> 0,1,2."""
    return (y.astype(int) + 1).astype(int)


def _class_weights(y: np.ndarray) -> np.ndarray:
    counts = Counter(int(v) for v in y)
    total = len(y)
    n_classes = max(len(counts), 1)
    return np.array([total / (n_classes * counts[int(v)]) for v in y], dtype=float)


def _safe_auc(y_true: np.ndarray, proba: np.ndarray, label: int) -> float:
    y_bin = (y_true == label).astype(int)
    if len(np.unique(y_bin)) < 2:
        return float("nan")
    return float(roc_auc_score(y_bin, proba[:, label]))


def _safe_ap(y_true: np.ndarray, proba: np.ndarray, label: int) -> float:
    y_bin = (y_true == label).astype(int)
    if len(np.unique(y_bin)) < 2:
        return float("nan")
    return float(average_precision_score(y_bin, proba[:, label]))


def _metrics(model, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    raw_proba = model.predict_proba(X)
    classes = [int(c) for c in getattr(model, "classes_", range(raw_proba.shape[1]))]
    proba = np.zeros((len(X), 3), dtype=float)
    for col, label in enumerate(classes):
        if 0 <= label <= 2:
            proba[:, label] = raw_proba[:, col]
    row_sum = proba.sum(axis=1)
    missing = row_sum <= 0
    if np.any(missing):
        proba[missing, 1] = 1.0
    pred = np.argmax(proba, axis=1)
    labels = [0, 1, 2]
    out = {
        "logloss": float(log_loss(y, proba, labels=labels)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "auc_short": _safe_auc(y, proba, 0),
        "auc_no_trade": _safe_auc(y, proba, 1),
        "auc_long": _safe_auc(y, proba, 2),
        "avg_precision_short": _safe_ap(y, proba, 0),
        "avg_precision_no_trade": _safe_ap(y, proba, 1),
        "avg_precision_long": _safe_ap(y, proba, 2),
    }
    for label, name in [(0, "short"), (1, "no_trade"), (2, "long")]:
        y_label = y == label
        pred_label = pred == label
        out[f"precision_{name}"] = float(
            precision_score(y_label, pred_label, zero_division=0)
        )
        out[f"recall_{name}"] = float(recall_score(y_label, pred_label, zero_division=0))
        out[f"pred_rate_{name}"] = float(pred_label.mean())
        out[f"actual_rate_{name}"] = float(y_label.mean())
    return out


def _log_common(
    model,
    experiment_name: str,
    run_name: str,
    data_cfg: dict,
    xgb_params: dict,
    feature_cols: list[str],
    arrays: dict[str, tuple[np.ndarray, np.ndarray]],
) -> str:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({f"data_{k}": str(v) for k, v in data_cfg.items()})
        mlflow.log_params({f"xgb_{k}": v for k, v in xgb_params.items()})
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_dict({"features": feature_cols}, "features.json")
        for split, (_, y) in arrays.items():
            for label, count in Counter(int(v) for v in y).items():
                mlflow.log_metric(f"{split}_class_{label}_count", count)
                mlflow.log_metric(f"{split}_class_{label}_rate", count / len(y))
        for split, (X, y) in arrays.items():
            mlflow.log_metrics({f"{split}_{k}": v for k, v in _metrics(model, X, y).items()})
        mlflow.xgboost.log_model(model, "model")
        return run.info.run_id


def train_h1(args) -> str:
    data_cfg = dict(H1_DATA_CFG)
    data_cfg.update(
        {
            "years": args.years,
            "barrier_norm_method": args.h1_barrier_norm_method,
            "barrier_price": "hl",
            "k_up": args.k_up,
            "k_down": args.k_down,
            "horizon_bars": args.horizon_bars,
        }
    )
    loader = ForexDataLoader()
    df_m1 = loader.load_and_merge(
        str(PROJECT_ROOT / "histdata"),
        pair=data_cfg["pair"],
        years=data_cfg["years"],
        weekends=data_cfg["weekends"],
    )
    pipeline = ForexPipeline(
        lags=data_cfg["lags"],
        target_horizons=data_cfg["target_horizons"],
        gap_bars=data_cfg["gap_bars"],
        scaling=data_cfg["scaling"],
        window_size=data_cfg["window_size"],
        norm_method=data_cfg["norm_method"],
        fracdiff_d=data_cfg["fracdiff_d"],
        target_type=data_cfg["target_type"],
        k_up=data_cfg["k_up"],
        k_down=data_cfg["k_down"],
        horizon_bars=data_cfg["horizon_bars"],
        barrier_price=data_cfg["barrier_price"],
        barrier_norm_method=data_cfg["barrier_norm_method"],
        threshold=data_cfg["threshold"],
        weekends=data_cfg["weekends"],
    )
    results = pipeline.run(df_m1, timeframe=data_cfg["timeframe"])
    feature_cols = results["feature_cols"]

    arrays = {}
    for split in ["train", "val", "test"]:
        X, y_raw = pipeline.get_xy(results[split], data_cfg["target_col"], feature_cols)
        arrays[split] = (X, _remap_direction(y_raw))

    xgb_params = dict(BASE_XGB_PARAMS)
    model = xgb.XGBClassifier(**xgb_params)
    X_train, y_train = arrays["train"]
    X_val, y_val = arrays["val"]
    sample_weight = _class_weights(y_train) if args.class_weight else None
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    run_name = (
        f"{data_cfg['pair']}_{data_cfg['timeframe']}_rawbarrier_"
        f"{min(data_cfg['years'])}-{max(data_cfg['years'])}"
    )
    return _log_common(
        model,
        "xgboost_forex_bot_raw",
        run_name,
        data_cfg,
        xgb_params,
        feature_cols,
        arrays,
    )


def train_fvg(args) -> str:
    data_cfg = dict(FVG_DATA_CFG)
    data_cfg.update({"years": args.years})
    loader = ForexDataLoader()
    df_m1 = loader.load_and_merge(
        str(PROJECT_ROOT / "histdata"),
        pair=data_cfg["pair"],
        years=data_cfg["years"],
        weekends=data_cfg["weekends"],
    )
    pipeline = FVGFractalPipeline(
        base_timeframe=data_cfg["base_timeframe"],
        higher_timeframe=data_cfg["higher_timeframe"],
        fractal_window=data_cfg["fractal_window"],
        lookahead_bars=data_cfg["lookahead_bars"],
        min_fvg_atr=data_cfg["min_fvg_atr"],
        lags=data_cfg["lags"],
        gap_events=data_cfg["gap_events"],
        scaling=data_cfg["scaling"],
        window_size=data_cfg["window_size"],
        norm_method=data_cfg["norm_method"],
        fracdiff_d=data_cfg["fracdiff_d"],
        threshold=data_cfg["threshold"],
        use_engineered_features=data_cfg["use_engineered_features"],
        decision_delay_bars=data_cfg["decision_delay_bars"],
        single_timeframe=data_cfg["single_timeframe"],
        require_unbroken_levels=data_cfg["require_unbroken_levels"],
    )
    results = pipeline.run(df_m1)
    feature_cols = results["feature_cols"]

    arrays = {}
    for split in ["train", "val", "test"]:
        X, y_raw = pipeline.get_xy(
            results[split],
            "target_first_break_dir",
            feature_cols,
            drop_timeout=False,
            binary_direction=False,
        )
        arrays[split] = (X, _remap_direction(y_raw))

    xgb_params = dict(BASE_XGB_PARAMS)
    model = xgb.XGBClassifier(**xgb_params)
    X_train, y_train = arrays["train"]
    X_val, y_val = arrays["val"]
    sample_weight = _class_weights(y_train) if args.class_weight else None
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    run_name = (
        f"{data_cfg['pair']}_{data_cfg['base_timeframe']}_fvg_multiclass_"
        f"{min(data_cfg['years'])}-{max(data_cfg['years'])}"
    )
    return _log_common(
        model,
        "xgboost_fvg_fractals_multiclass",
        run_name,
        data_cfg,
        xgb_params,
        feature_cols,
        arrays,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train bot-oriented XGBoost models.")
    parser.add_argument("--model", choices=["h1", "fvg", "both"], default="both")
    parser.add_argument("--years", type=_years, default=[2020, 2021, 2022, 2023, 2024])
    parser.add_argument("--class-weight", action="store_true")
    parser.add_argument("--h1-barrier-norm-method", choices=["raw", "log_returns"], default="raw")
    parser.add_argument("--k-up", type=float, default=2.0)
    parser.add_argument("--k-down", type=float, default=1.0)
    parser.add_argument("--horizon-bars", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_ids = {}
    if args.model in {"h1", "both"}:
        run_ids["h1"] = train_h1(args)
        print(f"h1_run_id={run_ids['h1']}")
    if args.model in {"fvg", "both"}:
        run_ids["fvg"] = train_fvg(args)
        print(f"fvg_run_id={run_ids['fvg']}")
    print(json.dumps(run_ids, indent=2))


if __name__ == "__main__":
    main()
