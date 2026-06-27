"""
Trade Bot — main entry point.

Loads 5 years of M1 histdata for EURUSD and GBPUSD, runs the feature
pipeline, trains XGBoost, and runs a Chronos zero-shot forecast check.

Key pipeline options (edit ForexPipeline call in main() to switch):
  norm_method   : "log_returns" | "fracdiff" | "raw"
  target_type   : "lag" | "triple_barrier"
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
from Pipeline.pipeline import ForexDataLoader, ForexPipeline

HISTDATA_DIR = Path(__file__).parent / "histdata"
PAIRS = ["EURUSD", "GBPUSD"]
TIMEFRAMES = ["M1", "M5", "M15", "H1"]


def train_xgboost(results: dict, pipeline: ForexPipeline) -> None:
    """Train XGBoost classifier and print val/test accuracy + AUC."""
    pair = results["pair"]
    tf   = results["timeframe"]
    feat_cols = results["feature_cols"]

    # Select the right classification target column
    if pipeline.target_type == "triple_barrier":
        target = "tb_label"
        # XGBoost multiclass: labels must be 0-indexed (−1 → 0, 0 → 1, 1 → 2)
        def remap(y): return y + 1
        objective = "multi:softprob"
        num_class = 3
        use_auc   = False
    else:
        target = "direction_1"
        remap  = lambda y: y
        objective = "binary:logistic"
        num_class = None
        use_auc   = True

    X_train, y_train = pipeline.get_xy(results["train"], target, feat_cols)
    X_val,   y_val   = pipeline.get_xy(results["val"],   target, feat_cols)
    X_test,  y_test  = pipeline.get_xy(results["test"],  target, feat_cols)

    y_train, y_val, y_test = remap(y_train), remap(y_val), remap(y_test)

    params = dict(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss" if not use_auc else "logloss",
        early_stopping_rounds=20,
        verbosity=0,
        objective=objective,
    )
    if num_class:
        params["num_class"] = num_class

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    for name, X, y in [("Val", X_val, y_val), ("Test", X_test, y_test)]:
        preds = model.predict(X)
        acc   = accuracy_score(y, preds)
        if use_auc:
            proba = model.predict_proba(X)[:, 1]
            auc   = roc_auc_score(y, proba)
            print(f"  [{pair} {tf}] XGBoost {name}: acc={acc:.4f}  AUC={auc:.4f}")
        else:
            print(f"  [{pair} {tf}] XGBoost {name}: acc={acc:.4f}  (multiclass, AUC skipped)")


def check_chronos(results: dict, pair: str, horizon: int = 60) -> None:
    """
    Zero-shot Chronos forecast on raw close prices.
    Uses result["raw_m1"] — always raw price levels regardless of norm_method.
    """
    try:
        import torch
        from chronos import BaseChronosPipeline
    except ImportError:
        print(f"  [Chronos] Skipped — install chronos-forecasting + torch first")
        print(f"  Run: uv add chronos-forecasting torch")
        return

    df_raw = results["raw_m1"]
    print(f"\n[Chronos] Running zero-shot forecast for {pair}...")
    close   = df_raw["close"].iloc[-512:].values
    context = torch.tensor(close, dtype=torch.float32).unsqueeze(0)

    pipe = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    forecast = pipe.predict(context, prediction_length=horizon)
    median   = np.median(forecast[0].numpy(), axis=0)

    print(f"  [{pair}] Last close: {close[-1]:.5f}")
    print(f"  [{pair}] Chronos forecast next {horizon} bars (median):")
    print(f"    min={median.min():.5f}  max={median.max():.5f}  "
          f"mean={median.mean():.5f}  end={median[-1]:.5f}")


def main():
    loader = ForexDataLoader()
    pipeline = ForexPipeline(
        lags=[1, 2, 5, 10],
        target_horizons=[1, 5, 15],
        gap_bars=50,
        norm_method="log_returns",    # "log_returns" | "fracdiff" | "raw"
        fracdiff_d=0.4,
        target_type="lag",            # "lag" | "triple_barrier"
        k_up=2.0,
        k_down=1.0,
        horizon_bars=10,
    )

    for pair in PAIRS:
        print(f"\n{'#'*60}")
        print(f"  Pair: {pair}")
        print(f"{'#'*60}")

        df_m1 = loader.load_and_merge(HISTDATA_DIR, pair)

        print("\n[XGBoost] Training on all timeframes...")
        for tf in TIMEFRAMES:
            results = pipeline.run(df_m1, timeframe=tf)
            train_xgboost(results, pipeline)

        # Chronos runs once per pair — uses raw_m1 from last pipeline.run() call
        check_chronos(results, pair)


if __name__ == "__main__":
    main()
