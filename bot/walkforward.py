from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from Pipeline.pipeline import ForexDataLoader, ForexPipeline
from strategy_fvg_fractals.pipeline import FVGFractalPipeline

from .config import CostConfig, DecisionParams, FVG_DATA_CFG, H1_DATA_CFG, OptimizerConfig, RiskConfig
from .fvg_utils import add_fvg_profit_labels, fvg_level_candidates
from .model_adapters import SignalDataset, _class_probabilities, _market_with_indicators, _next_bar_times
from .optimize import _flat_metrics, objective_score, run_optimization
from .reports import write_backtest_outputs
from .simulator import BacktestResult, simulate
from .train_xgboost import BASE_XGB_PARAMS, _class_weights, _metrics, _remap_direction


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FOLDS = [
    {
        "name": "fold_a_2023_test",
        "train_start": "2020-01-01",
        "train_end": "2021-12-31 23:59:59",
        "val_start": "2022-01-01",
        "val_end": "2022-12-31 23:59:59",
        "test_start": "2023-01-01",
        "test_end": "2023-12-31 23:59:59",
        "years": [2020, 2021, 2022, 2023],
    },
    {
        "name": "fold_b_2024_test",
        "train_start": "2020-01-01",
        "train_end": "2022-12-31 23:59:59",
        "val_start": "2023-01-01",
        "val_end": "2023-12-31 23:59:59",
        "test_start": "2024-01-01",
        "test_end": "2024-12-31 23:59:59",
        "years": [2020, 2021, 2022, 2023, 2024],
    },
]

_FVG_CACHE: dict[tuple, dict] = {}
FVG_CACHE_VERSION = 1

FVG_LEVEL_FAMILIES = {
    "all": (None, True),
    "base_only": ({"base_last", "base_second"}, True),
    "higher_only": ({"higher_last", "higher_second"}, True),
    "no_seconds": ({"base_last", "higher_last"}, True),
    "no_gap": (None, False),
}


def _slice(df: pd.DataFrame, start: str, end: str, time_col: str | None = None) -> pd.DataFrame:
    if time_col is None:
        idx = pd.DatetimeIndex(df.index)
    else:
        idx = pd.DatetimeIndex(df[time_col])
    mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
    return df.loc[mask].copy()


def _dataset_market(market_full: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return market_full.iloc[0:0].copy()
    return market_full.loc[signals.index.min():signals.index.max()].copy()


def _split_signals_in_half(signals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    signals = signals.sort_index()
    split_at = len(signals) // 2
    return signals.iloc[:split_at].copy(), signals.iloc[split_at:].copy()


def _fvg_disk_cache_path(cache_key: tuple) -> Path:
    cache_root = PROJECT_ROOT / "outputs" / "fvg_feature_cache"
    payload = json.dumps(cache_key, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return cache_root / f"fvg_{digest}.pkl"


def _simulate_signal_subset(
    market_full: pd.DataFrame,
    signals: pd.DataFrame,
    params: DecisionParams,
    costs: CostConfig,
    risk: RiskConfig,
) -> BacktestResult:
    return simulate(_dataset_market(market_full, signals), signals, params, costs, risk)


def _fit_xgb(
    arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    class_weight: bool,
    seed: int,
    n_estimators: int,
) -> xgb.XGBClassifier:
    params = dict(BASE_XGB_PARAMS)
    params["random_state"] = seed
    params["n_estimators"] = n_estimators
    model = xgb.XGBClassifier(**params)
    X_train, y_train = arrays["train"]
    X_val, y_val = arrays["val"]
    sample_weight = _class_weights(y_train) if class_weight else None
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def _fit_xgb_regressor(
    arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    seed: int,
    n_estimators: int,
) -> xgb.XGBRegressor:
    params = dict(BASE_XGB_PARAMS)
    params.pop("num_class", None)
    params["random_state"] = seed
    params["n_estimators"] = n_estimators
    params["objective"] = "reg:squarederror"
    params["eval_metric"] = "rmse"
    model = xgb.XGBRegressor(**params)
    X_train, y_train = arrays["train"]
    X_val, y_val = arrays["val"]
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def _fit_ridge_regressor(arrays: dict[str, tuple[np.ndarray, np.ndarray]]) -> object:
    X_train, y_train = arrays["train"]
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=10.0),
    )
    model.fit(X_train, y_train)
    return model


def _regression_metrics(model, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred = _predict_regression(model, X)
    err = pred - y
    corr = float(np.corrcoef(pred, y)[0, 1]) if len(y) > 1 and np.std(pred) > 0 and np.std(y) > 0 else 0.0
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))) if len(y) else 0.0,
        "mae": float(np.mean(np.abs(err))) if len(y) else 0.0,
        "corr": corr if math.isfinite(corr) else 0.0,
        "target_mean": float(np.mean(y)) if len(y) else 0.0,
        "target_positive_rate": float((y > 0).mean()) if len(y) else 0.0,
        "pred_mean": float(np.mean(pred)) if len(pred) else 0.0,
        "pred_positive_rate": float((pred > 0).mean()) if len(pred) else 0.0,
    }


def _predict_regression(model, X: np.ndarray) -> np.ndarray:
    if isinstance(model, list):
        preds = [np.asarray(item.predict(X), dtype=float) for item in model]
        return np.mean(preds, axis=0)
    return np.asarray(model.predict(X), dtype=float)


def _predict_regression_std(model, X: np.ndarray) -> np.ndarray:
    if not isinstance(model, list):
        return np.full(len(X), np.nan, dtype=float)
    preds = [np.asarray(item.predict(X), dtype=float) for item in model]
    return np.std(preds, axis=0)


def _h1_signals(
    split: pd.DataFrame,
    raw_bars: pd.DataFrame,
    market_full: pd.DataFrame,
    feature_cols: list[str],
    model,
    cfg: dict,
) -> pd.DataFrame:
    X = split[feature_cols].to_numpy()
    probs = _class_probabilities(model, X, labels=[0, 1, 2])
    entry_times, valid = _next_bar_times(raw_bars.index, split.index)
    source_times = pd.DatetimeIndex(split.index)[valid]
    source_market = market_full.reindex(source_times)
    source_close = source_market["close"].to_numpy()
    sigma = source_market["sigma_pct"].fillna(0.0).to_numpy()
    upper = source_close * (1.0 + sigma * float(cfg["k_up"]))
    lower = source_close * (1.0 - sigma * float(cfg["k_down"]))
    frame = pd.DataFrame(
        {
            "p_short": probs[0][valid],
            "p_hold": probs[1][valid],
            "p_long": probs[2][valid],
            "source_time": source_times,
            "long_take_price": upper,
            "long_stop_price": lower,
            "short_take_price": lower,
            "short_stop_price": upper,
            "level_basis": "h1_raw_vol_barrier",
        },
        index=entry_times,
    )
    return frame.sort_index().join(market_full[["open"]], how="inner").drop(columns=["open"])


def _fvg_signals(
    split: pd.DataFrame,
    market_full: pd.DataFrame,
    feature_cols: list[str],
    model,
    label_mode: str,
    pair: str,
    allowed_sources: set[str] | None,
    use_gap_fallback: bool,
) -> pd.DataFrame:
    X = split[feature_cols].to_numpy()
    predicted_long_net_pips = np.full(len(split), np.nan, dtype=float)
    predicted_long_net_pips_std = np.full(len(split), np.nan, dtype=float)
    if label_mode == "rule_long":
        p_long = np.ones(len(split), dtype=float)
        p_hold = np.zeros(len(split), dtype=float)
        p_short = np.zeros(len(split), dtype=float)
    elif label_mode == "profit_pips":
        classifier = model["classifier"]
        regressor = model["regressor"]
        probs = _class_probabilities(classifier, X, labels=[0, 1, 2])
        p_hold = probs[1]
        p_long = probs[2]
        p_short = probs[0]
        predicted_long_net_pips = np.asarray(regressor.predict(X), dtype=float)
    elif label_mode in {"long_pips", "long_pips_bagged", "long_pips_ridge"}:
        predicted_long_net_pips = _predict_regression(model, X)
        predicted_long_net_pips_std = _predict_regression_std(model, X)
        p_long = np.ones(len(split), dtype=float)
        p_hold = np.zeros(len(split), dtype=float)
        p_short = np.zeros(len(split), dtype=float)
    else:
        probs = _class_probabilities(model, X, labels=[0, 1, 2])
        if label_mode == "long_profit":
            p_short = np.zeros(len(split), dtype=float)
            p_hold = probs[0]
            p_long = probs[1]
        elif label_mode == "profit":
            p_hold = probs[1]
            p_long = probs[2]
            p_short = probs[0]
        else:
            p_short = probs[0]
            p_hold = probs[1]
            p_long = probs[2]

    rows = []
    pip_size = 0.01 if pair.endswith("JPY") else 0.0001
    for i, (_, row) in enumerate(split.iterrows()):
        levels = fvg_level_candidates(
            row,
            allowed_sources=allowed_sources,
            use_gap_fallback=use_gap_fallback,
        )
        high = levels["high"]
        low = levels["low"]
        size = abs(float(row["fvg_gap_high"]) - float(row["fvg_gap_low"]))
        atr = float(row["atr"])
        rows.append(
            {
                "entry_time": row["decision_time"],
                "p_short": p_short[i],
                "p_hold": p_hold[i],
                "p_long": p_long[i],
                "predicted_long_net_pips": predicted_long_net_pips[i],
                "predicted_short_net_pips": np.nan,
                "predicted_long_net_pips_std": predicted_long_net_pips_std[i],
                "source_time": row.name,
                "long_take_price": high.price if high is not None else np.nan,
                "long_stop_price": low.price if low is not None else np.nan,
                "short_take_price": low.price if low is not None else np.nan,
                "short_stop_price": high.price if high is not None else np.nan,
                "level_basis": "fvg_known_fractal_levels",
                "long_take_source": high.source if high is not None else "",
                "long_stop_source": low.source if low is not None else "",
                "short_take_source": low.source if low is not None else "",
                "short_stop_source": high.source if high is not None else "",
                "fvg_size_pips": size / pip_size,
                "fvg_size_atr": size / atr if np.isfinite(atr) and atr > 0 else np.nan,
                "signal_atr": atr,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.set_index("entry_time").sort_index()
    frame = _collapse_duplicate_rows(frame)
    return frame.join(market_full[["open"]], how="inner").drop(columns=["open"])


def _collapse_duplicate_rows(signals: pd.DataFrame) -> pd.DataFrame:
    if not signals.index.has_duplicates:
        return signals.sort_index()
    rows = []
    for entry_time, group in signals.groupby(level=0, sort=True):
        score = group[["p_long", "p_short"]].max(axis=1)
        best = group.loc[score.idxmax()].copy()
        best["entry_time"] = entry_time
        rows.append(best.to_dict())
    return pd.DataFrame(rows).set_index("entry_time").sort_index()


def build_h1_dataset(fold: dict, seed: int, class_weight: bool, n_estimators: int) -> tuple[SignalDataset, dict]:
    cfg = dict(H1_DATA_CFG)
    cfg.update({"years": fold["years"], "barrier_norm_method": "log_returns"})
    loader = ForexDataLoader()
    df_m1 = loader.load_and_merge(
        str(PROJECT_ROOT / "histdata"),
        pair=cfg["pair"],
        years=cfg["years"],
        weekends=cfg["weekends"],
    )
    pipeline = ForexPipeline(
        lags=cfg["lags"],
        target_horizons=cfg["target_horizons"],
        gap_bars=cfg["gap_bars"],
        scaling=cfg["scaling"],
        window_size=cfg["window_size"],
        norm_method=cfg["norm_method"],
        fracdiff_d=cfg["fracdiff_d"],
        target_type=cfg["target_type"],
        k_up=cfg["k_up"],
        k_down=cfg["k_down"],
        horizon_bars=cfg["horizon_bars"],
        barrier_price=cfg["barrier_price"],
        barrier_norm_method=cfg["barrier_norm_method"],
        threshold=cfg["threshold"],
        weekends=cfg["weekends"],
    )
    results = pipeline.run(df_m1, timeframe=cfg["timeframe"])
    feature_cols = results["feature_cols"]
    all_frame = pd.concat([results["train"], results["val"], results["test"]]).sort_index()
    splits = {
        "train": _slice(all_frame, fold["train_start"], fold["train_end"]),
        "val": _slice(all_frame, fold["val_start"], fold["val_end"]),
        "test": _slice(all_frame, fold["test_start"], fold["test_end"]),
    }
    arrays = {}
    for split_name, frame in splits.items():
        X = frame[feature_cols].to_numpy()
        y = _remap_direction(frame[cfg["target_col"]].to_numpy())
        arrays[split_name] = (X, y)
    model = _fit_xgb(arrays, class_weight, seed, n_estimators)

    raw_bars = results["raw_m1"]
    market_full = _market_with_indicators(raw_bars, cfg["pair"])
    split_signals = {
        "val": _h1_signals(splits["val"], raw_bars, market_full, feature_cols, model, cfg),
        "test": _h1_signals(splits["test"], raw_bars, market_full, feature_cols, model, cfg),
    }
    split_market = {name: _dataset_market(market_full, signals) for name, signals in split_signals.items()}
    dataset = SignalDataset(
        "h1",
        f"in_memory_{fold['name']}",
        split_signals,
        split_market,
        raw_bars,
        cfg,
        feature_cols,
    )
    metrics = {split: _metrics(model, *arrays[split]) for split in ["train", "val", "test"]}
    return dataset, metrics


def build_fvg_dataset(
    fold: dict,
    seed: int,
    class_weight: bool,
    n_estimators: int,
    label_mode: str,
    profit_buffer_pips: float,
    base_timeframe: str,
    higher_timeframe: str,
    level_family: str,
    min_fvg_atr: float | None = None,
    require_unbroken_levels: bool | None = None,
    decision_delay_bars: int | None = None,
    single_timeframe: bool | None = None,
) -> tuple[SignalDataset, dict]:
    if level_family not in FVG_LEVEL_FAMILIES:
        raise ValueError(f"Unknown FVG level family: {level_family}")
    allowed_sources, use_gap_fallback = FVG_LEVEL_FAMILIES[level_family]
    cfg = dict(FVG_DATA_CFG)
    cfg.update(
        {
            "years": fold["years"],
            "base_timeframe": base_timeframe,
            "higher_timeframe": higher_timeframe,
        }
    )
    if min_fvg_atr is not None:
        cfg["min_fvg_atr"] = min_fvg_atr
    if require_unbroken_levels is not None:
        cfg["require_unbroken_levels"] = require_unbroken_levels
    if decision_delay_bars is not None:
        cfg["decision_delay_bars"] = decision_delay_bars
    if single_timeframe is not None:
        cfg["single_timeframe"] = single_timeframe
    cache_key = (
        FVG_CACHE_VERSION,
        tuple(cfg["years"]),
        cfg["pair"],
        cfg["base_timeframe"],
        cfg["higher_timeframe"],
        cfg["min_fvg_atr"],
        cfg["decision_delay_bars"],
        cfg["single_timeframe"],
        cfg["require_unbroken_levels"],
    )
    if cache_key in _FVG_CACHE:
        results = _FVG_CACHE[cache_key]
    else:
        cache_path = _fvg_disk_cache_path(cache_key)
        if cache_path.exists():
            try:
                with cache_path.open("rb") as fh:
                    results = pickle.load(fh)
                print(f"[FVG cache] loaded {cache_path}")
            except (OSError, pickle.PickleError, EOFError, AttributeError, ValueError) as exc:
                print(f"[FVG cache] ignoring unreadable cache {cache_path}: {exc}")
                results = None
        else:
            results = None

        if results is None:
            loader = ForexDataLoader()
            df_m1 = loader.load_and_merge(
                str(PROJECT_ROOT / "histdata"),
                pair=cfg["pair"],
                years=cfg["years"],
                weekends=cfg["weekends"],
            )
            pipeline = FVGFractalPipeline(
                base_timeframe=cfg["base_timeframe"],
                higher_timeframe=cfg["higher_timeframe"],
                fractal_window=cfg["fractal_window"],
                lookahead_bars=cfg["lookahead_bars"],
                min_fvg_atr=cfg["min_fvg_atr"],
                lags=cfg["lags"],
                gap_events=cfg["gap_events"],
                scaling=cfg["scaling"],
                window_size=cfg["window_size"],
                norm_method=cfg["norm_method"],
                fracdiff_d=cfg["fracdiff_d"],
                threshold=cfg["threshold"],
                use_engineered_features=cfg["use_engineered_features"],
                decision_delay_bars=cfg["decision_delay_bars"],
                single_timeframe=cfg["single_timeframe"],
                require_unbroken_levels=cfg["require_unbroken_levels"],
            )
            results = pipeline.run(df_m1)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("wb") as fh:
                pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[FVG cache] saved {cache_path}")
        _FVG_CACHE[cache_key] = results
    raw_bars = results["raw_base"]
    pip_size = 0.01 if cfg["pair"].endswith("JPY") else 0.0001
    events = add_fvg_profit_labels(
        results["events"],
        raw_bars,
        costs=CostConfig(),
        pip_size=pip_size,
        max_hold_bars=min(48, int(cfg["lookahead_bars"])),
        profit_buffer_pips=profit_buffer_pips,
        allowed_sources=allowed_sources,
        use_gap_fallback=use_gap_fallback,
    )
    feature_cols = results["feature_cols"]
    splits = {
        "train": _slice(events, fold["train_start"], fold["train_end"], "decision_time"),
        "val": _slice(events, fold["val_start"], fold["val_end"], "decision_time"),
        "test": _slice(events, fold["test_start"], fold["test_end"], "decision_time"),
    }
    rule_mode = label_mode == "rule_long"
    regression_mode = label_mode in {"long_pips", "long_pips_bagged", "long_pips_ridge"}
    hybrid_mode = label_mode == "profit_pips"
    if rule_mode:
        target = "target_long_net_pips"

        def y_transform(y):
            values = pd.Series(y).astype(float).replace([np.inf, -np.inf], np.nan).fillna(-20.0).to_numpy()
            return np.clip(values, -80.0, 80.0).astype(float)
    elif label_mode in {"profit", "profit_pips"}:
        target = "target_profit_label"
        # Profit target is 0=no_trade, 1=long, 2=short. Convert to bot convention:
        # 0=short, 1=no_trade, 2=long.
        mapper = {0: 1, 1: 2, 2: 0}
        y_transform = lambda y: np.array([mapper[int(v)] for v in y], dtype=int)
    elif label_mode == "long_profit":
        target = "target_profit_label"
        # Binary long-only target: 1=profitable long, 0=skip everything else.
        y_transform = lambda y: (y.astype(int) == 1).astype(int)
    elif label_mode in {"long_pips", "long_pips_bagged", "long_pips_ridge"}:
        target = "target_long_net_pips"

        def y_transform(y):
            values = pd.Series(y).astype(float).replace([np.inf, -np.inf], np.nan).fillna(-20.0).to_numpy()
            return np.clip(values, -80.0, 80.0).astype(float)
    else:
        target = "target_first_break_dir"
        y_transform = _remap_direction

    arrays = {}
    for split_name, frame in splits.items():
        X = frame[feature_cols].to_numpy()
        y = y_transform(frame[target].to_numpy())
        arrays[split_name] = (X, y)

    if rule_mode:
        model = None
    elif hybrid_mode:
        reg_arrays = {}
        for split_name, frame in splits.items():
            X = frame[feature_cols].to_numpy()
            y = pd.Series(frame["target_long_net_pips"]).astype(float).replace([np.inf, -np.inf], np.nan).fillna(-20.0).to_numpy()
            reg_arrays[split_name] = (X, np.clip(y, -80.0, 80.0).astype(float))
        model = {
            "classifier": _fit_xgb(arrays, class_weight, seed, n_estimators),
            "regressor": _fit_xgb_regressor(reg_arrays, seed, n_estimators),
        }
    elif label_mode == "long_pips_bagged":
        model = [
            _fit_xgb_regressor(arrays, seed + offset, n_estimators)
            for offset in [0, 17, 43]
        ]
    elif label_mode == "long_pips_ridge":
        model = _fit_ridge_regressor(arrays)
    elif regression_mode:
        model = _fit_xgb_regressor(arrays, seed, n_estimators)
    else:
        model = _fit_xgb(arrays, class_weight, seed, n_estimators)

    market_full = _market_with_indicators(raw_bars, cfg["pair"])
    split_signals = {
        "val": _fvg_signals(
            splits["val"],
            market_full,
            feature_cols,
            model,
            label_mode,
            cfg["pair"],
            allowed_sources,
            use_gap_fallback,
        ),
        "test": _fvg_signals(
            splits["test"],
            market_full,
            feature_cols,
            model,
            label_mode,
            cfg["pair"],
            allowed_sources,
            use_gap_fallback,
        ),
    }
    split_market = {name: _dataset_market(market_full, signals) for name, signals in split_signals.items()}
    dataset = SignalDataset(
        "fvg",
        f"in_memory_{label_mode}_{fold['name']}",
        split_signals,
        split_market,
        raw_bars,
        {
            **cfg,
            "label_mode": label_mode,
            "profit_buffer_pips": profit_buffer_pips,
            "level_family": level_family,
            "allowed_level_sources": sorted(allowed_sources) if allowed_sources is not None else None,
            "use_gap_fallback": use_gap_fallback,
        },
        feature_cols,
    )
    if rule_mode:
        metrics = {
            split: {
                "event_count": int(len(arrays[split][1])),
                "target_mean": float(np.mean(arrays[split][1])) if len(arrays[split][1]) else 0.0,
                "target_positive_rate": float((arrays[split][1] > 0).mean()) if len(arrays[split][1]) else 0.0,
            }
            for split in ["train", "val", "test"]
        }
        metrics["target_quantiles"] = {
            split: {str(q): float(np.quantile(arrays[split][1], q)) for q in [0.05, 0.25, 0.5, 0.75, 0.95]}
            for split in ["train", "val", "test"]
        }
    elif hybrid_mode:
        metrics = {
            split: {
                **{f"classifier_{k}": v for k, v in _metrics(model["classifier"], *arrays[split]).items()},
                **{f"regressor_{k}": v for k, v in _regression_metrics(model["regressor"], *reg_arrays[split]).items()},
            }
            for split in ["train", "val", "test"]
        }
        metrics["target_counts"] = {
            split: {str(k): int(v) for k, v in pd.Series(arrays[split][1]).value_counts().sort_index().items()}
            for split in ["train", "val", "test"]
        }
    elif regression_mode:
        metrics = {split: _regression_metrics(model, *arrays[split]) for split in ["train", "val", "test"]}
        metrics["target_quantiles"] = {
            split: {str(q): float(np.quantile(arrays[split][1], q)) for q in [0.05, 0.25, 0.5, 0.75, 0.95]}
            for split in ["train", "val", "test"]
        }
    else:
        metrics = {split: _metrics(model, *arrays[split]) for split in ["train", "val", "test"]}
        metrics["target_counts"] = {
            split: {str(k): int(v) for k, v in pd.Series(arrays[split][1]).value_counts().sort_index().items()}
            for split in ["train", "val", "test"]
        }
    return dataset, metrics


def _fixed_grid(level_mode: str, trade_side: str, model_name: str, profile: str) -> list[DecisionParams]:
    ema200_dist_windows = [(-999.0, 999.0)]
    ema50_slope_windows = [(-999.0, 999.0)]
    if profile == "ensemble_fvg" and model_name == "fvg":
        thresholds = [0.55, 0.60, 0.65]
        gaps = [0.03, 0.08]
        holds = [24, 48]
        fvg_size_filters = [0.0, 0.50, 0.75]
        max_trades_per_day_values = [0]
        min_predicted_net_pips_values = [-2.0, -1.0, 0.0, 1.0]
        min_edge_pips_values = [0.3]
        max_prediction_std_pips_values = [999.0]
        session_windows = [(-1, -1)]
        atr_windows = [(0.0, 999.0)]
        allowed_months_values = [""]
    elif profile == "pips_fvg" and model_name == "fvg":
        thresholds = [0.55]
        gaps = [0.03]
        holds = [24, 48]
        fvg_size_filters = [0.0, 0.50, 0.75]
        max_trades_per_day_values = [0]
        min_predicted_net_pips_values = [-5.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0]
        min_edge_pips_values = [0.3]
        max_prediction_std_pips_values = [999.0]
        session_windows = [(-1, -1)]
        atr_windows = [(0.0, 999.0)]
        allowed_months_values = [""]
    elif profile == "session_fvg" and model_name == "fvg":
        thresholds = [0.55]
        gaps = [0.03]
        holds = [24, 48]
        fvg_size_filters = [0.0, 0.50, 0.75]
        max_trades_per_day_values = [0]
        min_predicted_net_pips_values = [-2.0, -1.0, 0.0]
        min_edge_pips_values = [0.3]
        max_prediction_std_pips_values = [999.0]
        session_windows = [(-1, -1), (0, 6), (6, 12), (7, 12), (12, 17), (13, 17), (17, 23)]
        atr_windows = [(0.0, 999.0)]
        allowed_months_values = [""]
    elif profile == "stability_fvg" and model_name == "fvg":
        thresholds = [0.55]
        gaps = [0.03]
        holds = [24, 48]
        fvg_size_filters = [0.0, 0.50]
        max_trades_per_day_values = [0]
        min_predicted_net_pips_values = [-2.0, -1.0]
        min_edge_pips_values = [0.3]
        max_prediction_std_pips_values = [0.15, 0.30, 0.50, 999.0]
        session_windows = [(-1, -1), (6, 12), (12, 17)]
        atr_windows = [(0.0, 999.0)]
        allowed_months_values = [""]
    elif profile == "linear_fvg" and model_name == "fvg":
        thresholds = [0.55]
        gaps = [0.03]
        holds = [24, 48]
        fvg_size_filters = [0.0, 0.50, 0.75]
        max_trades_per_day_values = [0]
        min_predicted_net_pips_values = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0]
        min_edge_pips_values = [0.3]
        max_prediction_std_pips_values = [999.0]
        session_windows = [(-1, -1), (6, 12), (12, 17)]
        atr_windows = [(0.0, 999.0)]
        allowed_months_values = [""]
    elif profile == "regime_fvg" and model_name == "fvg":
        thresholds = [0.55]
        gaps = [0.03]
        holds = [48]
        fvg_size_filters = [0.0, 0.50]
        max_trades_per_day_values = [0]
        min_predicted_net_pips_values = [-2.0, -1.0]
        min_edge_pips_values = [0.3]
        max_prediction_std_pips_values = [999.0]
        session_windows = [(-1, -1), (12, 17)]
        atr_windows = [(0.0, 999.0), (0.0, 10.0), (10.0, 999.0)]
        allowed_months_values = [""]
    elif profile == "calendar_fvg" and model_name == "fvg":
        thresholds = [0.55]
        gaps = [0.03]
        holds = [48]
        fvg_size_filters = [0.0, 0.50]
        max_trades_per_day_values = [0]
        min_predicted_net_pips_values = [-2.0, -1.0]
        min_edge_pips_values = [0.3]
        max_prediction_std_pips_values = [999.0]
        session_windows = [(12, 17)]
        atr_windows = [(0.0, 999.0), (0.0, 10.0), (10.0, 999.0)]
        allowed_months_values = [
            "",
            "4,5,6,7,8,9",
            "4,5,6,7,8,9,10,11,12",
            "2,3,4,5,6,7,8,9,10,11",
            "6,7,8,9,10,11,12",
        ]
    elif profile == "trend_fvg" and model_name == "fvg":
        thresholds = [0.55]
        gaps = [0.03]
        holds = [48]
        fvg_size_filters = [0.0]
        max_trades_per_day_values = [0]
        min_predicted_net_pips_values = [-2.0, -1.0]
        min_edge_pips_values = [0.3]
        max_prediction_std_pips_values = [999.0]
        session_windows = [(12, 17)]
        atr_windows = [(0.0, 10.0)]
        allowed_months_values = ["", "2,3,4,5,6,7,8,9,10,11"]
        ema200_dist_windows = [
            (-999.0, 999.0),
            (0.0, 999.0),
            (-999.0, 0.0),
            (-50.0, 999.0),
            (-999.0, 50.0),
        ]
        ema50_slope_windows = [
            (-999.0, 999.0),
            (0.0, 999.0),
            (-999.0, 0.0),
        ]
    elif profile == "rule_fvg" and model_name == "fvg":
        thresholds = [0.55]
        gaps = [0.03]
        holds = [24, 48]
        fvg_size_filters = [0.0, 0.50, 0.75]
        max_trades_per_day_values = [0, 1, 2]
        min_predicted_net_pips_values = [-999.0]
        min_edge_pips_values = [0.3]
        max_prediction_std_pips_values = [999.0]
        session_windows = [(-1, -1), (6, 12), (12, 17)]
        atr_windows = [(0.0, 999.0), (0.0, 10.0), (10.0, 999.0)]
        allowed_months_values = [""]
    elif profile == "direction_diag_fvg" and model_name == "fvg":
        thresholds = [0.40, 0.45, 0.50, 0.55]
        gaps = [0.00, 0.02, 0.05]
        holds = [24, 48]
        fvg_size_filters = [0.0, 0.50]
        max_trades_per_day_values = [0]
        min_predicted_net_pips_values = [-999.0]
        min_edge_pips_values = [-2.0, 0.0, 0.3]
        max_prediction_std_pips_values = [999.0]
        session_windows = [(-1, -1)]
        atr_windows = [(0.0, 999.0)]
        allowed_months_values = [""]
    elif profile == "narrow_fvg" and model_name == "fvg":
        thresholds = [0.62, 0.64, 0.66, 0.68]
        gaps = [0.03, 0.08, 0.12]
        holds = [24, 36, 48]
        fvg_size_filters = [0.0]
        max_trades_per_day_values = [0, 1, 2]
        min_predicted_net_pips_values = [-999.0, -5.0, 0.0, 2.0, 5.0]
        min_edge_pips_values = [0.3]
        max_prediction_std_pips_values = [999.0]
        session_windows = [(-1, -1)]
        atr_windows = [(0.0, 999.0)]
        allowed_months_values = [""]
    else:
        thresholds = [0.60, 0.65, 0.70] if model_name == "h1" else [0.55, 0.60, 0.65]
        gaps = [0.03, 0.08]
        holds = [24, 46] if model_name == "h1" else [24, 48]
        fvg_size_filters = [0.0] if model_name != "fvg" else [0.0, 0.50, 0.75]
        max_trades_per_day_values = [0]
        min_predicted_net_pips_values = [-999.0] if model_name != "fvg" else [-999.0, -5.0, 0.0, 2.0, 5.0]
        min_edge_pips_values = [0.3]
        max_prediction_std_pips_values = [999.0]
        session_windows = [(-1, -1)]
        atr_windows = [(0.0, 999.0)]
        allowed_months_values = [""]
    out = []
    for threshold in thresholds:
        for gap in gaps:
            for hold in holds:
                for min_fvg_size_atr in fvg_size_filters:
                    for max_trades_per_day in max_trades_per_day_values:
                        for min_predicted_net_pips in min_predicted_net_pips_values:
                            for min_edge_pips in min_edge_pips_values:
                                for max_prediction_std_pips in max_prediction_std_pips_values:
                                    for session_start_hour, session_end_hour in session_windows:
                                        for min_signal_atr_pips, max_signal_atr_pips in atr_windows:
                                            for allowed_months in allowed_months_values:
                                                for min_ema200_dist_pips, max_ema200_dist_pips in ema200_dist_windows:
                                                    for min_ema50_slope_pips, max_ema50_slope_pips in ema50_slope_windows:
                                                        out.append(
                                                            DecisionParams(
                                                                level_mode=level_mode,
                                                                trade_side=trade_side,
                                                                entry_threshold=threshold,
                                                                exit_threshold=0.60,
                                                                exit_floor=0.35,
                                                                min_conf_gap=gap,
                                                                min_edge_pips=min_edge_pips,
                                                                stop_atr=1.0,
                                                                take_atr=1.0,
                                                                max_hold_bars=hold,
                                                                cooldown_bars=2,
                                                                risk_per_trade=0.01,
                                                                min_fvg_size_atr=min_fvg_size_atr,
                                                                min_signal_atr_pips=min_signal_atr_pips,
                                                                max_signal_atr_pips=max_signal_atr_pips,
                                                                max_trades_per_day=max_trades_per_day,
                                                                min_predicted_net_pips=min_predicted_net_pips,
                                                                max_prediction_std_pips=max_prediction_std_pips,
                                                                session_start_hour=session_start_hour,
                                                                session_end_hour=session_end_hour,
                                                                allowed_months=allowed_months,
                                                                min_ema200_dist_pips=min_ema200_dist_pips,
                                                                max_ema200_dist_pips=max_ema200_dist_pips,
                                                                min_ema50_slope_pips=min_ema50_slope_pips,
                                                                max_ema50_slope_pips=max_ema50_slope_pips,
                                                            )
                                                        )
    return out


def run_fixed_grid(
    dataset: SignalDataset,
    costs: CostConfig,
    risk: RiskConfig,
    optimizer_cfg: OptimizerConfig,
    level_mode: str,
    trade_side: str,
    fixed_grid_profile: str,
    selection_mode: str,
) -> tuple[pd.DataFrame, DecisionParams, BacktestResult, BacktestResult]:
    rows = []
    best_score = -float("inf")
    best_params = None
    best_val = None
    val_early_signals, val_late_signals = _split_signals_in_half(dataset.split_signals["val"])
    half_optimizer_cfg = OptimizerConfig(
        trials=optimizer_cfg.trials,
        seed=optimizer_cfg.seed,
        min_trades=max(5, int(math.ceil(optimizer_cfg.min_trades / 2))),
        max_acceptable_drawdown=optimizer_cfg.max_acceptable_drawdown,
    )
    for trial, params in enumerate(_fixed_grid(level_mode, trade_side, dataset.model_name, fixed_grid_profile)):
        val_result = simulate(dataset.split_market["val"], dataset.split_signals["val"], params, costs, risk)
        test_result = simulate(dataset.split_market["test"], dataset.split_signals["test"], params, costs, risk)
        val_early_result = _simulate_signal_subset(
            dataset.split_market["val"], val_early_signals, params, costs, risk
        )
        val_late_result = _simulate_signal_subset(
            dataset.split_market["val"], val_late_signals, params, costs, risk
        )
        standard_score = objective_score(val_result.metrics, optimizer_cfg)
        early_score = objective_score(val_early_result.metrics, half_optimizer_cfg)
        late_score = objective_score(val_late_result.metrics, half_optimizer_cfg)
        if selection_mode == "val_halves":
            score = min(early_score, late_score) + 0.25 * standard_score
            if val_early_result.metrics.get("net_profit", 0.0) <= 0 or val_late_result.metrics.get("net_profit", 0.0) <= 0:
                score -= 2.0
        else:
            score = standard_score
        rows.append(
            {
                "trial": trial,
                "score": score,
                "standard_score": standard_score,
                "val_early_score": early_score,
                "val_late_score": late_score,
                **params.to_dict(),
                **_flat_metrics(val_result.metrics),
                **{f"val_early_{k}": v for k, v in _flat_metrics(val_early_result.metrics).items()},
                **{f"val_late_{k}": v for k, v in _flat_metrics(val_late_result.metrics).items()},
                **{f"test_{k}": v for k, v in _flat_metrics(test_result.metrics).items()},
            }
        )
        if score > best_score:
            best_score = score
            best_params = params
            best_val = val_result
    if best_params is None or best_val is None:
        raise RuntimeError("Fixed grid had no trials.")
    test = simulate(dataset.split_market["test"], dataset.split_signals["test"], best_params, costs, risk)
    return pd.DataFrame(rows).sort_values("score", ascending=False), best_params, best_val, test


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run_strategy(
    name: str,
    dataset: SignalDataset,
    model_metrics: dict,
    out_dir: Path,
    args,
    trade_side: str,
    mode: str,
) -> dict:
    costs = CostConfig(
        spread_pips=args.spread_pips,
        slippage_pips_per_side=args.slippage_pips,
        commission_pips_per_side=args.commission_pips,
    )
    risk = RiskConfig(
        initial_equity=args.initial_equity,
        max_leverage=args.max_leverage,
        max_drawdown_stop=args.max_drawdown,
        daily_loss_stop=args.daily_loss_stop,
    )
    optimizer_cfg = OptimizerConfig(
        trials=args.trials,
        seed=args.seed,
        min_trades=args.min_trades,
        max_acceptable_drawdown=args.max_drawdown,
    )
    if mode == "fixed":
        trials, best_params, val_result, test_result = run_fixed_grid(
            dataset,
            costs,
            risk,
            optimizer_cfg,
            args.level_mode,
            trade_side,
            args.fixed_grid_profile,
            args.fixed_selection,
        )
    else:
        trials, best_params, val_result, test_result = run_optimization(
            dataset, optimizer_cfg, costs, risk, args.level_mode, trade_side
        )
    summary = _clean(
        {
            "name": name,
            "model": dataset.model_name,
            "run_id": dataset.run_id,
            "data_cfg": dataset.data_cfg,
            "model_metrics": model_metrics,
            "optimizer_mode": mode,
            "fixed_selection": args.fixed_selection if mode == "fixed" else None,
            "optimizer": asdict(optimizer_cfg),
            "costs": asdict(costs),
            "best_params": best_params.to_dict(),
            "validation_metrics": _flat_metrics(val_result.metrics),
            "test_metrics": _flat_metrics(test_result.metrics),
        }
    )
    write_backtest_outputs(out_dir / name, test_result, summary, trials=trials)
    write_backtest_outputs(out_dir / name / "validation_best", val_result, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EURUSD walk-forward bot experiments.")
    parser.add_argument("--output-dir", default="outputs/walkforward_runs")
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--initial-equity", type=float, default=10_000.0)
    parser.add_argument("--max-leverage", type=float, default=10.0)
    parser.add_argument("--max-drawdown", type=float, default=0.12)
    parser.add_argument("--daily-loss-stop", type=float, default=0.03)
    parser.add_argument("--spread-pips", type=float, default=1.0)
    parser.add_argument("--slippage-pips", type=float, default=0.2)
    parser.add_argument("--commission-pips", type=float, default=0.0)
    parser.add_argument("--level-mode", choices=["model", "atr"], default="model")
    parser.add_argument("--class-weight", action="store_true", default=True)
    parser.add_argument("--xgb-estimators", type=int, default=250)
    parser.add_argument("--folds", default="all", help="Comma-separated fold names, or all.")
    parser.add_argument("--profit-buffers", default=None, help="Comma-separated FVG profit buffers.")
    parser.add_argument(
        "--fvg-label-modes",
        default="profit",
        help="Comma-separated FVG label modes: profit,profit_pips,long_profit,long_pips,long_pips_bagged,long_pips_ridge,rule_long,first_break.",
    )
    parser.add_argument(
        "--fixed-grid-profile",
        choices=[
            "default",
            "narrow_fvg",
            "pips_fvg",
            "ensemble_fvg",
            "direction_diag_fvg",
            "session_fvg",
            "stability_fvg",
            "linear_fvg",
            "regime_fvg",
            "calendar_fvg",
            "trend_fvg",
            "rule_fvg",
        ],
        default="default",
    )
    parser.add_argument(
        "--fixed-selection",
        choices=["standard", "val_halves"],
        default="standard",
        help="How fixed-grid rows are selected from validation results.",
    )
    parser.add_argument(
        "--fvg-level-families",
        default="all",
        help=f"Comma-separated FVG level source presets: {','.join(FVG_LEVEL_FAMILIES)}",
    )
    parser.add_argument(
        "--fvg-min-fvg-atr",
        type=float,
        default=None,
        help="Override the FVG event minimum gap size in ATR units.",
    )
    parser.add_argument(
        "--fvg-require-unbroken-levels",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether FVG levels must remain unbroken before decision time.",
    )
    parser.add_argument(
        "--fvg-decision-delay-bars",
        type=int,
        default=None,
        help="Override FVG decision delay after event detection, in base-timeframe bars.",
    )
    parser.add_argument(
        "--fvg-base-timeframe",
        default="M15",
        help="Base timeframe for focused FVG experiments.",
    )
    parser.add_argument(
        "--fvg-higher-timeframe",
        default="H1",
        help="Higher timeframe for focused FVG experiments.",
    )
    parser.add_argument(
        "--fvg-single-timeframe",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use only the base timeframe for focused FVG experiments.",
    )
    parser.add_argument(
        "--fvg-trade-side",
        choices=["long", "short", "both"],
        default="long",
        help="Trade side for focused FVG experiments.",
    )
    parser.add_argument(
        "--fvg-focused",
        action="store_true",
        help="Run only FVG profit-label long fixed-grid experiments for the selected folds.",
    )
    parser.add_argument(
        "--fvg-diagnostic",
        action="store_true",
        help="Run only Fold B FVG profit-label fixed-grid level-family diagnostics.",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _parse_float_csv(value: str | None, default: list[float]) -> list[float]:
    if value is None:
        return default
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        return default
    return [float(item) for item in items]


def _select_folds(value: str) -> list[dict]:
    if value == "all":
        return FOLDS
    names = {item.strip() for item in value.split(",") if item.strip()}
    known = {fold["name"] for fold in FOLDS}
    unknown = sorted(names - known)
    if unknown:
        raise ValueError(f"Unknown --folds values: {unknown}")
    return [fold for fold in FOLDS if fold["name"] in names]


def _parse_label_modes(value: str) -> list[str]:
    modes = [item.strip() for item in value.split(",") if item.strip()]
    allowed = {
        "profit",
        "profit_pips",
        "long_profit",
        "long_pips",
        "long_pips_bagged",
        "long_pips_ridge",
        "rule_long",
        "first_break",
    }
    unknown = sorted(set(modes) - allowed)
    if unknown:
        raise ValueError(f"Unknown --fvg-label-modes values: {unknown}")
    return modes or ["profit"]


def main() -> None:
    args = parse_args()
    fvg_level_families = [item.strip() for item in args.fvg_level_families.split(",") if item.strip()]
    unknown_families = [item for item in fvg_level_families if item not in FVG_LEVEL_FAMILIES]
    if unknown_families:
        raise ValueError(f"Unknown --fvg-level-families values: {unknown_families}")
    fvg_label_modes = _parse_label_modes(args.fvg_label_modes)

    if args.fvg_focused:
        folds = _select_folds(args.folds)
        profit_buffers = _parse_float_csv(args.profit_buffers, [0.75, 1.0, 1.25])
    elif args.fvg_diagnostic:
        folds = FOLDS[-1:]
        profit_buffers = [0.5]
        fvg_level_families = fvg_level_families if fvg_level_families != ["all"] else list(FVG_LEVEL_FAMILIES)
    elif args.smoke:
        args.trials = min(args.trials, 8)
        args.xgb_estimators = min(args.xgb_estimators, 80)
        folds = _select_folds(args.folds) if args.folds != "all" else FOLDS[-1:]
        profit_buffers = _parse_float_csv(args.profit_buffers, [0.5])
    else:
        folds = _select_folds(args.folds)
        profit_buffers = _parse_float_csv(args.profit_buffers, [0.0, 0.5, 1.0])

    base_out = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out.mkdir(parents=True, exist_ok=True)

    summaries = []
    for fold in folds:
        fold_out = base_out / fold["name"]
        if args.fvg_focused or args.fvg_diagnostic:
            timeframe_suffix = (
                f"{args.fvg_base_timeframe}_single"
                if args.fvg_single_timeframe
                else f"{args.fvg_base_timeframe}_{args.fvg_higher_timeframe}"
            ).lower()
            for label_mode in fvg_label_modes:
                for level_family in fvg_level_families:
                    for buffer in profit_buffers:
                        fvg_profit, fvg_profit_metrics = build_fvg_dataset(
                            fold,
                            args.seed,
                            args.class_weight,
                            args.xgb_estimators,
                            label_mode=label_mode,
                            profit_buffer_pips=buffer,
                            base_timeframe=args.fvg_base_timeframe,
                            higher_timeframe=args.fvg_higher_timeframe,
                            level_family=level_family,
                            min_fvg_atr=args.fvg_min_fvg_atr,
                            require_unbroken_levels=args.fvg_require_unbroken_levels,
                            decision_delay_bars=args.fvg_decision_delay_bars,
                            single_timeframe=args.fvg_single_timeframe,
                        )
                        suffix = str(buffer).replace(".", "p")
                        summaries.append(
                            run_strategy(
                                f"{fold['name']}_fvg_{label_mode}_b{suffix}_{level_family}_{timeframe_suffix}_{args.fvg_trade_side}_fixed",
                                fvg_profit,
                                fvg_profit_metrics,
                                fold_out,
                                args,
                                trade_side=args.fvg_trade_side,
                                mode="fixed",
                            )
                        )
            continue

        h1_dataset, h1_metrics = build_h1_dataset(
            fold, args.seed, args.class_weight, args.xgb_estimators
        )
        summaries.append(
            run_strategy(
                f"{fold['name']}_h1_long_fixed",
                h1_dataset,
                h1_metrics,
                fold_out,
                args,
                trade_side="long",
                mode="fixed",
            )
        )
        summaries.append(
            run_strategy(
                f"{fold['name']}_h1_long_optimized",
                h1_dataset,
                h1_metrics,
                fold_out,
                args,
                trade_side="long",
                mode="optimized",
            )
        )

        fvg_first, fvg_first_metrics = build_fvg_dataset(
            fold,
            args.seed,
            args.class_weight,
            args.xgb_estimators,
            label_mode="first_break",
            profit_buffer_pips=0.0,
            base_timeframe="M15",
            higher_timeframe="H1",
            level_family="all",
            min_fvg_atr=args.fvg_min_fvg_atr,
            require_unbroken_levels=args.fvg_require_unbroken_levels,
            decision_delay_bars=args.fvg_decision_delay_bars,
            single_timeframe=args.fvg_single_timeframe,
        )
        summaries.append(
            run_strategy(
                f"{fold['name']}_fvg_first_break_long_fixed",
                fvg_first,
                fvg_first_metrics,
                fold_out,
                args,
                trade_side="long",
                mode="fixed",
            )
        )

        for buffer in profit_buffers:
            for level_family in fvg_level_families:
                fvg_profit, fvg_profit_metrics = build_fvg_dataset(
                    fold,
                    args.seed,
                    args.class_weight,
                    args.xgb_estimators,
                    label_mode="profit",
                    profit_buffer_pips=buffer,
                    base_timeframe="M15",
                    higher_timeframe="H1",
                    level_family=level_family,
                    min_fvg_atr=args.fvg_min_fvg_atr,
                    require_unbroken_levels=args.fvg_require_unbroken_levels,
                    decision_delay_bars=args.fvg_decision_delay_bars,
                    single_timeframe=args.fvg_single_timeframe,
                )
                suffix = str(buffer).replace(".", "p")
                family_suffix = "" if level_family == "all" else f"_{level_family}"
                summaries.append(
                    run_strategy(
                        f"{fold['name']}_fvg_profit_b{suffix}{family_suffix}_long_fixed",
                        fvg_profit,
                        fvg_profit_metrics,
                        fold_out,
                        args,
                        trade_side="long",
                        mode="fixed",
                    )
                )
                summaries.append(
                    run_strategy(
                        f"{fold['name']}_fvg_profit_b{suffix}{family_suffix}_long_optimized",
                        fvg_profit,
                        fvg_profit_metrics,
                        fold_out,
                        args,
                        trade_side="long",
                        mode="optimized",
                    )
                )

        if not args.smoke:
            fvg_h1, fvg_h1_metrics = build_fvg_dataset(
                fold,
                args.seed,
                args.class_weight,
                args.xgb_estimators,
                label_mode="profit",
                profit_buffer_pips=0.5,
                base_timeframe="H1",
                higher_timeframe="H4",
                level_family="all",
                min_fvg_atr=args.fvg_min_fvg_atr,
                require_unbroken_levels=args.fvg_require_unbroken_levels,
                decision_delay_bars=args.fvg_decision_delay_bars,
                single_timeframe=args.fvg_single_timeframe,
            )
            summaries.append(
                run_strategy(
                    f"{fold['name']}_fvg_h1_profit_b0p5_long_fixed",
                    fvg_h1,
                    fvg_h1_metrics,
                    fold_out,
                    args,
                    trade_side="long",
                    mode="fixed",
                )
            )

    (base_out / "summary.json").write_text(json.dumps(_clean(summaries), indent=2, allow_nan=False))
    rows = []
    for item in summaries:
        rows.append(
            {
                "name": item["name"],
                "model": item["model"],
                "optimizer_mode": item["optimizer_mode"],
                "val_net": item["validation_metrics"]["net_profit"],
                "test_net": item["test_metrics"]["net_profit"],
                "val_sharpe": item["validation_metrics"]["sharpe"],
                "test_sharpe": item["test_metrics"]["sharpe"],
                "val_trades": item["validation_metrics"]["trade_count"],
                "test_trades": item["test_metrics"]["trade_count"],
                "test_max_dd": item["test_metrics"]["max_drawdown"],
            }
        )
    pd.DataFrame(rows).to_csv(base_out / "summary.csv", index=False)
    print(f"Walk-forward outputs written to: {base_out}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
