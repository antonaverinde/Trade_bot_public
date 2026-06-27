from __future__ import annotations

import json
import ast
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from Pipeline.pipeline import ForexDataLoader, ForexPipeline
from strategy_fvg_fractals.pipeline import FVGFractalPipeline

from .config import FVG_DATA_CFG, H1_DATA_CFG
from .fvg_utils import fvg_level_candidates


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"


@dataclass
class SignalDataset:
    model_name: str
    run_id: str
    split_signals: dict[str, pd.DataFrame]
    split_market: dict[str, pd.DataFrame]
    raw_bars: pd.DataFrame
    data_cfg: dict
    feature_cols: list[str]


def _set_tracking() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)


def latest_successful_run_id(experiment_name: str) -> str:
    _set_tracking()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name}")
    runs = mlflow.search_runs(
        [experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise ValueError(f"No successful runs found for experiment: {experiment_name}")
    return str(runs.iloc[0]["run_id"])


def _load_model(run_id: str):
    _set_tracking()
    return mlflow.xgboost.load_model(f"runs:/{run_id}/model")


def _parse_param(value: str):
    if value in {"True", "False"}:
        return value == "True"
    if value in {"None", "null"}:
        return None
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _load_data_cfg(run_id: str, defaults: dict) -> dict:
    _set_tracking()
    run = mlflow.get_run(run_id)
    cfg = dict(defaults)
    for key in defaults:
        param_key = f"data_{key}"
        if param_key in run.data.params:
            cfg[key] = _parse_param(run.data.params[param_key])
    return cfg


def _load_feature_cols(run_id: str) -> list[str]:
    _set_tracking()
    client = MlflowClient(TRACKING_URI)
    try:
        path = client.download_artifacts(run_id, "features.json")
        payload = json.loads(Path(path).read_text())
        if isinstance(payload, dict) and "features" in payload:
            return list(payload["features"])
        if isinstance(payload, list):
            return list(payload)
    except Exception:
        pass

    path = client.download_artifacts(run_id, "features.txt")
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def _pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("JPY") else 0.0001


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _next_bar_times(raw_index: pd.DatetimeIndex, signal_index: pd.Index) -> pd.DatetimeIndex:
    positions = raw_index.searchsorted(pd.DatetimeIndex(signal_index), side="right")
    valid = positions < len(raw_index)
    return pd.DatetimeIndex(raw_index[positions[valid]]), valid


def _class_probabilities(model, X: np.ndarray, labels: list[int]) -> dict[int, np.ndarray]:
    proba = model.predict_proba(X)
    classes = [int(c) for c in getattr(model, "classes_", labels)]
    mapped = {}
    for label in labels:
        if label in classes:
            mapped[label] = proba[:, classes.index(label)]
        else:
            mapped[label] = np.zeros(len(X), dtype=float)
    return mapped


def _market_with_indicators(raw_bars: pd.DataFrame, pair: str) -> pd.DataFrame:
    raw = raw_bars.copy()
    raw["atr"] = _atr(raw)
    raw["sigma_pct"] = raw["close"].pct_change().ewm(span=100, adjust=False).std()
    raw["pip_size"] = _pip_size(pair)
    ema50 = raw["close"].ewm(span=50, adjust=False).mean()
    ema200 = raw["close"].ewm(span=200, adjust=False).mean()
    raw["ema200_dist_pips"] = (raw["close"] - ema200) / raw["pip_size"]
    raw["ema50_slope_pips"] = (ema50 - ema50.shift(20)) / raw["pip_size"]
    return raw[
        [
            "open",
            "high",
            "low",
            "close",
            "atr",
            "sigma_pct",
            "pip_size",
            "ema200_dist_pips",
            "ema50_slope_pips",
        ]
    ].dropna()


def _attach_valid_market_times(signals: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    return signals.sort_index().join(market[["open"]], how="inner").drop(columns=["open"])


def build_h1_signal_dataset(run_id: str | None = None) -> SignalDataset:
    run_id = run_id or latest_successful_run_id("xgboost_forex")
    model = _load_model(run_id)
    cfg = _load_data_cfg(run_id, H1_DATA_CFG)
    feature_cols = _load_feature_cols(run_id)

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
    raw_bars = results["raw_m1"]

    market_full = _market_with_indicators(raw_bars, cfg["pair"])
    split_signals: dict[str, pd.DataFrame] = {}
    split_market: dict[str, pd.DataFrame] = {}
    for split_name in ["val", "test"]:
        split = results[split_name]
        missing = [c for c in feature_cols if c not in split.columns]
        if missing:
            raise ValueError(f"{split_name} is missing model features: {missing[:8]}")
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
        frame = _attach_valid_market_times(frame, market_full)
        split_signals[split_name] = frame
        if frame.empty:
            split_market[split_name] = market_full.iloc[0:0].copy()
        else:
            split_market[split_name] = market_full.loc[frame.index.min():frame.index.max()].copy()

    return SignalDataset("h1", run_id, split_signals, split_market, raw_bars, cfg, feature_cols)


def _collapse_duplicate_fvg_signals(signals: pd.DataFrame) -> pd.DataFrame:
    if not signals.index.has_duplicates:
        return signals.sort_index()

    rows = []
    for entry_time, group in signals.groupby(level=0, sort=True):
        score = group[["p_long", "p_short"]].max(axis=1)
        best = group.loc[score.idxmax()].copy()
        best["p_hold"] = max(0.0, 1.0 - max(float(best["p_long"]), float(best["p_short"])))
        best["entry_time"] = entry_time
        rows.append(best.to_dict())
    return pd.DataFrame(rows).set_index("entry_time").sort_index()


def build_fvg_signal_dataset(run_id: str | None = None) -> SignalDataset:
    run_id = run_id or latest_successful_run_id("xgboost_fvg_fractals")
    model = _load_model(run_id)
    cfg = _load_data_cfg(run_id, FVG_DATA_CFG)
    feature_cols = _load_feature_cols(run_id)

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
    raw_bars = results["raw_base"]

    market_full = _market_with_indicators(raw_bars, cfg["pair"])
    split_signals: dict[str, pd.DataFrame] = {}
    split_market: dict[str, pd.DataFrame] = {}
    for split_name in ["val", "test"]:
        split = results[split_name]
        tradable = split.copy()
        missing = [c for c in feature_cols if c not in tradable.columns]
        if missing:
            raise ValueError(f"{split_name} is missing model features: {missing[:8]}")
        X = tradable[feature_cols].to_numpy()
        probs = _class_probabilities(model, X, labels=[0, 1, 2])
        if set(int(c) for c in getattr(model, "classes_", [0, 1])) == {0, 1}:
            p_short = probs[0]
            p_hold = np.zeros(len(tradable), dtype=float)
            p_long = probs[1]
        else:
            p_short = probs[0]
            p_hold = probs[1]
            p_long = probs[2]
        long_take = []
        long_stop = []
        short_take = []
        short_stop = []
        long_take_source = []
        long_stop_source = []
        short_take_source = []
        short_stop_source = []
        fvg_size_pips = []
        fvg_size_atr = []
        signal_atr = []
        for _, row in tradable.iterrows():
            ref = float(row["decision_close"])
            levels = fvg_level_candidates(row, ref)
            high = levels["high"]
            low = levels["low"]
            long_take.append(high.price if high is not None else np.nan)
            long_stop.append(low.price if low is not None else np.nan)
            short_take.append(low.price if low is not None else np.nan)
            short_stop.append(high.price if high is not None else np.nan)
            long_take_source.append(high.source if high is not None else "")
            long_stop_source.append(low.source if low is not None else "")
            short_take_source.append(low.source if low is not None else "")
            short_stop_source.append(high.source if high is not None else "")
            pip = _pip_size(cfg["pair"])
            size = abs(float(row["fvg_gap_high"]) - float(row["fvg_gap_low"]))
            atr = float(row["atr"])
            fvg_size_pips.append(size / pip)
            fvg_size_atr.append(size / atr if np.isfinite(atr) and atr > 0 else np.nan)
            signal_atr.append(atr)
        frame = pd.DataFrame(
            {
                "p_short": p_short,
                "p_hold": p_hold,
                "p_long": p_long,
                "source_time": pd.DatetimeIndex(tradable.index),
                "long_take_price": long_take,
                "long_stop_price": long_stop,
                "short_take_price": short_take,
                "short_stop_price": short_stop,
                "level_basis": "fvg_known_fractal_levels",
                "long_take_source": long_take_source,
                "long_stop_source": long_stop_source,
                "short_take_source": short_take_source,
                "short_stop_source": short_stop_source,
                "fvg_size_pips": fvg_size_pips,
                "fvg_size_atr": fvg_size_atr,
                "signal_atr": signal_atr,
            },
            index=pd.DatetimeIndex(tradable["decision_time"]),
        )
        frame = _collapse_duplicate_fvg_signals(frame)
        frame = _attach_valid_market_times(frame, market_full)
        split_signals[split_name] = frame
        if frame.empty:
            split_market[split_name] = market_full.iloc[0:0].copy()
        else:
            split_market[split_name] = market_full.loc[frame.index.min():frame.index.max()].copy()

    return SignalDataset("fvg", run_id, split_signals, split_market, raw_bars, cfg, feature_cols)


def build_signal_dataset(model_name: str, run_id: str | None = None) -> SignalDataset:
    if model_name == "h1":
        return build_h1_signal_dataset(run_id=run_id)
    if model_name == "fvg":
        return build_fvg_signal_dataset(run_id=run_id)
    raise ValueError("model_name must be 'h1' or 'fvg'")
