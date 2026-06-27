from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from Pipeline.pipeline import ForexDataLoader

from .config import CostConfig, DecisionParams, OptimizerConfig, RiskConfig
from .model_adapters import _market_with_indicators, _next_bar_times
from .optimize import _flat_metrics, objective_score
from .reports import write_backtest_outputs
from .simulator import simulate
from .walkforward import FOLDS, PROJECT_ROOT, _select_folds


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


def _resample_ohlc(raw: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if timeframe.upper() in {"M1", "1T", "1MIN"}:
        return raw.copy()
    rule = timeframe.upper()
    if rule.startswith("M"):
        rule = f"{int(rule[1:])}min"
    elif rule.startswith("H"):
        rule = f"{int(rule[1:])}h"
    out = raw.resample(rule, label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )
    return out.dropna()


def _slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    idx = pd.DatetimeIndex(df.index)
    return df.loc[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))].copy()


def _load_market(years: list[int], timeframe: str, pair: str) -> pd.DataFrame:
    loader = ForexDataLoader()
    raw = loader.load_and_merge(
        str(PROJECT_ROOT / "histdata"),
        pair=pair,
        years=years,
        weekends="nogap",
    )
    bars = _resample_ohlc(raw, timeframe)
    return _market_with_indicators(bars, pair)


def _add_signal_features(market: pd.DataFrame) -> pd.DataFrame:
    frame = market.copy()
    close = frame["close"]
    frame["ema20"] = close.ewm(span=20, adjust=False).mean()
    frame["ema50"] = close.ewm(span=50, adjust=False).mean()
    frame["ema200"] = close.ewm(span=200, adjust=False).mean()
    frame["ema20_slope_pips"] = (frame["ema20"] - frame["ema20"].shift(10)) / frame["pip_size"]
    frame["ema50_slope_pips"] = (frame["ema50"] - frame["ema50"].shift(20)) / frame["pip_size"]
    frame["ema200_dist_pips"] = (close - frame["ema200"]) / frame["pip_size"]
    rolling_mean = close.rolling(96, min_periods=48).mean()
    rolling_std = close.rolling(96, min_periods=48).std(ddof=0)
    frame["z96"] = (close - rolling_mean) / rolling_std.replace(0.0, np.nan)
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0.0, np.nan)
    frame["rsi14"] = 100.0 - (100.0 / (1.0 + rs))
    frame["atr_pips"] = frame["atr"] / frame["pip_size"]
    return frame.dropna()


def _signals_from_rule(
    market: pd.DataFrame,
    rule: str,
    side: str,
    slope_threshold: float,
    ema_dist_threshold: float,
    z_threshold: float,
    atr_min_pips: float,
    atr_max_pips: float,
) -> pd.DataFrame:
    features = _add_signal_features(market)
    atr_ok = (features["atr_pips"] >= atr_min_pips) & (features["atr_pips"] <= atr_max_pips)

    if rule in {"trend", "trend_fade"}:
        trend_long_mask = (
            atr_ok
            & (features["ema200_dist_pips"] >= ema_dist_threshold)
            & (features["ema50_slope_pips"] >= slope_threshold)
            & (features["ema20"] >= features["ema50"])
        )
        trend_short_mask = (
            atr_ok
            & (features["ema200_dist_pips"] <= -ema_dist_threshold)
            & (features["ema50_slope_pips"] <= -slope_threshold)
            & (features["ema20"] <= features["ema50"])
        )
        if rule == "trend_fade":
            long_mask = trend_short_mask
            short_mask = trend_long_mask
        else:
            long_mask = trend_long_mask
            short_mask = trend_short_mask
    elif rule == "mean_revert":
        long_mask = (
            atr_ok
            & (features["z96"] <= -z_threshold)
            & (features["ema200_dist_pips"] >= -ema_dist_threshold)
        )
        short_mask = (
            atr_ok
            & (features["z96"] >= z_threshold)
            & (features["ema200_dist_pips"] <= ema_dist_threshold)
        )
    elif rule == "rsi_revert":
        dist_ok = features["ema200_dist_pips"].abs() <= ema_dist_threshold
        long_mask = atr_ok & dist_ok & (features["rsi14"] <= z_threshold)
        short_mask = atr_ok & dist_ok & (features["rsi14"] >= 100.0 - z_threshold)
    else:
        raise ValueError(f"Unknown rule: {rule}")

    if side == "long":
        short_mask &= False
    elif side == "short":
        long_mask &= False
    elif side != "both":
        raise ValueError(f"Unknown side: {side}")

    rows = []
    for ts in features.index[long_mask | short_mask]:
        is_long = bool(long_mask.loc[ts])
        is_short = bool(short_mask.loc[ts])
        if is_long == is_short:
            continue
        rows.append(
            {
                "entry_time": ts,
                "p_long": 1.0 if is_long else 0.0,
                "p_short": 1.0 if is_short else 0.0,
                "p_hold": 0.0,
                "source_time": ts,
                "source_ema200_dist_pips": float(features.loc[ts, "ema200_dist_pips"]),
                "source_ema50_slope_pips": float(features.loc[ts, "ema50_slope_pips"]),
                "source_atr_pips": float(features.loc[ts, "atr_pips"]),
                "source_z96": float(features.loc[ts, "z96"]),
                "source_rsi14": float(features.loc[ts, "rsi14"]),
                "level_basis": f"baseline_{rule}",
            }
    )
    if not rows:
        return pd.DataFrame(
            columns=[
                "p_long",
                "p_short",
                "p_hold",
                "source_time",
                "source_ema200_dist_pips",
                "source_ema50_slope_pips",
                "source_atr_pips",
                "source_z96",
                "source_rsi14",
                "level_basis",
            ]
        )
    return pd.DataFrame(rows).set_index("entry_time").sort_index()


def _delay_signals_to_next_bar(signals: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return signals.copy()
    source_times = pd.DatetimeIndex(signals.index)
    entry_times, valid = _next_bar_times(pd.DatetimeIndex(market.index), source_times)
    delayed = signals.iloc[np.flatnonzero(valid)].copy()
    delayed["source_time"] = source_times[valid]
    source_market = market.reindex(source_times[valid])
    delayed["signal_atr"] = source_market["atr"].to_numpy()
    delayed.index = entry_times
    delayed.index.name = "entry_time"
    return delayed.sort_index()


def _period_summary(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "positive_quarters": 0,
            "quarter_count": 0,
            "min_quarter_pnl": 0.0,
        }
    frame = trades.copy()
    frame["quarter"] = pd.PeriodIndex(pd.DatetimeIndex(frame["entry_time"]), freq="Q").astype(str)
    pnl = frame.groupby("quarter")["pnl"].sum()
    return {
        "positive_quarters": int((pnl > 0).sum()),
        "quarter_count": int(len(pnl)),
        "min_quarter_pnl": float(pnl.min()) if len(pnl) else 0.0,
    }


def _selected_by_fold_summary(trials: pd.DataFrame, optimizer_cfg: OptimizerConfig) -> pd.DataFrame:
    if trials.empty:
        return pd.DataFrame()

    rows = []
    for fold, group in trials.groupby("fold", sort=True):
        ranked = group.sort_values("score", ascending=False)
        eligible = ranked[
            (ranked["val_net_profit"] > 0.0)
            & (ranked["val_trade_count"] >= optimizer_cfg.min_trades)
            & (ranked["val_max_drawdown"] <= optimizer_cfg.max_acceptable_drawdown)
        ]
        selected = eligible.head(1)
        if selected.empty:
            selected = ranked.head(1).copy()
            selected["selection_status"] = "no_eligible_validation_row"
        else:
            selected = selected.copy()
            selected["selection_status"] = "eligible_validation_row"
        rows.append(selected)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values("fold")


def _selection_metrics(selected: pd.DataFrame) -> dict:
    if selected.empty:
        return {}
    return {
        "folds": int(selected["fold"].nunique()),
        "eligible_folds": int((selected["selection_status"] == "eligible_validation_row").sum()),
        "min_validation_net": float(selected["val_net_profit"].min()),
        "min_test_net": float(selected["test_net_profit"].min()),
        "validation_net_sum": float(selected["val_net_profit"].sum()),
        "test_net_sum": float(selected["test_net_profit"].sum()),
        "min_validation_trades": float(selected["val_trade_count"].min()),
        "min_test_trades": float(selected["test_trade_count"].min()),
        "max_validation_drawdown": float(selected["val_max_drawdown"].max()),
        "max_test_drawdown": float(selected["test_max_drawdown"].max()),
        "min_test_quarter_pnl": float(selected["test_min_quarter_pnl"].min())
        if "test_min_quarter_pnl" in selected.columns
        else 0.0,
    }


def _param_id(row: dict) -> str:
    gate_suffix = ""
    if any(
        key in row
        for key in [
            "min_ema200_dist_pips",
            "max_ema200_dist_pips",
            "min_ema50_slope_pips",
            "max_ema50_slope_pips",
        ]
    ):
        gate_suffix = (
            f"_ema200{row.get('min_ema200_dist_pips', -999):g}-{row.get('max_ema200_dist_pips', 999):g}"
            f"_slope50{row.get('min_ema50_slope_pips', -999):g}-{row.get('max_ema50_slope_pips', 999):g}"
        )
    excluded_suffix = ""
    if row.get("excluded_hours", ""):
        excluded_suffix = f"_nohour{str(row['excluded_hours']).replace(',', '-')}"
    month_suffix = ""
    if row.get("allowed_months", ""):
        month_suffix = f"_months{str(row['allowed_months']).replace(',', '-')}"
    weekday_suffix = ""
    if row.get("excluded_weekdays", ""):
        weekday_suffix = f"_nowday{str(row['excluded_weekdays']).replace(',', '-')}"
    return (
        f"{row['rule']}_{row['timeframe']}_{row['side']}_"
        f"slope{row['slope_threshold']:g}_dist{row['ema_dist_threshold']:g}_"
        f"z{row['z_threshold']:g}_atr{row['atr_min_pips']:g}-{row['atr_max_pips']:g}_"
        f"hold{row['max_hold_bars']}_stop{row['stop_atr']:g}_take{row['take_atr']:g}_"
        f"sess{row['session_start_hour']}-{row['session_end_hour']}_"
        f"roll{row['rolling_pnl_window']}_{row['min_rolling_pnl']:g}"
        f"{gate_suffix}"
        f"{excluded_suffix}"
        f"{month_suffix}"
        f"{weekday_suffix}"
    )


def _grid(profile: str) -> list[dict]:
    excluded_weekdays_options = [""]
    if profile == "fade":
        timeframes = ["H1"]
        rules = ["trend_fade"]
        sides = ["long", "short", "both"]
        slope_thresholds = [0.0, 2.0]
        ema_dist_thresholds = [0.0, 20.0]
        z_thresholds = [1.0]
        atr_windows = [(0.0, 999.0)]
        holds = [8, 16]
        stop_take = [(1.0, 1.0), (1.0, 1.5), (1.5, 1.0)]
        sessions = [(7, 17)]
        rolling_windows = [0]
        min_rolling_pnls = [-999999.0]
    elif profile == "rsi":
        timeframes = ["M15"]
        rules = ["rsi_revert"]
        sides = ["both"]
        slope_thresholds = [0.0]
        ema_dist_thresholds = [999.0]
        z_thresholds = [25.0, 30.0]
        atr_windows = [(0.0, 999.0)]
        holds = [8, 16]
        stop_take = [(1.0, 1.0), (1.0, 1.5)]
        sessions = [(7, 17)]
        rolling_windows = [0]
        min_rolling_pnls = [-999999.0]
    elif profile == "adaptive":
        timeframes = ["H1"]
        rules = ["trend"]
        sides = ["short"]
        slope_thresholds = [0.0, 2.0]
        ema_dist_thresholds = [20.0]
        z_thresholds = [1.0]
        atr_windows = [(0.0, 999.0)]
        holds = [8, 16]
        stop_take = [(1.0, 1.5)]
        sessions = [(7, 17)]
        rolling_windows = [0, 10, 25, 50, 100]
        min_rolling_pnls = [0.0, -100.0, -250.0]
    elif profile == "mr_long":
        timeframes = ["H1"]
        rules = ["mean_revert"]
        sides = ["long"]
        slope_thresholds = [0.0]
        ema_dist_thresholds = [20.0, 50.0]
        z_thresholds = [0.75, 1.0, 1.25]
        atr_windows = [(0.0, 999.0), (0.0, 12.0)]
        holds = [16, 24]
        stop_take = [(1.0, 1.5), (1.0, 2.0), (1.5, 2.0)]
        sessions = [(-1, -1), (0, 7), (7, 12), (7, 17), (12, 17)]
        rolling_windows = [0]
        min_rolling_pnls = [-999999.0]
    elif profile == "mr_long_refine":
        timeframes = ["H1"]
        rules = ["mean_revert"]
        sides = ["long"]
        slope_thresholds = [0.0]
        ema_dist_thresholds = [20.0]
        z_thresholds = [0.9, 1.0, 1.1, 1.25]
        atr_windows = [(0.0, 999.0)]
        holds = [12, 16, 20]
        stop_take = [(1.25, 2.0), (1.5, 2.0), (1.5, 2.25), (1.75, 2.25)]
        sessions = [(7, 12), (7, 15), (7, 17), (8, 17), (12, 17)]
        rolling_windows = [0, 5, 10, 20]
        min_rolling_pnls = [-250.0, 0.0, 150.0]
        ema200_gates = [(-999.0, 999.0)]
        ema50_slope_gates = [(-999.0, 999.0)]
        excluded_hours_options = [""]
        allowed_months_options = [""]
    elif profile == "mr_long_source_gate":
        timeframes = ["H1"]
        rules = ["mean_revert"]
        sides = ["long"]
        slope_thresholds = [0.0]
        ema_dist_thresholds = [20.0]
        z_thresholds = [1.0]
        atr_windows = [(0.0, 999.0)]
        holds = [12]
        stop_take = [(1.5, 2.25)]
        sessions = [(7, 15)]
        rolling_windows = [0]
        min_rolling_pnls = [-999999.0]
        ema200_gates = [(-999.0, 40.0), (-20.0, 40.0), (-999.0, 80.0), (-999.0, 999.0)]
        ema50_slope_gates = [(-999.0, 999.0), (-999.0, 20.0)]
        excluded_hours_options = ["", "10"]
        allowed_months_options = [""]
    elif profile == "mr_long_stable_fixed":
        timeframes = ["H1"]
        rules = ["mean_revert"]
        sides = ["long"]
        slope_thresholds = [0.0]
        ema_dist_thresholds = [20.0]
        z_thresholds = [1.0]
        atr_windows = [(0.0, 999.0)]
        holds = [12]
        stop_take = [(1.5, 2.25)]
        sessions = [(7, 15)]
        rolling_windows = [0]
        min_rolling_pnls = [-999999.0]
        ema200_gates = [(-999.0, 80.0)]
        ema50_slope_gates = [(-999.0, 999.0)]
        excluded_hours_options = ["11"]
        allowed_months_options = ["1,2,3,4,5,6,7,8,10,11,12"]
        excluded_weekdays_options = [""]
    elif profile == "mr_long_weekday_stable":
        timeframes = ["H1"]
        rules = ["mean_revert"]
        sides = ["long"]
        slope_thresholds = [0.0]
        ema_dist_thresholds = [20.0]
        z_thresholds = [1.0]
        atr_windows = [(0.0, 999.0)]
        holds = [14]
        stop_take = [(1.5, 2.25)]
        sessions = [(7, 15)]
        rolling_windows = [0]
        min_rolling_pnls = [-999999.0]
        ema200_gates = [(-999.0, 80.0)]
        ema50_slope_gates = [(-999.0, 999.0)]
        excluded_hours_options = ["11"]
        allowed_months_options = ["1,2,3,4,5,6,7,8,10,11,12"]
        excluded_weekdays_options = ["1"]
    elif profile == "quick":
        timeframes = ["H1"]
        rules = ["trend", "mean_revert"]
        sides = ["long", "short"]
        slope_thresholds = [0.0, 2.0]
        ema_dist_thresholds = [0.0, 20.0]
        z_thresholds = [1.0, 1.5]
        atr_windows = [(0.0, 999.0), (2.0, 12.0)]
        holds = [8, 16]
        stop_take = [(1.0, 1.0), (1.0, 1.5)]
        sessions = [(7, 17)]
        rolling_windows = [0]
        min_rolling_pnls = [-999999.0]
        ema200_gates = [(-999.0, 999.0)]
        ema50_slope_gates = [(-999.0, 999.0)]
        excluded_hours_options = [""]
        allowed_months_options = [""]
        excluded_weekdays_options = [""]
    else:
        timeframes = ["M15", "H1"]
        rules = ["trend", "mean_revert"]
        sides = ["long", "short", "both"]
        slope_thresholds = [0.0, 2.0, 5.0]
        ema_dist_thresholds = [0.0, 20.0, 50.0]
        z_thresholds = [1.0, 1.5, 2.0]
        atr_windows = [(0.0, 999.0), (2.0, 12.0), (5.0, 20.0)]
        holds = [8, 16, 32]
        stop_take = [(1.0, 1.0), (1.0, 1.5), (1.5, 1.0)]
        sessions = [(-1, -1), (7, 17), (12, 17)]
        rolling_windows = [0]
        min_rolling_pnls = [-999999.0]
        ema200_gates = [(-999.0, 999.0)]
        ema50_slope_gates = [(-999.0, 999.0)]
        excluded_hours_options = [""]
        allowed_months_options = [""]
        excluded_weekdays_options = [""]

    if profile in {"fade", "rsi", "adaptive", "mr_long"}:
        ema200_gates = [(-999.0, 999.0)]
        ema50_slope_gates = [(-999.0, 999.0)]
        excluded_hours_options = [""]
        allowed_months_options = [""]
        excluded_weekdays_options = [""]

    rows = []
    for timeframe in timeframes:
        for rule in rules:
            for side in sides:
                for slope_threshold in slope_thresholds:
                    for ema_dist_threshold in ema_dist_thresholds:
                        for z_threshold in z_thresholds:
                            for atr_min_pips, atr_max_pips in atr_windows:
                                for max_hold_bars in holds:
                                    for stop_atr, take_atr in stop_take:
                                        for session_start_hour, session_end_hour in sessions:
                                            for rolling_pnl_window in rolling_windows:
                                                pnl_thresholds = (
                                                    [-999999.0]
                                                    if rolling_pnl_window <= 0
                                                    else min_rolling_pnls
                                                )
                                                for min_rolling_pnl in pnl_thresholds:
                                                    for excluded_hours in excluded_hours_options:
                                                        for allowed_months in allowed_months_options:
                                                            for excluded_weekdays in excluded_weekdays_options:
                                                                for min_ema200_dist_pips, max_ema200_dist_pips in ema200_gates:
                                                                    for min_ema50_slope_pips, max_ema50_slope_pips in ema50_slope_gates:
                                                                        rows.append(
                                                                            {
                                                                                "timeframe": timeframe,
                                                                                "rule": rule,
                                                                                "side": side,
                                                                                "slope_threshold": slope_threshold,
                                                                                "ema_dist_threshold": ema_dist_threshold,
                                                                                "z_threshold": z_threshold,
                                                                                "atr_min_pips": atr_min_pips,
                                                                                "atr_max_pips": atr_max_pips,
                                                                                "max_hold_bars": max_hold_bars,
                                                                                "stop_atr": stop_atr,
                                                                                "take_atr": take_atr,
                                                                                "session_start_hour": session_start_hour,
                                                                                "session_end_hour": session_end_hour,
                                                                                "excluded_hours": excluded_hours,
                                                                                "allowed_months": allowed_months,
                                                                                "excluded_weekdays": excluded_weekdays,
                                                                                "rolling_pnl_window": rolling_pnl_window,
                                                                                "min_rolling_pnl": min_rolling_pnl,
                                                                                "min_ema200_dist_pips": min_ema200_dist_pips,
                                                                                "max_ema200_dist_pips": max_ema200_dist_pips,
                                                                                "min_ema50_slope_pips": min_ema50_slope_pips,
                                                                                "max_ema50_slope_pips": max_ema50_slope_pips,
                                                                            }
                                                                        )
    return rows


def run_baselines(args: argparse.Namespace) -> Path:
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

    out_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = _select_folds(args.folds)
    grid = _grid(args.profile)
    if args.row_start > 0 or args.row_end > 0:
        start = max(0, args.row_start)
        end = args.row_end if args.row_end > 0 else len(grid)
        grid = grid[start:end]
    if args.max_rows > 0:
        grid = grid[: args.max_rows]

    market_cache: dict[tuple, pd.DataFrame] = {}
    summaries = []
    trials = []
    best_by_fold: dict[str, tuple[float, dict, object, object]] = {}

    for fold in folds:
        for row_i, row in enumerate(grid):
            cache_key = (tuple(fold["years"]), row["timeframe"], args.pair)
            if cache_key not in market_cache:
                market_cache[cache_key] = _load_market(fold["years"], row["timeframe"], args.pair)
            market = market_cache[cache_key]
            signals = _signals_from_rule(
                market,
                row["rule"],
                row["side"],
                row["slope_threshold"],
                row["ema_dist_threshold"],
                row["z_threshold"],
                row["atr_min_pips"],
                row["atr_max_pips"],
            )
            signals = _delay_signals_to_next_bar(signals, market)
            params = DecisionParams(
                level_mode="atr",
                trade_side=row["side"],
                entry_threshold=0.55,
                exit_threshold=1.1,
                exit_floor=-0.1,
                min_conf_gap=0.0,
                min_edge_pips=-999.0,
                stop_atr=row["stop_atr"],
                take_atr=row["take_atr"],
                max_hold_bars=row["max_hold_bars"],
                cooldown_bars=1,
                risk_per_trade=args.risk_per_trade,
                min_signal_atr_pips=row["atr_min_pips"],
                max_signal_atr_pips=row["atr_max_pips"],
                session_start_hour=row["session_start_hour"],
                session_end_hour=row["session_end_hour"],
                excluded_hours=row.get("excluded_hours", ""),
                allowed_months=row.get("allowed_months", ""),
                excluded_weekdays=row.get("excluded_weekdays", ""),
                rolling_pnl_window=row["rolling_pnl_window"],
                min_rolling_pnl=row["min_rolling_pnl"],
                min_ema200_dist_pips=row.get("min_ema200_dist_pips", -999.0),
                max_ema200_dist_pips=row.get("max_ema200_dist_pips", 999.0),
                min_ema50_slope_pips=row.get("min_ema50_slope_pips", -999.0),
                max_ema50_slope_pips=row.get("max_ema50_slope_pips", 999.0),
            )
            split_results = {}
            for split in ["val", "test"]:
                start = fold[f"{split}_start"]
                end = fold[f"{split}_end"]
                split_market = _slice(market, start, end)
                split_signals = _slice(signals, start, end)
                split_results[split] = simulate(split_market, split_signals, params, costs, risk)

            val_metrics = _flat_metrics(split_results["val"].metrics)
            test_metrics = _flat_metrics(split_results["test"].metrics)
            score = objective_score(split_results["val"].metrics, optimizer_cfg)
            param_id = _param_id(row)
            trial = {
                "fold": fold["name"],
                "row": row_i,
                "param_id": param_id,
                "score": score,
                **row,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
                **{f"val_{k}": v for k, v in _period_summary(split_results["val"].trades).items()},
                **{f"test_{k}": v for k, v in _period_summary(split_results["test"].trades).items()},
            }
            trials.append(trial)
            current_best = best_by_fold.get(fold["name"])
            if current_best is None or score > current_best[0]:
                best_by_fold[fold["name"]] = (score, row, split_results["val"], split_results["test"])

        if fold["name"] in best_by_fold:
            _, best_row, best_val, best_test = best_by_fold[fold["name"]]
            param_id = _param_id(best_row)
            summary = _clean(
                {
                    "fold": fold["name"],
                    "param_id": param_id,
                    "params": best_row,
                    "validation_metrics": _flat_metrics(best_val.metrics),
                    "test_metrics": _flat_metrics(best_test.metrics),
                    "costs": asdict(costs),
                    "risk": asdict(risk),
                    "optimizer": asdict(optimizer_cfg),
                }
            )
            fold_out = out_dir / fold["name"] / param_id
            write_backtest_outputs(fold_out, best_test, summary, trials=pd.DataFrame(trials))
            write_backtest_outputs(fold_out / "validation_best", best_val, summary)
            summaries.append(summary)

    trials_df = pd.DataFrame(trials)
    if not trials_df.empty:
        trials_df = trials_df.sort_values(["fold", "score"], ascending=[True, False])
    trials_df.to_csv(out_dir / "trials.csv", index=False)
    if not trials_df.empty:
        aggregate = (
            trials_df.groupby("param_id")
            .agg(
                folds=("fold", "nunique"),
                val_net_min=("val_net_profit", "min"),
                test_net_min=("test_net_profit", "min"),
                val_net_sum=("val_net_profit", "sum"),
                test_net_sum=("test_net_profit", "sum"),
                val_trades_min=("val_trade_count", "min"),
                test_trades_min=("test_trade_count", "min"),
                val_drawdown_max=("val_max_drawdown", "max"),
                test_drawdown_max=("test_max_drawdown", "max"),
                val_positive_quarters_sum=("val_positive_quarters", "sum"),
                test_positive_quarters_sum=("test_positive_quarters", "sum"),
                test_min_quarter_pnl=("test_min_quarter_pnl", "min"),
            )
            .reset_index()
            .sort_values(
                ["test_net_min", "val_net_min", "test_drawdown_max"],
                ascending=[False, False, True],
            )
        )
        aggregate.to_csv(out_dir / "exact_params_summary.csv", index=False)
        selected = _selected_by_fold_summary(trials_df, optimizer_cfg)
        selected.to_csv(out_dir / "selected_by_fold.csv", index=False)
        (out_dir / "selection_summary.json").write_text(
            json.dumps(_clean(_selection_metrics(selected)), indent=2, allow_nan=False)
        )
    (out_dir / "summary.json").write_text(json.dumps(_clean(summaries), indent=2, allow_nan=False))

    if trials_df.empty:
        print(f"Baseline outputs written to: {out_dir}")
        print("No trial rows produced.")
        return out_dir

    print(f"Baseline outputs written to: {out_dir}")
    display = trials_df[
        [
            "fold",
            "param_id",
            "val_net_profit",
            "test_net_profit",
            "val_trade_count",
            "test_trade_count",
            "val_max_drawdown",
            "test_max_drawdown",
            "score",
        ]
    ].head(args.print_top)
    print(display.to_string(index=False))
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simple raw-price baseline strategies.")
    parser.add_argument("--output-dir", default="outputs/baseline_runs")
    parser.add_argument("--folds", default="all")
    parser.add_argument("--pair", default="EURUSD")
    parser.add_argument("--profile", choices=["quick", "full", "adaptive", "fade", "rsi", "mr_long", "mr_long_refine", "mr_long_source_gate", "mr_long_stable_fixed", "mr_long_weekday_stable"], default="quick")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--row-end", type=int, default=0)
    parser.add_argument("--print-top", type=int, default=20)
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--initial-equity", type=float, default=10_000.0)
    parser.add_argument("--max-leverage", type=float, default=10.0)
    parser.add_argument("--max-drawdown", type=float, default=0.12)
    parser.add_argument("--daily-loss-stop", type=float, default=0.03)
    parser.add_argument("--risk-per-trade", type=float, default=0.005)
    parser.add_argument("--spread-pips", type=float, default=1.0)
    parser.add_argument("--slippage-pips", type=float, default=0.2)
    parser.add_argument("--commission-pips", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    run_baselines(parse_args())


if __name__ == "__main__":
    main()
