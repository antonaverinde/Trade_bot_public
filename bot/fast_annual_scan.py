from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline_strategies import (
    _delay_signals_to_next_bar,
    _load_market,
    _param_id,
    _signals_from_rule,
    _slice,
)
from .config import CostConfig, DecisionParams, OptimizerConfig, RiskConfig
from .optimize import objective_score
from .simulator import simulate


def _parse_hours(value: str) -> set[int]:
    value = str(value).strip()
    if not value:
        return set()
    out = {int(item.strip()) for item in value.split(",") if item.strip()}
    if any(hour < 0 or hour > 23 for hour in out):
        raise ValueError("excluded hours must be in 0..23")
    return out


def _parse_months(value: str) -> set[int]:
    value = str(value).strip()
    if not value:
        return set()
    out = {int(item.strip()) for item in value.split(",") if item.strip()}
    if any(month < 1 or month > 12 for month in out):
        raise ValueError("allowed months must be in 1..12")
    return out


def _session_mask(index: pd.DatetimeIndex, start: int, end: int, excluded_hours: str) -> np.ndarray:
    hours = index.hour.to_numpy()
    excluded = _parse_hours(excluded_hours)
    if start < 0 or end < 0 or start == end:
        mask = np.ones(len(index), dtype=bool)
    elif start < end:
        mask = (hours >= start) & (hours < end)
    else:
        mask = (hours >= start) | (hours < end)
    if excluded:
        mask &= ~np.isin(hours, np.array(sorted(excluded), dtype=int))
    return mask


def _month_mask(index: pd.DatetimeIndex, allowed_months: str) -> np.ndarray:
    months = _parse_months(allowed_months)
    if not months:
        return np.ones(len(index), dtype=bool)
    return np.isin(index.month.to_numpy(), np.array(sorted(months), dtype=int))


def _weekday_mask(index: pd.DatetimeIndex, excluded_weekdays: str) -> np.ndarray:
    value = str(excluded_weekdays).strip()
    if not value:
        return np.ones(len(index), dtype=bool)
    excluded = {int(item.strip()) for item in value.split(",") if item.strip()}
    if any(day < 0 or day > 6 for day in excluded):
        raise ValueError("excluded weekdays must be in 0..6")
    return ~np.isin(index.weekday.to_numpy(), np.array(sorted(excluded), dtype=int))


def _fast_simulate_long_atr(
    market: pd.DataFrame,
    signals: pd.DataFrame,
    params: DecisionParams,
    costs: CostConfig,
    risk: RiskConfig,
) -> dict:
    if market.empty or signals.empty:
        return {
            "net_profit": 0.0,
            "max_drawdown": 0.0,
            "trade_count": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_trade_pnl": 0.0,
            "total_cost_pips": 0.0,
        }

    market = market.sort_index()
    signals = signals.sort_index()
    index = pd.DatetimeIndex(market.index)
    pos_by_time = pd.Series(np.arange(len(index)), index=index)
    entry_pos = pos_by_time.reindex(signals.index).dropna().astype(int)
    if entry_pos.empty:
        return {
            "net_profit": 0.0,
            "max_drawdown": 0.0,
            "trade_count": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_trade_pnl": 0.0,
            "total_cost_pips": 0.0,
        }

    signals = signals.loc[entry_pos.index].copy()
    signals["entry_pos"] = entry_pos.to_numpy()
    entry_index = pd.DatetimeIndex(signals.index)
    keep = _session_mask(
        entry_index,
        params.session_start_hour,
        params.session_end_hour,
        params.excluded_hours,
    )
    keep &= _month_mask(entry_index, params.allowed_months)
    keep &= _weekday_mask(entry_index, params.excluded_weekdays)
    keep &= (
        signals["source_ema200_dist_pips"].to_numpy() >= params.min_ema200_dist_pips
    ) & (
        signals["source_ema200_dist_pips"].to_numpy() <= params.max_ema200_dist_pips
    )
    keep &= (
        signals["source_ema50_slope_pips"].to_numpy() >= params.min_ema50_slope_pips
    ) & (
        signals["source_ema50_slope_pips"].to_numpy() <= params.max_ema50_slope_pips
    )
    signal_atr_pips = signals["signal_atr"].to_numpy() / market["pip_size"].reindex(signals.index).to_numpy()
    keep &= np.isfinite(signal_atr_pips)
    keep &= (signal_atr_pips >= params.min_signal_atr_pips) & (signal_atr_pips <= params.max_signal_atr_pips)
    signals = signals.loc[keep]
    if signals.empty:
        return {
            "net_profit": 0.0,
            "max_drawdown": 0.0,
            "trade_count": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_trade_pnl": 0.0,
            "total_cost_pips": 0.0,
        }

    open_arr = market["open"].to_numpy(dtype=float)
    high_arr = market["high"].to_numpy(dtype=float)
    low_arr = market["low"].to_numpy(dtype=float)
    close_arr = market["close"].to_numpy(dtype=float)
    pip_arr = market["pip_size"].to_numpy(dtype=float)
    times = pd.DatetimeIndex(market.index)

    equity = float(risk.initial_equity)
    peak_equity = equity
    max_dd = 0.0
    cooldown_until = -1
    daily_start_equity = equity
    current_day = None
    stop_trading = False
    pnls: list[float] = []

    for row in signals.itertuples():
        i = int(row.entry_pos)
        day = times[i].date()
        if day != current_day:
            current_day = day
            daily_start_equity = equity

        if stop_trading or i < cooldown_until:
            continue

        daily_loss = (daily_start_equity - equity) / daily_start_equity if daily_start_equity else 0.0
        realized_dd = (peak_equity - equity) / peak_equity if peak_equity else 0.0
        if daily_loss >= risk.daily_loss_stop or realized_dd >= risk.max_drawdown_stop:
            continue

        entry_price = open_arr[i]
        pip_size = pip_arr[i]
        signal_atr = float(row.signal_atr)
        if not np.isfinite(signal_atr) or signal_atr <= 0 or not np.isfinite(entry_price) or entry_price <= 0:
            continue

        stop_distance = signal_atr * params.stop_atr
        take_distance = signal_atr * params.take_atr
        if stop_distance <= 0 or take_distance <= 0:
            continue

        units = min(equity * params.risk_per_trade / stop_distance, equity * risk.max_leverage / entry_price)
        if units <= 0:
            continue

        stop_price = entry_price - stop_distance
        take_price = entry_price + take_distance
        exit_i = min(len(market) - 1, i + params.max_hold_bars - 1)
        exit_price = close_arr[exit_i]

        for j in range(i, exit_i + 1):
            mark_equity = equity + (close_arr[j] - entry_price) * units - costs.round_trip_pips * pip_arr[j] * units
            peak_equity = max(peak_equity, mark_equity)
            max_dd = max(max_dd, (peak_equity - mark_equity) / peak_equity if peak_equity else 0.0)
            if max_dd >= risk.max_drawdown_stop:
                stop_trading = True

            hit_stop = low_arr[j] <= stop_price
            hit_take = high_arr[j] >= take_price
            if hit_stop and hit_take:
                exit_i = j
                exit_price = stop_price
                break
            if hit_stop:
                exit_i = j
                exit_price = stop_price
                break
            if hit_take:
                exit_i = j
                exit_price = take_price
                break

        gross_pips = (exit_price - entry_price) / pip_size
        net_pips = gross_pips - costs.round_trip_pips
        pnl = net_pips * pip_size * units
        equity += pnl
        pnls.append(float(pnl))
        peak_equity = max(peak_equity, equity)
        max_dd = max(max_dd, (peak_equity - equity) / peak_equity if peak_equity else 0.0)
        cooldown_until = exit_i + params.cooldown_bars

    wins = [pnl for pnl in pnls if pnl > 0]
    losses = [-pnl for pnl in pnls if pnl < 0]
    gross_profit = float(np.sum(wins)) if wins else 0.0
    gross_loss = float(np.sum(losses)) if losses else 0.0
    return {
        "net_profit": float(equity - risk.initial_equity),
        "max_drawdown": float(max_dd),
        "trade_count": int(len(pnls)),
        "win_rate": float(len(wins) / len(pnls)) if pnls else 0.0,
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0),
        "avg_trade_pnl": float(np.mean(pnls)) if pnls else 0.0,
        "total_cost_pips": float(len(pnls) * costs.round_trip_pips),
    }


def _candidate_rows(grid: str) -> list[dict]:
    rows: list[dict] = []
    month_sets = [
        "",
        "2,3,4,5,6,7,8,9,10,11,12",
        "1,3,4,5,6,7,8,9,10,11,12",
        "1,2,4,5,6,7,8,9,10,11,12",
        "1,2,3,5,6,7,8,9,10,11,12",
        "1,2,3,4,6,7,8,9,10,11,12",
        "1,2,3,4,5,7,8,9,10,11,12",
        "1,2,3,4,5,6,8,9,10,11,12",
        "1,2,3,4,5,6,7,9,10,11,12",
        "1,2,3,4,5,6,7,8,10,11,12",
        "1,2,3,4,5,6,7,8,9,11,12",
        "1,2,3,4,5,6,7,8,9,10,12",
        "1,2,3,4,5,6,7,8,9,10,11",
    ]
    if grid == "fixed":
        z_thresholds = [1.0]
        ema_dist_thresholds = [20.0]
        holds = [12]
        stop_take = [(1.5, 2.25)]
        sessions = [(7, 15)]
        excluded_options = ["11"]
        ema_gates = [(-999.0, 80.0)]
        month_sets = ["1,2,3,4,5,6,7,8,10,11,12"]
    elif grid == "compact":
        z_thresholds = [1.0]
        ema_dist_thresholds = [20.0]
        holds = [12]
        stop_take = [(1.5, 2.25)]
        sessions = [(7, 15)]
        excluded_options = ["", "9", "10", "11", "10,11"]
        ema_gates = [(-999.0, 999.0), (-999.0, 80.0), (-999.0, 40.0), (-20.0, 40.0), (-10.0, 60.0)]
    elif grid == "focused":
        z_thresholds = [1.0]
        ema_dist_thresholds = [20.0]
        holds = [10, 12, 14]
        stop_take = [(1.25, 2.0), (1.5, 2.25), (1.75, 2.5)]
        sessions = [(7, 15)]
        excluded_options = ["", "9", "10", "11", "10,11"]
        ema_gates = [(-999.0, 999.0), (-999.0, 80.0), (-999.0, 40.0)]
    elif grid == "medium":
        z_thresholds = [0.9, 1.0, 1.1]
        ema_dist_thresholds = [10.0, 20.0, 40.0]
        holds = [10, 12, 14]
        stop_take = [(1.25, 2.0), (1.5, 2.25), (1.75, 2.5)]
        sessions = [(6, 14), (7, 15), (8, 16), (7, 12)]
        excluded_options = ["", "9", "10", "11", "10,11"]
        ema_gates = [(-999.0, 999.0), (-999.0, 80.0), (-999.0, 40.0), (-20.0, 40.0), (-10.0, 60.0)]
    else:
        z_thresholds = [0.75, 0.9, 1.0, 1.1, 1.25]
        ema_dist_thresholds = [0.0, 10.0, 20.0, 40.0, 60.0]
        holds = [8, 10, 12, 14, 16]
        stop_take = [(1.0, 1.5), (1.25, 2.0), (1.5, 2.0), (1.5, 2.25), (1.75, 2.5)]
        sessions = [(6, 14), (7, 15), (8, 16), (7, 12), (10, 17)]
        excluded_options = ["", "9", "10", "11", "10,11"]
        ema_gates = [(-999.0, 999.0), (-999.0, 80.0), (-999.0, 40.0), (-20.0, 40.0), (-10.0, 60.0)]

    for z_threshold in z_thresholds:
        for ema_dist_threshold in ema_dist_thresholds:
            for max_hold_bars in holds:
                for stop_atr, take_atr in stop_take:
                    for session_start_hour, session_end_hour in sessions:
                        for excluded_hours in excluded_options:
                            for min_ema200_dist_pips, max_ema200_dist_pips in ema_gates:
                                for allowed_months in month_sets:
                                    rows.append(
                                        {
                                            "timeframe": "H1",
                                            "rule": "mean_revert",
                                            "side": "long",
                                            "slope_threshold": 0.0,
                                            "ema_dist_threshold": ema_dist_threshold,
                                            "z_threshold": z_threshold,
                                            "atr_min_pips": 0.0,
                                            "atr_max_pips": 999.0,
                                            "max_hold_bars": max_hold_bars,
                                            "stop_atr": stop_atr,
                                            "take_atr": take_atr,
                                            "session_start_hour": session_start_hour,
                                            "session_end_hour": session_end_hour,
                                            "excluded_hours": excluded_hours,
                                            "allowed_months": allowed_months,
                                            "rolling_pnl_window": 0,
                                            "min_rolling_pnl": -999999.0,
                                            "min_ema200_dist_pips": min_ema200_dist_pips,
                                            "max_ema200_dist_pips": max_ema200_dist_pips,
                                            "min_ema50_slope_pips": -999.0,
                                            "max_ema50_slope_pips": 999.0,
                                        }
                                    )
    return rows


def _params_from_row(row: dict, risk_per_trade: float) -> DecisionParams:
    return DecisionParams(
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
        risk_per_trade=risk_per_trade,
        min_signal_atr_pips=row["atr_min_pips"],
        max_signal_atr_pips=row["atr_max_pips"],
        session_start_hour=row["session_start_hour"],
        session_end_hour=row["session_end_hour"],
        excluded_hours=row["excluded_hours"],
        allowed_months=row.get("allowed_months", ""),
        excluded_weekdays=row.get("excluded_weekdays", ""),
        min_ema200_dist_pips=row["min_ema200_dist_pips"],
        max_ema200_dist_pips=row["max_ema200_dist_pips"],
        min_ema50_slope_pips=row["min_ema50_slope_pips"],
        max_ema50_slope_pips=row["max_ema50_slope_pips"],
    )


def _year_slice(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    return _slice(frame, f"{year}-01-01", f"{year}-12-31 23:59:59")


def _selection_summary(frame: pd.DataFrame) -> dict:
    selected = frame.sort_values("validation_year")
    return {
        "windows": int(len(selected)),
        "eligible_windows": int((selected["selection_status"] == "eligible_validation_row").sum()),
        "positive_test_years": int((selected["test_net_profit"] > 0.0).sum()),
        "min_validation_net": float(selected["val_net_profit"].min()),
        "min_test_net": float(selected["test_net_profit"].min()),
        "validation_net_sum": float(selected["val_net_profit"].sum()),
        "test_net_sum": float(selected["test_net_profit"].sum()),
        "min_validation_trades": int(selected["val_trade_count"].min()),
        "min_test_trades": int(selected["test_trade_count"].min()),
        "max_validation_drawdown": float(selected["val_max_drawdown"].max()),
        "max_test_drawdown": float(selected["test_max_drawdown"].max()),
    }


def _selection_score(frame: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "conservative":
        return 0.25 * frame["val_net_profit"] + 20.0 * frame["val_trade_count"]
    if mode == "trade_count":
        return frame["val_trade_count"] + 0.001 * frame["val_net_profit"]
    return frame["score"]


def run(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
        min_trades=args.min_trades,
        max_acceptable_drawdown=args.max_drawdown,
    )

    market = _load_market(list(range(args.start_year, args.end_year + 1)), "H1", args.pair)
    rows = _candidate_rows(args.grid)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    signal_cache: dict[tuple[float, float], pd.DataFrame] = {}
    trials: list[dict] = []
    windows = [(year, year + 1) for year in range(args.start_year, args.end_year)]

    for row_i, row in enumerate(rows):
        cache_key = (row["ema_dist_threshold"], row["z_threshold"])
        if cache_key not in signal_cache:
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
            signal_cache[cache_key] = _delay_signals_to_next_bar(signals, market)
        signals = signal_cache[cache_key]
        params = _params_from_row(row, args.risk_per_trade)
        param_id = _param_id(row)

        for validation_year, test_year in windows:
            val_market = _year_slice(market, validation_year)
            test_market = _year_slice(market, test_year)
            val_signals = _year_slice(signals, validation_year)
            test_signals = _year_slice(signals, test_year)
            val_metrics = _fast_simulate_long_atr(val_market, val_signals, params, costs, risk)
            test_metrics = _fast_simulate_long_atr(test_market, test_signals, params, costs, risk)
            score = objective_score(val_metrics, optimizer_cfg)
            trials.append(
                {
                    "row": row_i,
                    "param_id": param_id,
                    "validation_year": validation_year,
                    "test_year": test_year,
                    "score": score,
                    **row,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                }
            )
        if args.progress_every > 0 and (row_i + 1) % args.progress_every == 0:
            print(f"scanned {row_i + 1}/{len(rows)} rows")

    trials_df = pd.DataFrame(trials)
    trials_df.to_csv(output_dir / "fast_trials.csv", index=False)

    selected_rows = []
    for validation_year, group in trials_df.groupby("validation_year", sort=True):
        group = group.copy()
        group["selection_score"] = _selection_score(group, args.selection_mode)
        eligible = group[
            (group["val_net_profit"] > 0.0)
            & (group["val_trade_count"] >= args.min_trades)
            & (group["val_max_drawdown"] <= args.max_drawdown)
        ].sort_values("selection_score", ascending=False)
        ranked = group.sort_values("selection_score", ascending=False)
        selected = eligible.head(1) if not eligible.empty else ranked.head(1)
        selected = selected.copy()
        selected["selection_status"] = "eligible_validation_row" if not eligible.empty else "no_eligible_validation_row"
        selected_rows.append(selected)
    selected_df = pd.concat(selected_rows, ignore_index=True)
    selected_df.to_csv(output_dir / "fast_selected_by_year.csv", index=False)
    (output_dir / "fast_selection_summary.json").write_text(
        json.dumps(_selection_summary(selected_df), indent=2, allow_nan=False)
    )

    exact_rows = []
    for row in selected_df.to_dict("records"):
        params = _params_from_row(row, args.risk_per_trade)
        signals = signal_cache[(row["ema_dist_threshold"], row["z_threshold"])]
        for prefix, year in [("val", int(row["validation_year"])), ("test", int(row["test_year"]))]:
            result = simulate(_year_slice(market, year), _year_slice(signals, year), params, costs, risk)
            exact_rows.append(
                {
                    "param_id": row["param_id"],
                    "validation_year": int(row["validation_year"]),
                    "test_year": int(row["test_year"]),
                    "split": prefix,
                    **{k: result.metrics[k] for k in ["net_profit", "max_drawdown", "trade_count", "win_rate", "profit_factor", "avg_trade_pnl", "total_cost_pips"]},
                }
            )
    exact_df = pd.DataFrame(exact_rows)
    exact_df.to_csv(output_dir / "exact_selected_simulator_check.csv", index=False)
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "costs": asdict(costs),
                "risk": asdict(risk),
                "args": vars(args),
                "candidate_rows": len(rows),
            },
            indent=2,
            allow_nan=False,
        )
    )
    print(f"Fast annual scan outputs written to: {output_dir}")
    print(selected_df[["validation_year", "test_year", "param_id", "val_net_profit", "test_net_profit", "selection_status"]].to_string(index=False))
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast annual discovery scan for H1 mean-reversion baselines.")
    parser.add_argument("--output-dir", default="outputs/baseline_runs/fast_annual_scan")
    parser.add_argument("--pair", default="EURUSD")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--grid", choices=["fixed", "compact", "focused", "medium", "broad"], default="compact")
    parser.add_argument("--selection-mode", choices=["objective", "conservative", "trade_count"], default="objective")
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--min-trades", type=int, default=20)
    parser.add_argument("--initial-equity", type=float, default=10_000.0)
    parser.add_argument("--max-leverage", type=float, default=10.0)
    parser.add_argument("--max-drawdown", type=float, default=0.12)
    parser.add_argument("--daily-loss-stop", type=float, default=0.03)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--spread-pips", type=float, default=1.0)
    parser.add_argument("--slippage-pips", type=float, default=0.2)
    parser.add_argument("--commission-pips", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
