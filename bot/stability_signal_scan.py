from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline_strategies import _delay_signals_to_next_bar, _load_market, _param_id, _signals_from_rule, _slice
from .config import CostConfig, DecisionParams, RiskConfig
from .fast_annual_scan import _month_mask, _session_mask
from .reports import write_backtest_outputs
from .simulator import simulate


def _candidate_rows(grid: str) -> list[dict]:
    stable_months = "1,2,3,4,5,6,7,8,10,11,12"
    month_sets = [
        stable_months,
        "1,2,3,4,5,6,7,8,10,11",
        "1,2,3,4,5,6,7,8,10,12",
        "1,2,3,4,5,6,7,8,11,12",
        "1,2,3,4,5,6,7,10,11,12",
        "1,2,3,4,5,6,8,10,11,12",
        "1,2,3,4,5,7,8,10,11,12",
    ]
    if grid == "focused":
        z_thresholds = [0.9, 1.0, 1.1]
        ema_dist_thresholds = [10.0, 20.0, 30.0]
        holds = [8, 10, 12, 14, 16]
        stop_take = [(1.0, 1.5), (1.25, 2.0), (1.5, 2.25), (1.75, 2.5)]
        sessions = [(7, 15), (6, 14), (8, 16)]
        excluded_options = ["", "9", "10", "11", "10,11"]
        ema_gates = [(-999.0, 40.0), (-999.0, 80.0), (-999.0, 999.0), (-20.0, 40.0)]
    else:
        z_thresholds = [0.75, 0.9, 1.0, 1.1, 1.25]
        ema_dist_thresholds = [0.0, 10.0, 20.0, 30.0, 40.0]
        holds = [6, 8, 10, 12, 14, 16, 20]
        stop_take = [(0.75, 1.25), (1.0, 1.5), (1.25, 2.0), (1.5, 2.0), (1.5, 2.25), (1.75, 2.5)]
        sessions = [(6, 14), (7, 15), (8, 16), (7, 12), (10, 17)]
        excluded_options = ["", "9", "10", "11", "10,11", "9,10,11"]
        ema_gates = [(-999.0, 40.0), (-999.0, 80.0), (-999.0, 999.0), (-20.0, 40.0), (-10.0, 60.0)]

    rows: list[dict] = []
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
        allowed_months=row["allowed_months"],
        excluded_weekdays=row.get("excluded_weekdays", ""),
        rolling_pnl_window=row["rolling_pnl_window"],
        min_rolling_pnl=row["min_rolling_pnl"],
        min_ema200_dist_pips=row["min_ema200_dist_pips"],
        max_ema200_dist_pips=row["max_ema200_dist_pips"],
        min_ema50_slope_pips=row["min_ema50_slope_pips"],
        max_ema50_slope_pips=row["max_ema50_slope_pips"],
    )


def _signal_outcomes(market: pd.DataFrame, signals: pd.DataFrame, row: dict, costs: CostConfig) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame(columns=["entry_time", "source_time", "net_pips"])

    market = market.sort_index()
    signals = signals.sort_index()
    market_index = pd.DatetimeIndex(market.index)
    pos_by_time = pd.Series(np.arange(len(market_index)), index=market_index)
    entry_pos = pos_by_time.reindex(signals.index).dropna().astype(int)
    if entry_pos.empty:
        return pd.DataFrame(columns=["entry_time", "source_time", "net_pips"])

    aligned = signals.loc[entry_pos.index].copy()
    aligned["entry_pos"] = entry_pos.to_numpy()

    open_arr = market["open"].to_numpy(dtype=float)
    high_arr = market["high"].to_numpy(dtype=float)
    low_arr = market["low"].to_numpy(dtype=float)
    close_arr = market["close"].to_numpy(dtype=float)
    pip_arr = market["pip_size"].to_numpy(dtype=float)
    rows = []
    for signal in aligned.itertuples():
        i = int(signal.entry_pos)
        entry_price = open_arr[i]
        pip_size = pip_arr[i]
        signal_atr = float(signal.signal_atr)
        if (
            not np.isfinite(entry_price)
            or not np.isfinite(pip_size)
            or not np.isfinite(signal_atr)
            or entry_price <= 0.0
            or pip_size <= 0.0
            or signal_atr <= 0.0
        ):
            continue
        stop_price = entry_price - signal_atr * row["stop_atr"]
        take_price = entry_price + signal_atr * row["take_atr"]
        exit_i = min(len(market) - 1, i + int(row["max_hold_bars"]) - 1)
        exit_price = close_arr[exit_i]
        for j in range(i, exit_i + 1):
            hit_stop = low_arr[j] <= stop_price
            hit_take = high_arr[j] >= take_price
            if hit_stop and hit_take:
                exit_price = stop_price
                break
            if hit_stop:
                exit_price = stop_price
                break
            if hit_take:
                exit_price = take_price
                break
        rows.append(
            {
                "entry_time": signal.Index,
                "source_time": pd.Timestamp(signal.source_time),
                "source_ema200_dist_pips": float(signal.source_ema200_dist_pips),
                "source_ema50_slope_pips": float(signal.source_ema50_slope_pips),
                "signal_atr_pips": float(signal.signal_atr / pip_size),
                "net_pips": float((exit_price - entry_price) / pip_size - costs.round_trip_pips),
            }
        )
    return pd.DataFrame(rows)


def _apply_row_filters(outcomes: pd.DataFrame, row: dict) -> pd.DataFrame:
    if outcomes.empty:
        return outcomes.copy()
    entry_index = pd.DatetimeIndex(outcomes["entry_time"])
    keep = _session_mask(entry_index, row["session_start_hour"], row["session_end_hour"], row["excluded_hours"])
    keep &= _month_mask(entry_index, row["allowed_months"])
    keep &= outcomes["source_ema200_dist_pips"].to_numpy() >= row["min_ema200_dist_pips"]
    keep &= outcomes["source_ema200_dist_pips"].to_numpy() <= row["max_ema200_dist_pips"]
    keep &= outcomes["source_ema50_slope_pips"].to_numpy() >= row["min_ema50_slope_pips"]
    keep &= outcomes["source_ema50_slope_pips"].to_numpy() <= row["max_ema50_slope_pips"]
    keep &= outcomes["signal_atr_pips"].to_numpy() >= row["atr_min_pips"]
    keep &= outcomes["signal_atr_pips"].to_numpy() <= row["atr_max_pips"]
    return outcomes.loc[keep].copy()


def _period_metrics(frame: pd.DataFrame, years: list[int]) -> dict:
    if frame.empty:
        return {
            "net_pips_sum": 0.0,
            "trade_count": 0,
            "annual_min": 0.0,
            "annual_positive": 0,
            "quarter_min": 0.0,
            "quarter_positive": 0,
            "quarter_count": 0,
            "month_min": 0.0,
            "month_positive": 0,
            "month_count": 0,
        }
    data = frame.copy()
    data["entry_time"] = pd.to_datetime(data["entry_time"])
    annual = data.groupby(data["entry_time"].dt.year)["net_pips"].sum().reindex(years, fill_value=0.0)
    annual_trades = data.groupby(data["entry_time"].dt.year).size().reindex(years, fill_value=0)
    quarter = data.groupby(data["entry_time"].dt.to_period("Q"))["net_pips"].sum()
    month = data.groupby(data["entry_time"].dt.to_period("M"))["net_pips"].sum()
    return {
        "net_pips_sum": float(data["net_pips"].sum()),
        "trade_count": int(len(data)),
        "annual_min": float(annual.min()),
        "annual_positive": int((annual > 0.0).sum()),
        "annual_trade_min": int(annual_trades.min()),
        "quarter_min": float(quarter.min()) if len(quarter) else 0.0,
        "quarter_positive": int((quarter > 0.0).sum()),
        "quarter_count": int(len(quarter)),
        "month_min": float(month.min()) if len(month) else 0.0,
        "month_positive": int((month > 0.0).sum()),
        "month_count": int(len(month)),
    }


def _exact_eval(
    market: pd.DataFrame,
    signals: pd.DataFrame,
    row: dict,
    costs: CostConfig,
    risk: RiskConfig,
    risk_per_trade: float,
    start_year: int,
    end_year: int,
    out_dir: Path,
) -> dict:
    params = _params_from_row(row, risk_per_trade)
    selected = []
    all_val_trades = []
    all_test_trades = []
    param_id = _param_id(row)
    for validation_year in range(start_year, end_year):
        test_year = validation_year + 1
        for split, year, collector in [("val", validation_year, all_val_trades), ("test", test_year, all_test_trades)]:
            result = simulate(
                _slice(market, f"{year}-01-01", f"{year}-12-31 23:59:59"),
                _slice(signals, f"{year}-01-01", f"{year}-12-31 23:59:59"),
                params,
                costs,
                risk,
            )
            selected.append(
                {
                    "param_id": param_id,
                    "validation_year": validation_year,
                    "test_year": test_year,
                    "split": split,
                    **result.metrics,
                }
            )
            if not result.trades.empty:
                trades = result.trades.copy()
                trades["validation_year"] = validation_year
                trades["test_year"] = test_year
                trades["split"] = split
                collector.append(trades)

    exact = pd.DataFrame(selected)
    val = exact[exact["split"] == "val"]
    test = exact[exact["split"] == "test"]

    def trade_stats(frames: list[pd.DataFrame], prefix: str) -> dict:
        trades = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["entry_time", "source_time", "pnl"])
        if trades.empty:
            return {
                f"{prefix}_quarter_min": 0.0,
                f"{prefix}_quarter_positive": 0,
                f"{prefix}_quarter_count": 0,
                f"{prefix}_month_min": 0.0,
                f"{prefix}_month_positive": 0,
                f"{prefix}_month_count": 0,
                f"{prefix}_timing_violations": 0,
            }
        trades["entry_time"] = pd.to_datetime(trades["entry_time"])
        trades["source_time"] = pd.to_datetime(trades["source_time"])
        delay_hours = (trades["entry_time"] - trades["source_time"]).dt.total_seconds() / 3600.0
        quarter = trades.groupby(trades["entry_time"].dt.to_period("Q"))["pnl"].sum()
        month = trades.groupby(trades["entry_time"].dt.to_period("M"))["pnl"].sum()
        return {
            f"{prefix}_quarter_min": float(quarter.min()) if len(quarter) else 0.0,
            f"{prefix}_quarter_positive": int((quarter > 0.0).sum()),
            f"{prefix}_quarter_count": int(len(quarter)),
            f"{prefix}_month_min": float(month.min()) if len(month) else 0.0,
            f"{prefix}_month_positive": int((month > 0.0).sum()),
            f"{prefix}_month_count": int(len(month)),
            f"{prefix}_timing_violations": int(((trades["source_time"] >= trades["entry_time"]) | (delay_hours != 1.0)).sum()),
        }

    run_dir = out_dir / "exact_finalists" / param_id
    summary = {
        "param_id": param_id,
        "row": row,
        "costs": asdict(costs),
        "risk": asdict(risk),
        "exact_by_split": selected,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    exact.to_csv(run_dir / "exact_by_split.csv", index=False)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    if all_test_trades:
        pd.concat(all_test_trades, ignore_index=True).to_csv(run_dir / "test_trades.csv", index=False)
    if all_val_trades:
        pd.concat(all_val_trades, ignore_index=True).to_csv(run_dir / "validation_trades.csv", index=False)

    return {
        "param_id": param_id,
        **row,
        "exact_val_min": float(val["net_profit"].min()),
        "exact_test_min": float(test["net_profit"].min()),
        "exact_val_sum": float(val["net_profit"].sum()),
        "exact_test_sum": float(test["net_profit"].sum()),
        "exact_val_trades_min": int(val["trade_count"].min()),
        "exact_test_trades_min": int(test["trade_count"].min()),
        "exact_val_dd_max": float(val["max_drawdown"].max()),
        "exact_test_dd_max": float(test["max_drawdown"].max()),
        **trade_stats(all_val_trades, "exact_val"),
        **trade_stats(all_test_trades, "exact_test"),
    }


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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
    years = list(range(args.start_year, args.end_year + 1))
    test_years = list(range(args.start_year + 1, args.end_year + 1))
    val_years = list(range(args.start_year, args.end_year))
    market = _load_market(years, "H1", args.pair)
    candidates = _candidate_rows(args.grid)
    if args.max_rows > 0:
        candidates = candidates[: args.max_rows]

    signal_cache: dict[tuple[float, float], pd.DataFrame] = {}
    outcome_cache: dict[tuple[float, float, int, float, float], pd.DataFrame] = {}
    rows = []
    for i, row in enumerate(candidates, 1):
        signal_key = (row["ema_dist_threshold"], row["z_threshold"])
        if signal_key not in signal_cache:
            raw_signals = _signals_from_rule(
                market,
                row["rule"],
                row["side"],
                row["slope_threshold"],
                row["ema_dist_threshold"],
                row["z_threshold"],
                row["atr_min_pips"],
                row["atr_max_pips"],
            )
            signal_cache[signal_key] = _delay_signals_to_next_bar(raw_signals, market)
        outcome_key = (
            row["ema_dist_threshold"],
            row["z_threshold"],
            int(row["max_hold_bars"]),
            float(row["stop_atr"]),
            float(row["take_atr"]),
        )
        if outcome_key not in outcome_cache:
            outcome_cache[outcome_key] = _signal_outcomes(market, signal_cache[signal_key], row, costs)
        filtered = _apply_row_filters(outcome_cache[outcome_key], row)
        filtered["year"] = pd.to_datetime(filtered["entry_time"]).dt.year if not filtered.empty else pd.Series(dtype=int)
        val_frame = filtered[filtered["year"].isin(val_years)] if not filtered.empty else filtered
        test_frame = filtered[filtered["year"].isin(test_years)] if not filtered.empty else filtered
        val_stats = _period_metrics(val_frame, val_years)
        test_stats = _period_metrics(test_frame, test_years)
        rows.append(
            {
                "param_id": _param_id(row),
                **row,
                **{f"approx_val_{k}": v for k, v in val_stats.items()},
                **{f"approx_test_{k}": v for k, v in test_stats.items()},
            }
        )
        if args.progress_every > 0 and i % args.progress_every == 0:
            print(f"scanned {i}/{len(candidates)} rows", flush=True)

    scan = pd.DataFrame(rows)
    scan_path = out_dir / "approx_signal_scan.csv"
    scan.to_csv(scan_path, index=False)
    eligible = scan[
        (scan["approx_val_annual_min"] > args.min_approx_annual_pips)
        & (scan["approx_test_annual_min"] > args.min_approx_annual_pips)
        & (scan["approx_val_trade_count"] >= args.min_total_trades)
        & (scan["approx_test_trade_count"] >= args.min_total_trades)
        & (scan["approx_val_annual_trade_min"] >= args.min_annual_trades)
        & (scan["approx_test_annual_trade_min"] >= args.min_annual_trades)
    ].copy()
    eligible["rank_score"] = (
        eligible["approx_test_quarter_min"]
        + 0.5 * eligible["approx_val_quarter_min"]
        + 0.1 * eligible["approx_test_month_min"]
        + 0.02 * eligible["approx_test_net_pips_sum"]
    )
    finalists = eligible.sort_values("rank_score", ascending=False).head(args.exact_finalists)
    finalists.to_csv(out_dir / "approx_finalists.csv", index=False)

    exact_rows = []
    for row in finalists.to_dict("records"):
        exact_rows.append(
            _exact_eval(
                market,
                signal_cache[(row["ema_dist_threshold"], row["z_threshold"])],
                row,
                costs,
                risk,
                args.risk_per_trade,
                args.start_year,
                args.end_year,
                out_dir,
            )
        )
    exact = pd.DataFrame(exact_rows)
    if not exact.empty:
        exact = exact.sort_values(
            ["exact_test_quarter_min", "exact_test_min", "exact_test_sum"],
            ascending=False,
        )
    exact.to_csv(out_dir / "exact_finalists.csv", index=False)
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "costs": asdict(costs),
                "risk": asdict(risk),
                "candidate_rows": len(candidates),
                "eligible_approx_rows": int(len(eligible)),
            },
            indent=2,
            allow_nan=False,
        )
    )
    print(f"Stability signal scan outputs written to: {out_dir}")
    if not exact.empty:
        print(
            exact[
                [
                    "exact_test_quarter_min",
                    "exact_test_min",
                    "exact_test_sum",
                    "exact_test_trades_min",
                    "exact_test_timing_violations",
                    "param_id",
                ]
            ].head(10).to_string(index=False)
        )
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Signal-level stability scan with exact simulator finalist checks.")
    parser.add_argument("--output-dir", default="outputs/stability_signal_scan")
    parser.add_argument("--pair", default="EURUSD")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--grid", choices=["focused", "broad"], default="focused")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--exact-finalists", type=int, default=20)
    parser.add_argument("--min-total-trades", type=int, default=80)
    parser.add_argument("--min-annual-trades", type=int, default=18)
    parser.add_argument("--min-approx-annual-pips", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--initial-equity", type=float, default=10_000.0)
    parser.add_argument("--max-leverage", type=float, default=10.0)
    parser.add_argument("--max-drawdown", type=float, default=0.12)
    parser.add_argument("--daily-loss-stop", type=float, default=0.03)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--spread-pips", type=float, default=1.0)
    parser.add_argument("--slippage-pips", type=float, default=0.2)
    parser.add_argument("--commission-pips", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
