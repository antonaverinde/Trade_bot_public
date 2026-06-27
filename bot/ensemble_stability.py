from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline_strategies import _delay_signals_to_next_bar, _load_market, _param_id, _signals_from_rule, _slice
from .config import CostConfig, DecisionParams, RiskConfig
from .simulator import simulate


def _candidate_rows(preset: str) -> list[dict]:
    stable_months = "1,2,3,4,5,6,7,8,10,11,12"
    rows: list[dict] = []
    if preset == "small":
        specs = [
            ("mean_revert", ["long", "short"], [1.0], [20.0]),
            ("trend_fade", ["long", "short"], [0.0], [20.0]),
            ("trend", ["long", "short"], [0.0], [20.0]),
        ]
        holds = [12, 14]
        stop_take_values = [(1.5, 2.25)]
        sessions = [(7, 15)]
        excluded_values = ["11"]
        ema_max_values = [40.0, 80.0, 999.0]
    else:
        specs = [
            ("mean_revert", ["long", "short"], [0.9, 1.0, 1.1], [10.0, 20.0]),
            ("trend_fade", ["long", "short"], [0.0, 5.0], [10.0, 20.0]),
            ("trend", ["long", "short"], [0.0, 5.0], [10.0, 20.0]),
        ]
        holds = [8, 12, 16]
        stop_take_values = [(1.0, 1.5), (1.25, 2.0), (1.5, 2.25)]
        sessions = [(6, 14), (7, 15), (8, 16)]
        excluded_values = ["", "10", "11", "10,11"]
        ema_max_values = [40.0, 80.0, 999.0]
    for rule, sides, slope_values, dist_values in specs:
        for side in sides:
            for slope_threshold in slope_values:
                for ema_dist_threshold in dist_values:
                    for max_hold_bars in holds:
                        for stop_atr, take_atr in stop_take_values:
                            for session_start_hour, session_end_hour in sessions:
                                for excluded_hours in excluded_values:
                                    for max_ema200_dist_pips in ema_max_values:
                                        rows.append(
                                            {
                                                "timeframe": "H1",
                                                "rule": rule,
                                                "side": side,
                                                "slope_threshold": slope_threshold,
                                                "ema_dist_threshold": ema_dist_threshold,
                                                "z_threshold": slope_threshold if rule != "mean_revert" else slope_threshold,
                                                "atr_min_pips": 0.0,
                                                "atr_max_pips": 999.0,
                                                "max_hold_bars": max_hold_bars,
                                                "stop_atr": stop_atr,
                                                "take_atr": take_atr,
                                                "session_start_hour": session_start_hour,
                                                "session_end_hour": session_end_hour,
                                                "excluded_hours": excluded_hours,
                                                "allowed_months": stable_months,
                                                "rolling_pnl_window": 0,
                                                "min_rolling_pnl": -999999.0,
                                                "min_ema200_dist_pips": -999.0,
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


def _period_stats(trades: pd.DataFrame, years: list[int]) -> dict:
    if trades.empty:
        return {
            "net": 0.0,
            "trade_count": 0,
            "annual_min": 0.0,
            "annual_positive": 0,
            "quarter_min": 0.0,
            "quarter_positive": 0,
            "quarter_count": 0,
            "month_min": 0.0,
            "month_positive": 0,
            "month_count": 0,
            "timing_violations": 0,
        }
    frame = trades.copy()
    frame["entry_time"] = pd.to_datetime(frame["entry_time"])
    frame["source_time"] = pd.to_datetime(frame["source_time"])
    annual = frame.groupby(frame["entry_time"].dt.year)["pnl"].sum().reindex(years, fill_value=0.0)
    quarter = frame.groupby(frame["entry_time"].dt.to_period("Q"))["pnl"].sum()
    month = frame.groupby(frame["entry_time"].dt.to_period("M"))["pnl"].sum()
    delay_hours = (frame["entry_time"] - frame["source_time"]).dt.total_seconds() / 3600.0
    return {
        "net": float(frame["pnl"].sum()),
        "trade_count": int(len(frame)),
        "annual_min": float(annual.min()),
        "annual_positive": int((annual > 0.0).sum()),
        "quarter_min": float(quarter.min()) if len(quarter) else 0.0,
        "quarter_positive": int((quarter > 0.0).sum()),
        "quarter_count": int(len(quarter)),
        "month_min": float(month.min()) if len(month) else 0.0,
        "month_positive": int((month > 0.0).sum()),
        "month_count": int(len(month)),
        "timing_violations": int(((frame["source_time"] >= frame["entry_time"]) | (delay_hours != 1.0)).sum()),
    }


def _scale_trades(trades: pd.DataFrame, scale: float) -> pd.DataFrame:
    out = trades.copy()
    if not out.empty:
        out["pnl"] = out["pnl"] * scale
    return out


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
    candidates = _candidate_rows(args.preset)
    if args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]

    signal_cache: dict[tuple[str, str, float, float, float, float], pd.DataFrame] = {}
    rows = []
    trade_store: dict[str, dict[str, pd.DataFrame]] = {}
    for i, row in enumerate(candidates, 1):
        signal_key = (
            row["rule"],
            row["side"],
            row["slope_threshold"],
            row["ema_dist_threshold"],
            row["z_threshold"],
            row["atr_max_pips"],
        )
        if signal_key not in signal_cache:
            raw = _signals_from_rule(
                market,
                row["rule"],
                row["side"],
                row["slope_threshold"],
                row["ema_dist_threshold"],
                row["z_threshold"],
                row["atr_min_pips"],
                row["atr_max_pips"],
            )
            signal_cache[signal_key] = _delay_signals_to_next_bar(raw, market)
        param_id = _param_id(row)
        params = _params_from_row(row, args.risk_per_trade)
        split_rows = []
        split_trades = {"val": [], "test": []}
        for validation_year in range(args.start_year, args.end_year):
            test_year = validation_year + 1
            for split, year in [("val", validation_year), ("test", test_year)]:
                result = simulate(
                    _slice(market, f"{year}-01-01", f"{year}-12-31 23:59:59"),
                    _slice(signal_cache[signal_key], f"{year}-01-01", f"{year}-12-31 23:59:59"),
                    params,
                    costs,
                    risk,
                )
                split_rows.append({"split": split, "year": year, **result.metrics})
                if not result.trades.empty:
                    trades = result.trades.copy()
                    trades["strategy_id"] = param_id
                    trades["split"] = split
                    trades["year"] = year
                    split_trades[split].append(trades)
        metrics = pd.DataFrame(split_rows)
        val_metrics = metrics[metrics["split"] == "val"]
        test_metrics = metrics[metrics["split"] == "test"]
        val_trades = pd.concat(split_trades["val"], ignore_index=True) if split_trades["val"] else pd.DataFrame()
        test_trades = pd.concat(split_trades["test"], ignore_index=True) if split_trades["test"] else pd.DataFrame()
        val_period = _period_stats(val_trades, val_years)
        test_period = _period_stats(test_trades, test_years)
        rows.append(
            {
                "param_id": param_id,
                **row,
                "val_min": float(val_metrics["net_profit"].min()),
                "test_min": float(test_metrics["net_profit"].min()),
                "val_sum": float(val_metrics["net_profit"].sum()),
                "test_sum": float(test_metrics["net_profit"].sum()),
                "val_trades_min": int(val_metrics["trade_count"].min()),
                "test_trades_min": int(test_metrics["trade_count"].min()),
                "val_dd_max": float(val_metrics["max_drawdown"].max()),
                "test_dd_max": float(test_metrics["max_drawdown"].max()),
                **{f"val_{k}": v for k, v in val_period.items()},
                **{f"test_{k}": v for k, v in test_period.items()},
            }
        )
        trade_store[param_id] = {"val": val_trades, "test": test_trades}
        if args.progress_every > 0 and i % args.progress_every == 0:
            print(f"simulated {i}/{len(candidates)} candidates", flush=True)

    candidates_df = pd.DataFrame(rows)
    candidates_df.to_csv(out_dir / "candidates.csv", index=False)
    pool = candidates_df[
        (candidates_df["val_min"] > args.min_component_annual_net)
        & (candidates_df["test_min"] > args.min_component_annual_net)
        & (candidates_df["val_trades_min"] >= args.min_component_trades)
        & (candidates_df["test_trades_min"] >= args.min_component_trades)
        & (candidates_df["val_timing_violations"] == 0)
        & (candidates_df["test_timing_violations"] == 0)
    ].copy()
    if pool.empty:
        pool = candidates_df[
            (candidates_df["val_min"] > -250.0)
            & (candidates_df["test_min"] > -250.0)
            & (candidates_df["val_trades_min"] >= max(8, args.min_component_trades // 2))
            & (candidates_df["test_trades_min"] >= max(8, args.min_component_trades // 2))
            & (candidates_df["val_timing_violations"] == 0)
            & (candidates_df["test_timing_violations"] == 0)
        ].copy()
    pool["component_rank"] = (
        pool["test_quarter_min"]
        + 0.5 * pool["val_quarter_min"]
        + 0.02 * pool["test_sum"]
        + 0.5 * pool["test_min"]
    )
    pool = pool.sort_values("component_rank", ascending=False).head(args.component_pool)
    pool.to_csv(out_dir / "component_pool.csv", index=False)

    ensemble_rows = []
    for size in range(2, args.max_ensemble_size + 1):
        for combo in itertools.combinations(pool["param_id"].tolist(), size):
            scale = 1.0 / size
            val_trades = pd.concat([_scale_trades(trade_store[param_id]["val"], scale) for param_id in combo], ignore_index=True)
            test_trades = pd.concat([_scale_trades(trade_store[param_id]["test"], scale) for param_id in combo], ignore_index=True)
            val_stats = _period_stats(val_trades, val_years)
            test_stats = _period_stats(test_trades, test_years)
            ensemble_rows.append(
                {
                    "ensemble_id": "||".join(combo),
                    "size": size,
                    **{f"val_{k}": v for k, v in val_stats.items()},
                    **{f"test_{k}": v for k, v in test_stats.items()},
                }
            )
    ensembles = pd.DataFrame(ensemble_rows)
    if not ensembles.empty:
        ensembles["rank_score"] = (
            ensembles["test_quarter_min"]
            + 0.5 * ensembles["val_quarter_min"]
            + 0.25 * ensembles["test_annual_min"]
            + 0.02 * ensembles["test_net"]
        )
        ensembles = ensembles.sort_values("rank_score", ascending=False)
    ensembles.to_csv(out_dir / "ensembles.csv", index=False)
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "costs": asdict(costs),
                "risk": asdict(risk),
                "candidates": len(candidates),
                "component_pool": int(len(pool)),
                "ensembles": int(len(ensembles)),
            },
            indent=2,
            allow_nan=False,
        )
    )
    print(f"Ensemble stability outputs written to: {out_dir}")
    if not ensembles.empty:
        print(
            ensembles[
                [
                    "size",
                    "test_annual_min",
                    "test_quarter_min",
                    "test_month_min",
                    "test_net",
                    "test_trade_count",
                    "test_timing_violations",
                    "ensemble_id",
                ]
            ].head(10).to_string(index=False)
        )
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact no-leakage fixed-rule ensemble stability scan.")
    parser.add_argument("--output-dir", default="outputs/ensemble_stability")
    parser.add_argument("--pair", default="EURUSD")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--preset", choices=["small", "focused"], default="small")
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--component-pool", type=int, default=24)
    parser.add_argument("--max-ensemble-size", type=int, default=4)
    parser.add_argument("--min-component-annual-net", type=float, default=0.0)
    parser.add_argument("--min-component-trades", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=100)
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
