from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CostConfig, OptimizerConfig, RiskConfig
from .optimize import _flat_metrics, objective_score
from .simulator import BacktestResult, simulate
from .walkforward import (
    FOLDS,
    FVG_LEVEL_FAMILIES,
    _dataset_market,
    _fixed_grid,
    _parse_float_csv,
    _parse_label_modes,
    _select_folds,
    build_fvg_dataset,
)


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


def _period_bounds(start: pd.Timestamp, end: pd.Timestamp, freq: str) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    periods = pd.period_range(start=start, end=end, freq=freq)
    out = []
    for period in periods:
        p_start = max(start, period.start_time)
        p_end = min(end, period.end_time)
        out.append((str(period), p_start, p_end))
    return out


def _slice_by_index(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    idx = pd.DatetimeIndex(df.index)
    return df.loc[(idx >= start) & (idx <= end)].copy()


def _simulate_period(
    market: pd.DataFrame,
    signals: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    params,
    costs: CostConfig,
    risk: RiskConfig,
) -> BacktestResult:
    period_signals = _slice_by_index(signals, start, end)
    if period_signals.empty:
        period_market = _slice_by_index(market, start, end)
    else:
        # Include bars from the first signal to the period end. This keeps each
        # period independent while allowing entries near the start to exit.
        period_market = _slice_by_index(market, min(start, period_signals.index.min()), end)
    return simulate(period_market, period_signals, params, costs, risk)


def _period_rows(
    fold: dict,
    split: str,
    market: pd.DataFrame,
    signals: pd.DataFrame,
    params,
    costs: CostConfig,
    risk: RiskConfig,
    freq: str,
) -> list[dict]:
    if split == "val":
        start = pd.Timestamp(fold["val_start"])
        end = pd.Timestamp(fold["val_end"])
    elif split == "test":
        start = pd.Timestamp(fold["test_start"])
        end = pd.Timestamp(fold["test_end"])
    else:
        raise ValueError(f"Unsupported split: {split}")

    rows = []
    for period, p_start, p_end in _period_bounds(start, end, freq):
        result = _simulate_period(market, signals, p_start, p_end, params, costs, risk)
        rows.append(
            {
                "fold": fold["name"],
                "split": split,
                "period": period,
                "period_start": p_start,
                "period_end": p_end,
                **_flat_metrics(result.metrics),
            }
        )
    return rows


def _score_periods(period_frame: pd.DataFrame, min_trades: int, max_drawdown: float) -> dict:
    if period_frame.empty:
        return {
            "period_count": 0,
            "positive_periods": 0,
            "tradable_periods": 0,
            "positive_tradable_periods": 0,
            "min_period_net": 0.0,
            "median_period_net": 0.0,
            "total_period_net": 0.0,
            "worst_period_drawdown": 0.0,
            "stable_score": -999.0,
        }
    net = period_frame["net_profit"].astype(float)
    trades = period_frame["trade_count"].astype(float)
    dd = period_frame["max_drawdown"].astype(float)
    tradable = trades >= min_trades
    positive = net > 0
    positive_tradable = positive & tradable
    min_net = float(net.min())
    median_net = float(net.median())
    total_net = float(net.sum())
    worst_dd = float(dd.max())
    stable_score = (
        total_net / 1000.0
        + float(positive_tradable.sum()) * 0.75
        + float(positive.sum()) * 0.20
        + min_net / 500.0
        - max(0.0, worst_dd - max_drawdown) * 10.0
    )
    if min_net <= 0:
        stable_score += min_net / 250.0
    return {
        "period_count": int(len(period_frame)),
        "positive_periods": int(positive.sum()),
        "tradable_periods": int(tradable.sum()),
        "positive_tradable_periods": int(positive_tradable.sum()),
        "min_period_net": min_net,
        "median_period_net": median_net,
        "total_period_net": total_net,
        "worst_period_drawdown": worst_dd,
        "stable_score": float(stable_score),
    }


def _param_id(params) -> str:
    parts = [
        f"h{params.max_hold_bars}",
        f"pred{params.min_predicted_net_pips:g}",
        f"fvg{params.min_fvg_size_atr:g}",
        f"atr{params.min_signal_atr_pips:g}-{params.max_signal_atr_pips:g}",
        f"s{params.session_start_hour}-{params.session_end_hour}",
        f"m{params.allowed_months or 'all'}",
        f"e200{params.min_ema200_dist_pips:g}-{params.max_ema200_dist_pips:g}",
        f"e50s{params.min_ema50_slope_pips:g}-{params.max_ema50_slope_pips:g}",
        f"day{params.max_trades_per_day}",
    ]
    return "_".join(parts)


def run_diagnostics(args: argparse.Namespace) -> Path:
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

    level_families = [item.strip() for item in args.fvg_level_families.split(",") if item.strip()]
    unknown_families = sorted(set(level_families) - set(FVG_LEVEL_FAMILIES))
    if unknown_families:
        raise ValueError(f"Unknown --fvg-level-families values: {unknown_families}")

    period_rows = []
    summary_rows = []
    run_meta = {
        "args": vars(args),
        "costs": asdict(costs),
        "risk": asdict(risk),
        "optimizer": asdict(optimizer_cfg),
    }
    selected_folds = _select_folds(args.folds)
    label_modes = _parse_label_modes(args.fvg_label_modes)
    profit_buffers = _parse_float_csv(args.profit_buffers, [0.5])

    for fold in selected_folds:
        for label_mode in label_modes:
            for level_family in level_families:
                for profit_buffer in profit_buffers:
                    dataset, model_metrics = build_fvg_dataset(
                        fold,
                        args.seed,
                        args.class_weight,
                        args.xgb_estimators,
                        label_mode=label_mode,
                        profit_buffer_pips=profit_buffer,
                        base_timeframe=args.fvg_base_timeframe,
                        higher_timeframe=args.fvg_higher_timeframe,
                        level_family=level_family,
                        min_fvg_atr=args.fvg_min_fvg_atr,
                        require_unbroken_levels=args.fvg_require_unbroken_levels,
                        decision_delay_bars=args.fvg_decision_delay_bars,
                        single_timeframe=args.fvg_single_timeframe,
                    )
                    params_grid = _fixed_grid(
                        args.level_mode,
                        args.fvg_trade_side,
                        dataset.model_name,
                        args.fixed_grid_profile,
                    )
                    if args.max_params > 0:
                        params_grid = params_grid[: args.max_params]

                    for trial, params in enumerate(params_grid):
                        param_key = _param_id(params)
                        full_metrics = {}
                        split_period_rows = []
                        for split in ["val", "test"]:
                            result = simulate(
                                dataset.split_market[split],
                                dataset.split_signals[split],
                                params,
                                costs,
                                risk,
                            )
                            full_metrics[split] = _flat_metrics(result.metrics)
                            rows = _period_rows(
                                fold,
                                split,
                                dataset.split_market[split],
                                dataset.split_signals[split],
                                params,
                                costs,
                                risk,
                                args.period_freq,
                            )
                            for row in rows:
                                row.update(
                                    {
                                        "label_mode": label_mode,
                                        "profit_buffer_pips": profit_buffer,
                                        "level_family": level_family,
                                        "trial": trial,
                                        "param_id": param_key,
                                        **params.to_dict(),
                                    }
                                )
                            split_period_rows.extend(rows)
                            period_rows.extend(rows)

                        period_frame = pd.DataFrame(split_period_rows)
                        period_stats = _score_periods(
                            period_frame,
                            args.period_min_trades,
                            args.max_drawdown,
                        )
                        score = objective_score(full_metrics["val"], optimizer_cfg)
                        summary_rows.append(
                            {
                                "fold": fold["name"],
                                "label_mode": label_mode,
                                "profit_buffer_pips": profit_buffer,
                                "level_family": level_family,
                                "trial": trial,
                                "param_id": param_key,
                                "val_score": score,
                                **params.to_dict(),
                                **{f"val_{k}": v for k, v in full_metrics["val"].items()},
                                **{f"test_{k}": v for k, v in full_metrics["test"].items()},
                                **period_stats,
                            }
                        )

                    run_meta.setdefault("model_metrics", {})[
                        f"{fold['name']}|{label_mode}|{level_family}|{profit_buffer}"
                    ] = model_metrics

    period_df = pd.DataFrame(period_rows)
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["stable_score", "positive_tradable_periods", "total_period_net"],
            ascending=[False, False, False],
        )
    period_df.to_csv(out_dir / "period_metrics.csv", index=False)
    summary_df.to_csv(out_dir / "params_summary.csv", index=False)
    (out_dir / "diagnostic_meta.json").write_text(json.dumps(_clean(run_meta), indent=2, allow_nan=False))

    print(f"Regime diagnostics written to: {out_dir}")
    if summary_df.empty:
        print("No rows produced.")
    else:
        display_cols = [
            "fold",
            "label_mode",
            "param_id",
            "val_net_profit",
            "test_net_profit",
            "val_trade_count",
            "test_trade_count",
            "positive_tradable_periods",
            "period_count",
            "min_period_net",
            "total_period_net",
            "worst_period_drawdown",
            "stable_score",
        ]
        print(summary_df[display_cols].head(args.print_top).to_string(index=False))
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose FVG strategy stability by calendar period.")
    parser.add_argument("--output-dir", default="outputs/regime_diagnostics")
    parser.add_argument("--folds", default="all")
    parser.add_argument("--period-freq", default="Q", help="Pandas period frequency, for example Q or M.")
    parser.add_argument("--period-min-trades", type=int, default=5)
    parser.add_argument("--print-top", type=int, default=20)
    parser.add_argument("--max-params", type=int, default=0, help="Optional first-N param cap for quick smoke runs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=120)
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
    parser.add_argument("--xgb-estimators", type=int, default=80)
    parser.add_argument("--profit-buffers", default="0.5")
    parser.add_argument(
        "--fvg-label-modes",
        default="long_pips_ridge",
        help="Comma-separated FVG label modes.",
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
        default="regime_fvg",
    )
    parser.add_argument(
        "--fvg-level-families",
        default="higher_only",
        help=f"Comma-separated FVG level source presets: {','.join(FVG_LEVEL_FAMILIES)}",
    )
    parser.add_argument("--fvg-min-fvg-atr", type=float, default=None)
    parser.add_argument("--fvg-require-unbroken-levels", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fvg-decision-delay-bars", type=int, default=None)
    parser.add_argument("--fvg-base-timeframe", default="M15")
    parser.add_argument("--fvg-higher-timeframe", default="H1")
    parser.add_argument("--fvg-single-timeframe", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fvg-trade-side", choices=["long", "short", "both"], default="long")
    return parser.parse_args()


def main() -> None:
    run_diagnostics(parse_args())


if __name__ == "__main__":
    main()
