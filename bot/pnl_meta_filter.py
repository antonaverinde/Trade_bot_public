from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .baseline_strategies import (
    _delay_signals_to_next_bar,
    _load_market,
    _param_id,
    _signals_from_rule,
    _slice,
)
from .config import CostConfig, DecisionParams, OptimizerConfig, RiskConfig
from .meta_filter import FEATURE_COLS, _add_meta_features
from .optimize import _flat_metrics
from .reports import write_backtest_outputs
from .simulator import simulate
from .walkforward import _select_folds


SOURCE_FEATURE_COLS = [
    "atr_pips",
    "ema200_dist_pips",
    "ema20_slope_pips",
    "ema50_slope_pips",
    "z96",
    "rsi14",
]

NO_CALENDAR_FEATURE_COLS = [
    "atr_pips",
    "ema200_dist_pips",
    "ema20_slope_pips",
    "ema50_slope_pips",
    "z96",
    "rsi14",
    "ret_1_pips",
    "ret_4_pips",
    "ret_24_pips",
    "vol_24_pips",
    "trend_eff_24",
    "return_autocorr_24",
]


def _feature_cols(feature_set: str) -> list[str]:
    if feature_set == "source":
        return SOURCE_FEATURE_COLS
    if feature_set == "no_calendar":
        return NO_CALENDAR_FEATURE_COLS
    return FEATURE_COLS


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


def _stable_row() -> dict:
    return {
        "timeframe": "H1",
        "rule": "mean_revert",
        "side": "long",
        "slope_threshold": 0.0,
        "ema_dist_threshold": 20.0,
        "z_threshold": 1.0,
        "atr_min_pips": 0.0,
        "atr_max_pips": 999.0,
        "max_hold_bars": 12,
        "stop_atr": 1.5,
        "take_atr": 2.25,
        "session_start_hour": 7,
        "session_end_hour": 15,
        "excluded_hours": "11",
        "allowed_months": "1,2,3,4,5,6,7,8,10,11,12",
        "rolling_pnl_window": 0,
        "min_rolling_pnl": -999999.0,
        "min_ema200_dist_pips": -999.0,
        "max_ema200_dist_pips": 80.0,
        "min_ema50_slope_pips": -999.0,
        "max_ema50_slope_pips": 999.0,
    }


def _annual_expanding_folds(start_year: int, end_year: int) -> list[dict]:
    folds = []
    for test_year in range(start_year + 2, end_year + 1):
        validation_year = test_year - 1
        train_years = list(range(start_year, validation_year))
        folds.append(
            {
                "name": f"annual_train_{start_year}_{validation_year - 1}_val_{validation_year}_test_{test_year}",
                "train_start": f"{start_year}-01-01",
                "train_end": f"{validation_year - 1}-12-31 23:59:59",
                "val_start": f"{validation_year}-01-01",
                "val_end": f"{validation_year}-12-31 23:59:59",
                "test_start": f"{test_year}-01-01",
                "test_end": f"{test_year}-12-31 23:59:59",
                "years": train_years + [validation_year, test_year],
            }
        )
    return folds


def _resolve_folds(args: argparse.Namespace) -> list[dict]:
    if args.folds == "annual_expanding":
        return _annual_expanding_folds(args.start_year, args.end_year)
    return _select_folds(args.folds)


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
        min_ema200_dist_pips=row["min_ema200_dist_pips"],
        max_ema200_dist_pips=row["max_ema200_dist_pips"],
        min_ema50_slope_pips=row["min_ema50_slope_pips"],
        max_ema50_slope_pips=row["max_ema50_slope_pips"],
    )


def _independent_signal_pnl_pips(
    market: pd.DataFrame,
    signals: pd.DataFrame,
    params: DecisionParams,
    costs: CostConfig,
) -> pd.DataFrame:
    if market.empty or signals.empty:
        return pd.DataFrame(columns=["target_net_pips"])

    market = market.sort_index()
    signals = signals.sort_index()
    index = pd.DatetimeIndex(market.index)
    pos_by_time = pd.Series(np.arange(len(index)), index=index)
    entry_pos = pos_by_time.reindex(signals.index).dropna().astype(int)
    if entry_pos.empty:
        return pd.DataFrame(columns=["target_net_pips"])

    signals = signals.loc[entry_pos.index].copy()
    signals["entry_pos"] = entry_pos.to_numpy()

    open_arr = market["open"].to_numpy(dtype=float)
    high_arr = market["high"].to_numpy(dtype=float)
    low_arr = market["low"].to_numpy(dtype=float)
    close_arr = market["close"].to_numpy(dtype=float)
    pip_arr = market["pip_size"].to_numpy(dtype=float)

    rows = []
    for row in signals.itertuples():
        i = int(row.entry_pos)
        entry_price = open_arr[i]
        signal_atr = float(row.signal_atr)
        pip_size = pip_arr[i]
        if (
            not np.isfinite(entry_price)
            or not np.isfinite(signal_atr)
            or not np.isfinite(pip_size)
            or entry_price <= 0.0
            or signal_atr <= 0.0
            or pip_size <= 0.0
        ):
            continue

        stop_price = entry_price - signal_atr * params.stop_atr
        take_price = entry_price + signal_atr * params.take_atr
        exit_i = min(len(market) - 1, i + params.max_hold_bars - 1)
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
        gross_pips = (exit_price - entry_price) / pip_size
        rows.append(
            {
                "entry_time": row.Index,
                "target_net_pips": float(gross_pips - costs.round_trip_pips),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["target_net_pips"])
    return pd.DataFrame(rows).set_index("entry_time").sort_index()


def _training_frame(
    signals: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    if signals.empty or labels.empty:
        return pd.DataFrame()
    aligned_signals = signals.reindex(labels.index)
    source_index = pd.DatetimeIndex(aligned_signals["source_time"])
    source_features = features.reindex(source_index)[feature_cols].copy()
    source_features.index = labels.index
    frame = source_features.join(labels, how="inner")
    return frame.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols + ["target_net_pips"])


def _quantile_grid(max_quantile: float, step: float) -> list[float]:
    count = int(np.floor(max_quantile / step)) + 1
    values = [round(i * step, 6) for i in range(count)]
    if 0.0 not in values:
        values.insert(0, 0.0)
    return values


def _filter_by_prediction(signals: pd.DataFrame, predictions: pd.Series, threshold: float) -> pd.DataFrame:
    if signals.empty:
        return signals.copy()
    if threshold == -float("inf"):
        return signals.copy()
    keep_index = predictions.index[predictions >= threshold]
    return signals.loc[signals.index.intersection(keep_index)].copy()


def _split_signals_in_half(signals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    signals = signals.sort_index()
    split_at = len(signals) // 2
    return signals.iloc[:split_at].copy(), signals.iloc[split_at:].copy()


def _selection_metrics(selected: pd.DataFrame) -> dict:
    if selected.empty:
        return {}
    metrics = {
        "folds": int(selected["fold"].nunique()),
        "eligible_folds": int((selected["selection_status"] == "eligible_validation_row").sum()),
        "min_validation_net": float(selected["val_net_profit"].min()),
        "min_test_net": float(selected["test_net_profit"].min()),
        "validation_net_sum": float(selected["val_net_profit"].sum()),
        "test_net_sum": float(selected["test_net_profit"].sum()),
        "min_validation_trades": int(selected["val_trade_count"].min()),
        "min_test_trades": int(selected["test_trade_count"].min()),
        "max_validation_drawdown": float(selected["val_max_drawdown"].max()),
        "max_test_drawdown": float(selected["test_max_drawdown"].max()),
    }
    optional = [
        "val_month_positive_ratio",
        "test_month_positive_ratio",
        "val_quarter_positive_ratio",
        "test_quarter_positive_ratio",
        "val_min_month_pnl",
        "test_min_month_pnl",
        "val_min_quarter_pnl",
        "test_min_quarter_pnl",
    ]
    for col in optional:
        if col in selected.columns:
            if col.endswith("_positive_ratio"):
                metrics[f"min_{col}"] = float(selected[col].min())
            else:
                metrics[f"min_{col}"] = float(selected[col].min())
    return metrics


def _period_stats(trades: pd.DataFrame, freq: str) -> dict:
    prefix = "month" if freq == "M" else "quarter"
    if trades.empty:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_positive_count": 0,
            f"{prefix}_positive_ratio": 0.0,
            f"min_{prefix}_pnl": 0.0,
            f"max_{prefix}_pnl": 0.0,
        }
    frame = trades.copy()
    frame["entry_time"] = pd.to_datetime(frame["entry_time"])
    period_pnl = frame.groupby(frame["entry_time"].dt.to_period(freq))["pnl"].sum()
    positive_count = int((period_pnl > 0.0).sum())
    count = int(len(period_pnl))
    return {
        f"{prefix}_count": count,
        f"{prefix}_positive_count": positive_count,
        f"{prefix}_positive_ratio": float(positive_count / count) if count else 0.0,
        f"min_{prefix}_pnl": float(period_pnl.min()) if count else 0.0,
        f"max_{prefix}_pnl": float(period_pnl.max()) if count else 0.0,
    }


def _split_period_metrics(result) -> dict:
    out = {}
    for freq in ["M", "Q"]:
        out.update(_period_stats(result.trades, freq))
    return out


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir) / datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    row = _stable_row()
    param_id = _param_id(row)
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
    params = _params_from_row(row, args.risk_per_trade)
    folds = _resolve_folds(args)
    feature_cols = _feature_cols(args.feature_set)

    selected_rows = []
    all_trials = []
    coefficient_rows = []

    for fold in folds:
        market = _load_market(fold["years"], row["timeframe"], args.pair)
        features = _add_meta_features(market)
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
        signals = _delay_signals_to_next_bar(raw_signals, market)
        labels = _independent_signal_pnl_pips(market, signals, params, costs)
        frame = _training_frame(signals, features, labels, feature_cols)
        train = _slice(frame, fold["train_start"], fold["train_end"])
        if len(train) < args.min_train_signals:
            raise RuntimeError(f"{fold['name']} has only {len(train)} train signals")

        model = make_pipeline(StandardScaler(), Ridge(alpha=args.ridge_alpha))
        model.fit(train[feature_cols], train["target_net_pips"])
        ridge = model.named_steps["ridge"]
        scaler = model.named_steps["standardscaler"]
        for feature, coef, mean, scale in zip(feature_cols, ridge.coef_, scaler.mean_, scaler.scale_):
            coefficient_rows.append(
                {
                    "fold": fold["name"],
                    "feature": feature,
                    "coefficient": float(coef),
                    "scaler_mean": float(mean),
                    "scaler_scale": float(scale),
                }
            )

        split_predictions = {}
        split_signals = {}
        split_market = {}
        for split in ["val", "test"]:
            split_market[split] = _slice(market, fold[f"{split}_start"], fold[f"{split}_end"])
            split_signals[split] = _slice(signals, fold[f"{split}_start"], fold[f"{split}_end"])
            split_frame = _slice(frame, fold[f"{split}_start"], fold[f"{split}_end"])
            if split_frame.empty:
                split_predictions[split] = pd.Series(dtype=float)
            else:
                split_predictions[split] = pd.Series(
                    model.predict(split_frame[feature_cols]),
                    index=split_frame.index,
                    name="predicted_net_pips",
                )

        trials = []
        val_preds = split_predictions["val"]
        baseline_val_result = simulate(split_market["val"], split_signals["val"], params, costs, risk)
        val_early_signals, val_late_signals = _split_signals_in_half(split_signals["val"])
        min_half_trades = max(5, int(math.ceil(args.min_trades / 2)))
        thresholds = []
        if not val_preds.empty:
            thresholds.append((-1.0, -float("inf")))
            for quantile in _quantile_grid(args.max_quantile, args.quantile_step):
                thresholds.append((quantile, float(val_preds.quantile(quantile))))
        if not thresholds:
            thresholds = [(0.0, -float("inf"))]

        for quantile, threshold in thresholds:
            filtered_val = _filter_by_prediction(split_signals["val"], split_predictions["val"], threshold)
            filtered_test = _filter_by_prediction(split_signals["test"], split_predictions["test"], threshold)
            val_result = simulate(split_market["val"], filtered_val, params, costs, risk)
            test_result = simulate(split_market["test"], filtered_test, params, costs, risk)
            early_result = simulate(
                split_market["val"],
                val_early_signals.loc[val_early_signals.index.intersection(filtered_val.index)],
                params,
                costs,
                risk,
            )
            late_result = simulate(
                split_market["val"],
                val_late_signals.loc[val_late_signals.index.intersection(filtered_val.index)],
                params,
                costs,
                risk,
            )
            val_metrics = val_result.metrics
            eligible = (
                val_metrics["net_profit"] > 0.0
                and val_metrics["trade_count"] >= optimizer_cfg.min_trades
                and val_metrics["max_drawdown"] <= optimizer_cfg.max_acceptable_drawdown
            )
            score = (
                val_metrics["net_profit"]
                + args.trade_bonus * val_metrics["trade_count"]
                - args.drawdown_penalty * val_metrics["max_drawdown"]
            )
            if args.selection_mode == "safe_fallback" and quantile >= 0.0:
                min_kept_trades = int(math.ceil(baseline_val_result.metrics["trade_count"] * args.min_keep_ratio))
                safe = (
                    val_metrics["net_profit"] > baseline_val_result.metrics["net_profit"]
                    and val_metrics["trade_count"] >= min_kept_trades
                    and early_result.metrics["net_profit"] > 0.0
                    and late_result.metrics["net_profit"] > 0.0
                    and early_result.metrics["trade_count"] >= min_half_trades
                    and late_result.metrics["trade_count"] >= min_half_trades
                )
                if not safe:
                    score -= 1_000_000.0
            if args.selection_mode == "light_filter" and quantile >= 0.0:
                min_kept_trades = int(math.ceil(baseline_val_result.metrics["trade_count"] * args.min_keep_ratio))
                light = (
                    quantile > 0.0
                    and quantile <= args.max_light_quantile
                    and val_metrics["net_profit"] >= baseline_val_result.metrics["net_profit"] - args.baseline_tolerance
                    and val_metrics["trade_count"] >= min_kept_trades
                )
                if light:
                    score += args.filter_bonus
                else:
                    score -= 1_000_000.0
            val_period_metrics = _split_period_metrics(val_result)
            test_period_metrics = _split_period_metrics(test_result)
            if args.selection_mode == "stable_light_filter" and quantile >= 0.0:
                min_kept_trades = int(math.ceil(baseline_val_result.metrics["trade_count"] * args.min_keep_ratio))
                stable_light = (
                    quantile > 0.0
                    and quantile <= args.max_light_quantile
                    and val_metrics["net_profit"] >= baseline_val_result.metrics["net_profit"] - args.baseline_tolerance
                    and val_metrics["trade_count"] >= min_kept_trades
                    and val_period_metrics["month_positive_ratio"] >= args.min_val_month_positive_ratio
                    and val_period_metrics["quarter_positive_ratio"] >= args.min_val_quarter_positive_ratio
                    and val_period_metrics["min_month_pnl"] >= args.min_val_month_pnl
                    and val_period_metrics["min_quarter_pnl"] >= args.min_val_quarter_pnl
                )
                if stable_light:
                    score += (
                        args.filter_bonus
                        + args.period_stability_bonus * val_period_metrics["month_positive_ratio"]
                        + args.period_stability_bonus * val_period_metrics["quarter_positive_ratio"]
                        + args.min_period_pnl_bonus * val_period_metrics["min_month_pnl"]
                        + args.min_period_pnl_bonus * val_period_metrics["min_quarter_pnl"]
                    )
                else:
                    score -= 1_000_000.0
            if not eligible:
                score -= 1_000_000.0
            trial = {
                "fold": fold["name"],
                "param_id": param_id,
                "quantile": quantile,
                "threshold": threshold,
                "score": float(score),
                "selection_status": "eligible_validation_row" if eligible else "ineligible_validation_row",
                "selection_mode": args.selection_mode,
                **row,
                **{f"val_{k}": v for k, v in _flat_metrics(val_result.metrics).items()},
                **{f"val_{k}": v for k, v in val_period_metrics.items()},
                **{f"val_early_{k}": v for k, v in _flat_metrics(early_result.metrics).items()},
                **{f"val_late_{k}": v for k, v in _flat_metrics(late_result.metrics).items()},
                **{f"test_{k}": v for k, v in _flat_metrics(test_result.metrics).items()},
                **{f"test_{k}": v for k, v in test_period_metrics.items()},
            }
            trials.append((trial, val_result, test_result, filtered_val, filtered_test))
            all_trials.append(trial)

        selected_trial, selected_val, selected_test, _, _ = max(trials, key=lambda item: item[0]["score"])
        selected_rows.append(selected_trial)
        fold_out = out_dir / fold["name"] / f"{param_id}_ridge_q{selected_trial['quantile']:g}"
        summary = {
            "fold": fold["name"],
            "pair": args.pair,
            "param_id": param_id,
            "model": "ridge_independent_signal_pnl",
            "costs": asdict(costs),
            "risk": asdict(risk),
            "row": row,
            "ridge_alpha": args.ridge_alpha,
            "selected": selected_trial,
            "validation_metrics": selected_val.metrics,
            "test_metrics": selected_test.metrics,
        }
        summary = _clean(summary)
        write_backtest_outputs(fold_out, selected_test, summary, trials=pd.DataFrame([t[0] for t in trials]))
        write_backtest_outputs(fold_out / "validation_best", selected_val, summary)

    selected_df = pd.DataFrame(selected_rows).sort_values("fold")
    trials_df = pd.DataFrame(all_trials).sort_values(["fold", "score"], ascending=[True, False])
    selected_df.to_csv(out_dir / "selected_by_fold.csv", index=False)
    trials_df.to_csv(out_dir / "trials.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(out_dir / "ridge_coefficients.csv", index=False)
    (out_dir / "selection_summary.json").write_text(
        json.dumps(_clean(_selection_metrics(selected_df)), indent=2, allow_nan=False)
    )
    (out_dir / "run_config.json").write_text(
        json.dumps(
            _clean(
                {
                    "args": vars(args),
                    "row": row,
                    "costs": asdict(costs),
                    "risk": asdict(risk),
                    "param_id": param_id,
                    "feature_cols": feature_cols,
                }
            ),
            indent=2,
            allow_nan=False,
        )
    )
    print(f"Ridge PnL meta-filter outputs written to: {out_dir}")
    print(
        selected_df[
            [
                "fold",
                "quantile",
                "val_net_profit",
                "test_net_profit",
                "val_trade_count",
                "test_trade_count",
                "selection_status",
            ]
        ].to_string(index=False)
    )
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No-leakage ridge PnL meta-filter for the stable H1 baseline.")
    parser.add_argument("--output-dir", default="outputs/pnl_meta_filter_runs")
    parser.add_argument("--folds", default="all")
    parser.add_argument("--pair", default="EURUSD")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--feature-set", choices=["all", "no_calendar", "source"], default="all")
    parser.add_argument("--min-trades", type=int, default=20)
    parser.add_argument("--min-train-signals", type=int, default=100)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--max-quantile", type=float, default=0.90)
    parser.add_argument("--quantile-step", type=float, default=0.05)
    parser.add_argument(
        "--selection-mode",
        choices=["standard", "safe_fallback", "light_filter", "stable_light_filter"],
        default="standard",
    )
    parser.add_argument("--min-keep-ratio", type=float, default=0.75)
    parser.add_argument("--max-light-quantile", type=float, default=0.05)
    parser.add_argument("--baseline-tolerance", type=float, default=1e-9)
    parser.add_argument("--filter-bonus", type=float, default=0.001)
    parser.add_argument("--min-val-month-positive-ratio", type=float, default=0.0)
    parser.add_argument("--min-val-quarter-positive-ratio", type=float, default=0.0)
    parser.add_argument("--min-val-month-pnl", type=float, default=-float("inf"))
    parser.add_argument("--min-val-quarter-pnl", type=float, default=-float("inf"))
    parser.add_argument("--period-stability-bonus", type=float, default=0.0)
    parser.add_argument("--min-period-pnl-bonus", type=float, default=0.0)
    parser.add_argument("--trade-bonus", type=float, default=5.0)
    parser.add_argument("--drawdown-penalty", type=float, default=1000.0)
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
