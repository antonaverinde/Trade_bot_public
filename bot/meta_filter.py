from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .baseline_strategies import (
    _add_signal_features,
    _grid,
    _load_market,
    _param_id,
    _signals_from_rule,
    _slice,
)
from .config import CostConfig, DecisionParams, OptimizerConfig, RiskConfig
from .model_adapters import _next_bar_times
from .optimize import _flat_metrics, objective_score
from .reports import write_backtest_outputs
from .simulator import simulate
from .walkforward import FOLDS, _select_folds


FEATURE_COLS = [
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
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]


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


def _add_meta_features(market: pd.DataFrame) -> pd.DataFrame:
    frame = _add_signal_features(market)
    close = frame["close"]
    pip_size = frame["pip_size"]
    returns = close.diff() / pip_size
    frame["ret_1_pips"] = returns
    frame["ret_4_pips"] = (close - close.shift(4)) / pip_size
    frame["ret_24_pips"] = (close - close.shift(24)) / pip_size
    frame["vol_24_pips"] = returns.rolling(24, min_periods=12).std(ddof=0)
    abs_path = returns.abs().rolling(24, min_periods=12).sum()
    direct_path = ((close - close.shift(24)) / pip_size).abs()
    frame["trend_eff_24"] = direct_path / abs_path.replace(0.0, np.nan)
    frame["return_autocorr_24"] = returns.rolling(24, min_periods=12).corr(returns.shift(1))
    idx = pd.DatetimeIndex(frame.index)
    frame["hour_sin"] = np.sin(2.0 * np.pi * idx.hour / 24.0)
    frame["hour_cos"] = np.cos(2.0 * np.pi * idx.hour / 24.0)
    frame["month_sin"] = np.sin(2.0 * np.pi * (idx.month - 1) / 12.0)
    frame["month_cos"] = np.cos(2.0 * np.pi * (idx.month - 1) / 12.0)
    return frame.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS)


def _params_from_row(row: dict, args: argparse.Namespace) -> DecisionParams:
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


def _training_frame(trades: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    frame = trades.copy()
    frame["entry_time"] = pd.to_datetime(frame["entry_time"])
    if "source_time" in frame.columns:
        frame["feature_time"] = pd.to_datetime(frame["source_time"])
    else:
        frame["feature_time"] = frame["entry_time"]
    frame = frame.set_index("feature_time", drop=False)
    joined = frame.join(features[FEATURE_COLS], how="inner")
    joined["target_win"] = (joined["pnl"] > 0.0).astype(int)
    return joined.dropna(subset=FEATURE_COLS + ["target_win"])


def _delay_signals_to_next_bar(signals: pd.DataFrame, market_index: pd.DatetimeIndex) -> pd.DataFrame:
    if signals.empty:
        return signals.copy()
    source_times = pd.DatetimeIndex(signals.index)
    entry_times, valid = _next_bar_times(market_index, source_times)
    delayed = signals.iloc[np.flatnonzero(valid)].copy()
    delayed["source_time"] = source_times[valid]
    delayed.index = entry_times
    delayed.index.name = "entry_time"
    return delayed.sort_index()


def _attach_source_market_values(signals: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    if signals.empty or "source_time" not in signals.columns:
        return signals
    out = signals.copy()
    source = features.reindex(pd.DatetimeIndex(out["source_time"]))
    if "atr" in source.columns:
        out["signal_atr"] = source["atr"].to_numpy()
    return out


def _fit_model(train_rows: pd.DataFrame, seed: int):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.5,
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
        ),
    )
    model.fit(train_rows[FEATURE_COLS].to_numpy(), train_rows["target_win"].to_numpy())
    return model


def _score_signals(signals: pd.DataFrame, features: pd.DataFrame, model) -> pd.DataFrame:
    if signals.empty:
        out = signals.copy()
        out["meta_win_prob"] = []
        return out
    feature_index = (
        pd.DatetimeIndex(signals["source_time"])
        if "source_time" in signals.columns
        else pd.DatetimeIndex(signals.index)
    )
    aligned = features.reindex(feature_index)
    valid = aligned[FEATURE_COLS].notna().all(axis=1).to_numpy()
    out = signals.iloc[np.flatnonzero(valid)].copy()
    if out.empty:
        out["meta_win_prob"] = []
        return out
    probs = model.predict_proba(aligned.loc[valid, FEATURE_COLS].to_numpy())[:, 1]
    out["meta_win_prob"] = probs
    return out


def _thresholds(min_prob: float, max_prob: float, step: float) -> list[float]:
    count = int(np.floor((max_prob - min_prob) / step)) + 1
    return [round(min_prob + i * step, 6) for i in range(max(0, count))]


def _split_signals_in_half(signals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    signals = signals.sort_index()
    split_at = len(signals) // 2
    return signals.iloc[:split_at].copy(), signals.iloc[split_at:].copy()


def _safe_selection_score(
    row: dict,
    optimizer_cfg: OptimizerConfig,
    min_half_trades: int,
    min_keep_ratio: float,
) -> float:
    threshold = float(row["threshold"])
    val_net = float(row["val_net_profit"])
    val_trades = int(row["val_trade_count"])
    val_dd = float(row["val_max_drawdown"])
    if threshold <= 0:
        return objective_score(
            {
                "net_profit": val_net,
                "trade_count": val_trades,
                "max_drawdown": val_dd,
                "sortino": row.get("val_sortino", 0.0),
                "sharpe": row.get("val_sharpe", 0.0),
                "total_return": row.get("val_total_return", 0.0),
                "calmar": row.get("val_calmar", 0.0),
                "profit_factor": row.get("val_profit_factor", 0.0),
            },
            optimizer_cfg,
        )

    base_val_net = float(row["base_val_net_profit"])
    base_val_trades = int(row["base_val_trade_count"])
    early_net = float(row["val_early_net_profit"])
    late_net = float(row["val_late_net_profit"])
    early_trades = int(row["val_early_trade_count"])
    late_trades = int(row["val_late_trade_count"])
    min_kept_trades = int(math.ceil(base_val_trades * min_keep_ratio))
    if (
        val_net <= base_val_net
        or val_net <= 0.0
        or val_trades < optimizer_cfg.min_trades
        or val_trades < min_kept_trades
        or val_dd > optimizer_cfg.max_acceptable_drawdown
        or early_net <= 0.0
        or late_net <= 0.0
        or early_trades < min_half_trades
        or late_trades < min_half_trades
    ):
        return -1_000_000.0
    return objective_score(
        {
            "net_profit": val_net,
            "trade_count": val_trades,
            "max_drawdown": val_dd,
            "sortino": row.get("val_sortino", 0.0),
            "sharpe": row.get("val_sharpe", 0.0),
            "total_return": row.get("val_total_return", 0.0),
            "calmar": row.get("val_calmar", 0.0),
            "profit_factor": row.get("val_profit_factor", 0.0),
        },
        optimizer_cfg,
    )


def _selection_metrics(selected: pd.DataFrame) -> dict:
    if selected.empty:
        return {}
    return {
        "folds": int(selected["fold"].nunique()),
        "baseline_fallback_folds": int((selected["selection_kind"] == "baseline").sum()),
        "meta_filter_folds": int((selected["selection_kind"] == "meta_filter").sum()),
        "min_validation_net": float(selected["val_net_profit"].min()),
        "min_test_net": float(selected["test_net_profit"].min()),
        "validation_net_sum": float(selected["val_net_profit"].sum()),
        "test_net_sum": float(selected["test_net_profit"].sum()),
        "min_validation_trades": float(selected["val_trade_count"].min()),
        "min_test_trades": float(selected["test_trade_count"].min()),
        "max_validation_drawdown": float(selected["val_max_drawdown"].max()),
        "max_test_drawdown": float(selected["test_max_drawdown"].max()),
    }


def _coefficient_rows(model, fold_name: str, param_id: str) -> list[dict]:
    lr = model.named_steps["logisticregression"]
    return [
        {
            "fold": fold_name,
            "param_id": param_id,
            "feature": feature,
            "coefficient": float(coef),
        }
        for feature, coef in zip(FEATURE_COLS, lr.coef_[0], strict=True)
    ]


def run_meta_filter(args: argparse.Namespace) -> Path:
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
        trials=0,
        seed=args.seed,
        min_trades=args.min_trades,
        max_acceptable_drawdown=args.max_drawdown,
    )
    min_half_trades = max(1, int(math.ceil(args.min_trades / 2)))

    out_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = _select_folds(args.folds)
    grid = _grid(args.profile)
    if args.max_rows > 0:
        grid = grid[: args.max_rows]

    market_cache: dict[tuple, pd.DataFrame] = {}
    feature_cache: dict[tuple, pd.DataFrame] = {}
    trials: list[dict] = []
    summaries: list[dict] = []
    coef_rows: list[dict] = []
    best_by_fold: dict[str, tuple[float, dict, float, object, object, object, object]] = {}

    for fold in folds:
        for row_i, row in enumerate(grid):
            cache_key = (tuple(fold["years"]), row["timeframe"], args.pair)
            if cache_key not in market_cache:
                market_cache[cache_key] = _load_market(fold["years"], row["timeframe"], args.pair)
                feature_cache[cache_key] = _add_meta_features(market_cache[cache_key])
            market = market_cache[cache_key]
            features = feature_cache[cache_key]
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
            signals = _delay_signals_to_next_bar(signals, pd.DatetimeIndex(market.index))
            signals = _attach_source_market_values(signals, features)
            params = _params_from_row(row, args)
            param_id = _param_id(row)

            train_result = simulate(
                _slice(market, fold["train_start"], fold["train_end"]),
                _slice(signals, fold["train_start"], fold["train_end"]),
                params,
                costs,
                risk,
            )
            train_rows = _training_frame(train_result.trades, features)
            if len(train_rows) < args.min_train_trades or train_rows["target_win"].nunique() < 2:
                continue

            model = _fit_model(train_rows, args.seed + row_i)
            coef_rows.extend(_coefficient_rows(model, fold["name"], param_id))

            split_signals = {
                split: _score_signals(
                    _slice(signals, fold[f"{split}_start"], fold[f"{split}_end"]),
                    features,
                    model,
                )
                for split in ["val", "test"]
            }
            split_market = {
                split: _slice(market, fold[f"{split}_start"], fold[f"{split}_end"])
                for split in ["val", "test"]
            }
            base_results = {
                split: simulate(
                    split_market[split],
                    _slice(signals, fold[f"{split}_start"], fold[f"{split}_end"]),
                    params,
                    costs,
                    risk,
                )
                for split in ["val", "test"]
            }
            val_early_signals, val_late_signals = _split_signals_in_half(split_signals["val"])

            threshold_rows: list[tuple[float, pd.DataFrame, pd.DataFrame, str]] = [
                (0.0, _slice(signals, fold["val_start"], fold["val_end"]), _slice(signals, fold["test_start"], fold["test_end"]), "baseline")
            ]
            for threshold in _thresholds(args.min_prob, args.max_prob, args.prob_step):
                threshold_rows.append(
                    (
                        threshold,
                        split_signals["val"].loc[split_signals["val"]["meta_win_prob"] >= threshold],
                        split_signals["test"].loc[split_signals["test"]["meta_win_prob"] >= threshold],
                        "meta_filter",
                    )
                )

            for threshold, filtered_val, filtered_test, selection_kind in threshold_rows:
                val_result = simulate(split_market["val"], filtered_val, params, costs, risk)
                score = objective_score(val_result.metrics, optimizer_cfg)
                test_result = simulate(split_market["test"], filtered_test, params, costs, risk)
                if threshold <= 0:
                    val_early_result = simulate(split_market["val"], val_early_signals, params, costs, risk)
                    val_late_result = simulate(split_market["val"], val_late_signals, params, costs, risk)
                else:
                    val_early_result = simulate(
                        split_market["val"],
                        val_early_signals.loc[val_early_signals["meta_win_prob"] >= threshold],
                        params,
                        costs,
                        risk,
                    )
                    val_late_result = simulate(
                        split_market["val"],
                        val_late_signals.loc[val_late_signals["meta_win_prob"] >= threshold],
                        params,
                        costs,
                        risk,
                    )
                trial = {
                    "fold": fold["name"],
                    "row": row_i,
                    "param_id": param_id,
                    "threshold": threshold,
                    "selection_kind": selection_kind,
                    "score": score,
                    "train_trades": int(len(train_rows)),
                    "train_win_rate": float(train_rows["target_win"].mean()),
                    "val_candidate_signals": int(len(split_signals["val"])),
                    "test_candidate_signals": int(len(split_signals["test"])),
                    "val_kept_signals": int(len(filtered_val)),
                    "test_kept_signals": int(len(filtered_test)),
                    **row,
                    **{f"base_val_{k}": v for k, v in _flat_metrics(base_results["val"].metrics).items()},
                    **{f"base_test_{k}": v for k, v in _flat_metrics(base_results["test"].metrics).items()},
                    **{f"val_{k}": v for k, v in _flat_metrics(val_result.metrics).items()},
                    **{f"val_early_{k}": v for k, v in _flat_metrics(val_early_result.metrics).items()},
                    **{f"val_late_{k}": v for k, v in _flat_metrics(val_late_result.metrics).items()},
                    **{f"test_{k}": v for k, v in _flat_metrics(test_result.metrics).items()},
                }
                trial["safe_score"] = _safe_selection_score(
                    trial,
                    optimizer_cfg,
                    min_half_trades,
                    args.min_keep_ratio,
                )
                trials.append(trial)
                current_best = best_by_fold.get(fold["name"])
                selection_score = trial["safe_score"] if args.selection_mode == "safe_fallback" else score
                if current_best is None or selection_score > current_best[0]:
                    best_by_fold[fold["name"]] = (
                        selection_score,
                        row,
                        threshold,
                        model,
                        val_result,
                        test_result,
                        train_result,
                    )

    trials_df = pd.DataFrame(trials)
    if not trials_df.empty:
        trials_df = trials_df.sort_values(["fold", "score"], ascending=[True, False])
    trials_df.to_csv(out_dir / "trials.csv", index=False)
    pd.DataFrame(coef_rows).to_csv(out_dir / "coefficients.csv", index=False)

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
                base_val_net_min=("base_val_net_profit", "min"),
                base_test_net_min=("base_test_net_profit", "min"),
            )
            .reset_index()
            .sort_values(["test_net_min", "val_net_min"], ascending=[False, False])
        )
        aggregate.to_csv(out_dir / "param_summary.csv", index=False)
        score_col = "safe_score" if args.selection_mode == "safe_fallback" else "score"
        selected = trials_df.sort_values(["fold", score_col], ascending=[True, False]).groupby("fold").head(1)
        selected.to_csv(out_dir / "selected_by_fold.csv", index=False)
        (out_dir / "selection_summary.json").write_text(
            json.dumps(_clean(_selection_metrics(selected)), indent=2, allow_nan=False)
        )

    for fold_name, (score, row, threshold, model, val_result, test_result, train_result) in best_by_fold.items():
        param_id = _param_id(row)
        summary = _clean(
            {
                "fold": fold_name,
                "param_id": param_id,
                "threshold": threshold,
                "score": score,
                "selection_mode": args.selection_mode,
                "params": row,
                "feature_cols": FEATURE_COLS,
                "train_metrics": _flat_metrics(train_result.metrics),
                "validation_metrics": _flat_metrics(val_result.metrics),
                "test_metrics": _flat_metrics(test_result.metrics),
                "costs": asdict(costs),
                "risk": asdict(risk),
                "optimizer": asdict(optimizer_cfg),
            }
        )
        fold_out = out_dir / fold_name / f"{param_id}_meta{threshold:g}"
        write_backtest_outputs(fold_out, test_result, summary, trials=trials_df)
        write_backtest_outputs(fold_out / "validation_best", val_result, summary)
        summaries.append(summary)

    (out_dir / "summary.json").write_text(json.dumps(_clean(summaries), indent=2, allow_nan=False))

    print(f"Meta-filter outputs written to: {out_dir}")
    if trials_df.empty:
        print("No valid meta-filter rows produced.")
        return out_dir
    display_cols = [
        "fold",
        "param_id",
        "threshold",
        "selection_kind",
        "safe_score",
        "train_trades",
        "val_net_profit",
        "test_net_profit",
        "val_trade_count",
        "test_trade_count",
        "val_max_drawdown",
        "test_max_drawdown",
        "score",
    ]
    print(trials_df[display_cols].head(args.print_top).to_string(index=False))
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Meta-filter raw baseline candidate trades.")
    parser.add_argument("--output-dir", default="outputs/meta_filter_runs")
    parser.add_argument("--folds", default="all")
    parser.add_argument("--pair", default="EURUSD")
    parser.add_argument(
        "--profile",
        choices=[
            "quick",
            "full",
            "adaptive",
            "fade",
            "rsi",
            "mr_long",
            "mr_long_refine",
            "mr_long_source_gate",
            "mr_long_stable_fixed",
        ],
        default="adaptive",
    )
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--print-top", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--min-train-trades", type=int, default=80)
    parser.add_argument("--min-prob", type=float, default=0.50)
    parser.add_argument("--max-prob", type=float, default=0.90)
    parser.add_argument("--prob-step", type=float, default=0.05)
    parser.add_argument("--selection-mode", choices=["standard", "safe_fallback"], default="standard")
    parser.add_argument("--min-keep-ratio", type=float, default=0.75)
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
    run_meta_filter(parse_args())


if __name__ == "__main__":
    main()
