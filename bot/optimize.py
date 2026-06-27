from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CostConfig, DecisionParams, OptimizerConfig, RiskConfig
from .model_adapters import SignalDataset, build_signal_dataset
from .reports import write_backtest_outputs
from .simulator import BacktestResult, simulate


def sample_params(rng: np.random.Generator, level_mode: str, trade_side: str) -> DecisionParams:
    if level_mode == "model":
        stop_range = (0.75, 1.50)
        take_range = (0.75, 1.75)
    else:
        stop_range = (0.5, 3.0)
        take_range = (0.7, 5.0)
    return DecisionParams(
        level_mode=level_mode,
        trade_side=trade_side,
        entry_threshold=float(rng.uniform(0.48, 0.76)),
        exit_threshold=float(rng.uniform(0.45, 0.68)),
        exit_floor=float(rng.uniform(0.20, 0.48)),
        min_conf_gap=float(rng.uniform(0.02, 0.25)),
        min_edge_pips=float(rng.uniform(0.0, 3.0)),
        stop_atr=float(rng.uniform(*stop_range)),
        take_atr=float(rng.uniform(*take_range)),
        max_hold_bars=int(rng.integers(2, 49)),
        cooldown_bars=int(rng.integers(0, 17)),
        risk_per_trade=float(rng.uniform(0.0025, 0.02)),
    )


def objective_score(metrics: dict, cfg: OptimizerConfig) -> float:
    trade_count = int(metrics.get("trade_count", 0))
    if trade_count == 0:
        return -1_000_000.0

    sortino = _finite(metrics.get("sortino", 0.0))
    sharpe = _finite(metrics.get("sharpe", 0.0))
    total_return = _finite(metrics.get("total_return", 0.0))
    calmar = _finite(metrics.get("calmar", 0.0))
    profit_factor = _finite(metrics.get("profit_factor", 0.0))
    max_drawdown = max(0.0, _finite(metrics.get("max_drawdown", 0.0)))

    score = (
        1.25 * sortino
        + 0.75 * sharpe
        + 0.50 * calmar
        + 0.25 * min(profit_factor, 3.0)
        + 2.00 * total_return
        - 2.00 * max_drawdown
    )
    if max_drawdown > cfg.max_acceptable_drawdown:
        score -= 25.0 * (max_drawdown - cfg.max_acceptable_drawdown)
    if trade_count < cfg.min_trades:
        score -= (cfg.min_trades - trade_count) / cfg.min_trades
    if metrics.get("net_profit", 0.0) <= 0:
        score -= 1.0
    return float(score)


def _finite(value: object) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _flat_metrics(metrics: dict) -> dict:
    return {
        key: _finite(value) if isinstance(value, (int, float, np.number)) else value
        for key, value in metrics.items()
        if key not in {"params", "risk", "costs"}
    }


def run_optimization(
    dataset: SignalDataset,
    optimizer_cfg: OptimizerConfig,
    costs: CostConfig,
    risk: RiskConfig,
    level_mode: str,
    trade_side: str,
) -> tuple[pd.DataFrame, DecisionParams, BacktestResult, BacktestResult]:
    rng = np.random.default_rng(optimizer_cfg.seed)
    rows = []
    best_score = -float("inf")
    best_params = DecisionParams()
    best_val_result: BacktestResult | None = None

    val_market = dataset.split_market["val"]
    val_signals = dataset.split_signals["val"]
    if val_market.empty or val_signals.empty:
        raise ValueError(f"{dataset.model_name} has no validation market/signals to optimize.")

    for trial in range(optimizer_cfg.trials):
        params = sample_params(rng, level_mode=level_mode, trade_side=trade_side)
        result = simulate(val_market, val_signals, params, costs=costs, risk=risk)
        score = objective_score(result.metrics, optimizer_cfg)
        row = {
            "trial": trial,
            "score": score,
            **params.to_dict(),
            **_flat_metrics(result.metrics),
        }
        rows.append(row)
        if score > best_score:
            best_score = score
            best_params = params
            best_val_result = result

    if best_val_result is None:
        raise RuntimeError("No optimization trials were completed.")

    trials = pd.DataFrame(rows).sort_values("score", ascending=False)
    test_result = simulate(
        dataset.split_market["test"],
        dataset.split_signals["test"],
        best_params,
        costs=costs,
        risk=risk,
    )
    return trials, best_params, best_val_result, test_result


def _clean_json(value):
    if isinstance(value, dict):
        return {k: _clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_json(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run_model(
    model_name: str,
    run_id: str | None,
    args,
    base_out_dir: Path,
) -> dict:
    dataset = build_signal_dataset(model_name, run_id=run_id)
    optimizer_cfg = OptimizerConfig(
        trials=args.trials,
        seed=args.seed,
        min_trades=args.min_trades,
        max_acceptable_drawdown=args.max_drawdown,
    )
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

    trials, best_params, val_result, test_result = run_optimization(
        dataset, optimizer_cfg, costs, risk, args.level_mode, args.trade_side
    )

    summary = {
        "model": model_name,
        "run_id": dataset.run_id,
        "data_cfg": dataset.data_cfg,
        "feature_count": len(dataset.feature_cols),
        "optimizer": asdict(optimizer_cfg),
        "best_params": best_params.to_dict(),
        "validation_metrics": _flat_metrics(val_result.metrics),
        "test_metrics": _flat_metrics(test_result.metrics),
    }
    summary = _clean_json(summary)
    out_dir = base_out_dir / model_name
    write_backtest_outputs(out_dir, test_result, summary, trials=trials)

    # Keep validation outputs too, because these are the values the optimizer saw.
    write_backtest_outputs(out_dir / "validation_best", val_result, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize decision/risk parameters for existing XGBoost trading models."
    )
    parser.add_argument("--model", choices=["h1", "fvg", "both"], default="both")
    parser.add_argument("--h1-run-id", default=None)
    parser.add_argument("--fvg-run-id", default=None)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-trades", type=int, default=20)
    parser.add_argument("--output-dir", default="outputs/bot_runs")
    parser.add_argument("--initial-equity", type=float, default=10_000.0)
    parser.add_argument("--max-leverage", type=float, default=10.0)
    parser.add_argument("--max-drawdown", type=float, default=0.20)
    parser.add_argument("--daily-loss-stop", type=float, default=0.03)
    parser.add_argument("--spread-pips", type=float, default=1.0)
    parser.add_argument("--slippage-pips", type=float, default=0.2)
    parser.add_argument("--commission-pips", type=float, default=0.0)
    parser.add_argument("--level-mode", choices=["model", "atr"], default="model")
    parser.add_argument("--trade-side", choices=["both", "long", "short"], default="both")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out_dir = Path(args.output_dir) / timestamp
    base_out_dir.mkdir(parents=True, exist_ok=True)

    models = ["h1", "fvg"] if args.model == "both" else [args.model]
    summaries = {}
    for model_name in models:
        run_id = args.h1_run_id if model_name == "h1" else args.fvg_run_id
        summaries[model_name] = run_model(model_name, run_id, args, base_out_dir)

    (base_out_dir / "summary.json").write_text(
        json.dumps(_clean_json(summaries), indent=2, allow_nan=False)
    )
    print(f"\nOptimization outputs written to: {base_out_dir}")
    for model_name, summary in summaries.items():
        test = summary["test_metrics"]
        print(
            f"{model_name}: run={summary['run_id']} "
            f"trades={test['trade_count']} net={test['net_profit']:.2f} "
            f"sharpe={test['sharpe']:.3f} sortino={test['sortino']:.3f} "
            f"max_dd={test['max_drawdown']:.2%}"
        )


if __name__ == "__main__":
    main()
