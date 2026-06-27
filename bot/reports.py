from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd

from .simulator import BacktestResult


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def write_backtest_outputs(
    out_dir: Path,
    result: BacktestResult,
    summary: dict,
    trials: pd.DataFrame | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result.trades.to_csv(out_dir / "trades.csv", index=False)
    result.equity_curve.to_csv(out_dir / "equity_curve.csv")
    if trials is not None:
        trials.to_csv(out_dir / "trials.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default, allow_nan=False)
    )
    write_charts(out_dir, result)


def write_charts(out_dir: Path, result: BacktestResult) -> None:
    if result.equity_curve.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    result.equity_curve["equity"].plot(ax=ax)
    ax.set_title("Equity Curve")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    fig.tight_layout()
    fig.savefig(out_dir / "equity_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    (-result.equity_curve["drawdown"] * 100).plot(ax=ax, color="crimson")
    ax.set_title("Drawdown")
    ax.set_xlabel("Time")
    ax.set_ylabel("Drawdown (%)")
    fig.tight_layout()
    fig.savefig(out_dir / "drawdown.png", dpi=150)
    plt.close(fig)
