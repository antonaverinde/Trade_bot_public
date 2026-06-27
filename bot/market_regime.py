from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from Pipeline.pipeline import ForexDataLoader

from .baseline_strategies import _resample_ohlc
from .model_adapters import _market_with_indicators
from .walkforward import PROJECT_ROOT


def _load_market(pair: str, years: list[int], timeframe: str) -> pd.DataFrame:
    loader = ForexDataLoader()
    raw = loader.load_and_merge(
        str(PROJECT_ROOT / "histdata"),
        pair=pair,
        years=years,
        weekends="nogap",
    )
    return _market_with_indicators(_resample_ohlc(raw, timeframe), pair)


def _max_drawdown_from_returns(returns: pd.Series) -> float:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    peak = equity.cummax()
    dd = (peak - equity) / peak
    return float(dd.max()) if len(dd) else 0.0


def _period_metrics(market: pd.DataFrame, freq: str) -> pd.DataFrame:
    frame = market.copy()
    close = frame["close"]
    frame["ret"] = close.pct_change()
    frame["abs_ret_pips"] = close.diff().abs() / frame["pip_size"]
    frame["signed_ret_pips"] = close.diff() / frame["pip_size"]
    frame["ema50"] = close.ewm(span=50, adjust=False).mean()
    frame["ema200"] = close.ewm(span=200, adjust=False).mean()
    frame["ema50_slope_pips"] = (frame["ema50"] - frame["ema50"].shift(20)) / frame["pip_size"]
    frame["ema200_dist_pips"] = (close - frame["ema200"]) / frame["pip_size"]
    frame["atr_pips"] = frame["atr"] / frame["pip_size"]
    frame["period"] = pd.PeriodIndex(frame.index, freq=freq).astype(str)

    rows = []
    for period, group in frame.groupby("period"):
        group = group.dropna()
        if len(group) < 10:
            continue
        net_pips = float((group["close"].iloc[-1] - group["close"].iloc[0]) / group["pip_size"].iloc[-1])
        path_pips = float(group["abs_ret_pips"].sum())
        ret = group["ret"].dropna()
        signed = group["signed_ret_pips"].dropna()
        autocorr = float(signed.autocorr(1)) if len(signed) > 2 else 0.0
        rows.append(
            {
                "period": period,
                "start": group.index.min(),
                "end": group.index.max(),
                "bars": int(len(group)),
                "net_pips": net_pips,
                "path_pips": path_pips,
                "trend_efficiency": abs(net_pips) / path_pips if path_pips > 0 else 0.0,
                "mean_atr_pips": float(group["atr_pips"].mean()),
                "median_atr_pips": float(group["atr_pips"].median()),
                "realized_vol_ann": float(ret.std(ddof=0) * np.sqrt(252 * 24)) if len(ret) else 0.0,
                "return_autocorr_1": autocorr if np.isfinite(autocorr) else 0.0,
                "positive_bar_rate": float((signed > 0).mean()) if len(signed) else 0.0,
                "above_ema200_rate": float((group["ema200_dist_pips"] > 0).mean()),
                "mean_ema50_slope_pips": float(group["ema50_slope_pips"].mean()),
                "abs_ema50_slope_pips": float(group["ema50_slope_pips"].abs().mean()),
                "buy_hold_max_dd": _max_drawdown_from_returns(ret),
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> Path:
    years = [int(item) for item in args.years.split(",") if item.strip()]
    market = _load_market(args.pair, years, args.timeframe)
    out_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    for freq, name in [("Y", "year"), ("Q", "quarter"), ("M", "month")]:
        df = _period_metrics(market, freq)
        df.to_csv(out_dir / f"{name}_regimes.csv", index=False)
        if name in {"year", "quarter"}:
            print(f"\n{name.upper()} REGIMES")
            print(
                df[
                    [
                        "period",
                        "net_pips",
                        "trend_efficiency",
                        "mean_atr_pips",
                        "return_autocorr_1",
                        "above_ema200_rate",
                        "mean_ema50_slope_pips",
                        "buy_hold_max_dd",
                    ]
                ].to_string(index=False)
            )
    print(f"\nMarket regime outputs written to: {out_dir}")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize raw EURUSD market regimes.")
    parser.add_argument("--pair", default="EURUSD")
    parser.add_argument("--years", default="2020,2021,2022,2023,2024")
    parser.add_argument("--timeframe", default="H1")
    parser.add_argument("--output-dir", default="outputs/market_regimes")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
