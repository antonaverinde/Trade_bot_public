"""
TimesFM 2.5 forecast visualizer — matplotlib dark-theme PNG.

Produces a 2-row chart:
  Row 1 — last N candlesticks + 5 quantile lines (P10/P30/P50/P70/P90)
           + ground truth overlay (faded) + right-side value labels
  Row 2 — P(price > last close) at each future bar, green/red fill around 50%

Usage (script):
    python Timeseriesfm/timesfm_plots.py

Usage (import):
    from Timeseriesfm.timesfm_plots import plot_forecast, save_png
    fig = plot_forecast(result)
    save_png(fig, "timesfm_forecast.png")
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from Timeseriesfm.timesfm_inference import TimesFMForecaster

# ── Palette (dark TradingView theme) ─────────────────────────────────────────
_BG    = "#131722"
_PANEL = "#161b2a"
_GRID  = "#1e222d"
_TEXT  = "#d1d4dc"
_DIM   = "#787b86"
_BLUE  = "#2962ff"
_TEAL  = "#26a69a"
_RED   = "#ef5350"
_GOLD  = "#ffc107"

# 5 quantile lines selected from the 10 available levels
# (quantile key, color, linewidth, linestyle, label)
_PLINES = [
    (0.1, "#ef5350", 1.0, "--", "P10"),
    (0.3, "#ff8a65", 1.2, "--", "P30"),
    (0.5, "#ffc107", 2.2, "-",  "P50"),
    (0.7, "#80cbc4", 1.2, "--", "P70"),
    (0.9, "#26a69a", 1.0, "--", "P90"),
]


# ── Candlestick helper ────────────────────────────────────────────────────────

def _draw_candles(ax: plt.Axes, df: pd.DataFrame, alpha: float = 1.0) -> None:
    if df.empty:
        return

    xs    = mdates.date2num(df.index.to_pydatetime())
    bar_w = 0.55 * (xs[1] - xs[0]) if len(xs) >= 2 else 0.0004

    for xi, row in zip(xs, df.itertuples()):
        is_bull = row.close >= row.open
        color   = _TEAL if is_bull else _RED

        ax.plot([xi, xi], [row.low, row.high],
                color=color, linewidth=0.9, alpha=alpha, zorder=2,
                solid_capstyle="round")

        body_lo = min(row.open, row.close)
        body_hi = max(row.open, row.close)
        body_h  = max(body_hi - body_lo, abs(row.close) * 1e-5)
        rect = mpatches.Rectangle(
            (xi - bar_w / 2, body_lo), bar_w, body_h,
            facecolor=color, edgecolor=color, linewidth=0,
            alpha=alpha, zorder=3,
        )
        ax.add_patch(rect)


# ── Main chart ────────────────────────────────────────────────────────────────

def plot_forecast(
    result: dict,
    *,
    context_bars_shown: int = 50,
    figsize: tuple = (15, 9),
) -> plt.Figure:
    """
    Build a matplotlib forecast chart from a TimesFMForecaster result dict.

    Parameters
    ----------
    result             : dict from forecast_from_df() or forecast_from_pipeline()
    context_bars_shown : number of context candles to show before the forecast
    figsize            : (width, height) in inches
    """
    ctx_df   = result["context_df"]
    fc       = result["forecast"]
    gt_df    = result["ground_truth_df"]
    fc_ts    = result["forecast_timestamps"]
    pair     = result.get("pair", "")
    tf       = result.get("timeframe", "")
    ctx_end  = result["context_end"]

    qs         = fc["quantiles"]
    last_close = float(ctx_df["close"].iloc[-1])
    prob_up    = TimesFMForecaster.prob_above(fc, last_close)

    vis_ctx = ctx_df.iloc[-context_bars_shown:]
    fc_x    = mdates.date2num(fc_ts.to_pydatetime())
    ctx_x   = mdates.date2num(vis_ctx.index.to_pydatetime())
    now_x   = mdates.date2num(pd.Timestamp(ctx_end).to_pydatetime())

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.04},
    )
    fig.patch.set_facecolor(_BG)

    for ax in (ax1, ax2):
        ax.set_facecolor(_PANEL)
        ax.grid(True, color=_GRID, linewidth=0.5, linestyle="-", alpha=0.8)
        ax.tick_params(colors=_DIM, which="both", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(_GRID)

    # ── Context candlesticks ──────────────────────────────────────────────────
    _draw_candles(ax1, vis_ctx)

    if gt_df is not None and len(gt_df) > 0:
        _draw_candles(ax1, gt_df, alpha=0.38)

    # ── Percentile lines ─────────────────────────────────────────────────────
    fc_span = fc_x[-1] - now_x
    label_x = fc_x[-1] + fc_span * 0.01

    for q, color, lw, ls, name in _PLINES:
        ax1.plot(fc_x, qs[q], color=color, linewidth=lw, linestyle=ls,
                 label=name, zorder=5)
        ax1.annotate(
            f" {name} {qs[q][-1]:.5f}",
            xy=(fc_x[-1], qs[q][-1]),
            xytext=(label_x, qs[q][-1]),
            color=color, fontsize=7, va="center", ha="left",
            annotation_clip=False,
        )

    # ── "Now" vertical separator ──────────────────────────────────────────────
    for ax in (ax1, ax2):
        ax.axvline(now_x, color=_GOLD, linestyle="--", linewidth=1.3,
                   alpha=0.85, zorder=6)
    ax1.text(now_x, 0.97, " now",
             transform=ax1.get_xaxis_transform(),
             color=_GOLD, fontsize=8, va="top", ha="left")

    # ── P(up) subplot ─────────────────────────────────────────────────────────
    pu = np.asarray(prob_up)
    ax2.fill_between(fc_x, pu, 0.5, where=(pu >= 0.5), color=_TEAL, alpha=0.45, linewidth=0)
    ax2.fill_between(fc_x, pu, 0.5, where=(pu < 0.5),  color=_RED,  alpha=0.45, linewidth=0)
    ax2.plot(fc_x, pu, color=_BLUE, linewidth=1.5, zorder=5)
    ax2.axhline(0.5, color=_GOLD, linestyle=":", linewidth=1.0, alpha=0.55)
    ax2.set_ylim(0.0, 1.0)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax2.set_ylabel("P(up)", color=_DIM, fontsize=9)

    final_pu = float(pu[-1])
    pu_color = _TEAL if final_pu >= 0.5 else _RED
    ax2.annotate(
        f" {final_pu:.0%}",
        xy=(fc_x[-1], final_pu),
        xytext=(label_x, final_pu),
        color=pu_color, fontsize=8, va="center", ha="left",
        annotation_clip=False,
    )

    # ── X-axis ────────────────────────────────────────────────────────────────
    date_fmt = mdates.DateFormatter("%m-%d %H:%M")
    ax2.xaxis.set_major_formatter(date_fmt)
    ax1.xaxis.set_visible(False)
    ax2.tick_params(axis="x", labelsize=7, colors=_DIM)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=25, ha="right")

    margin_l = (fc_x[-1] - ctx_x[0]) * 0.01
    margin_r = fc_span * 0.18
    for ax in (ax1, ax2):
        ax.set_xlim(ctx_x[0] - margin_l, fc_x[-1] + margin_r)
        ax.xaxis_date()

    # ── Y formatting ──────────────────────────────────────────────────────────
    price_range = float(vis_ctx["high"].max() - vis_ctx["low"].min())
    decimals    = max(1, min(6, int(-np.log10(price_range)) + 3)) if price_range > 0 else 5
    fmt_str     = f"{{:.{decimals}f}}"
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: fmt_str.format(x)))
    ax1.tick_params(axis="y", colors=_DIM, labelsize=7)

    # ── Title ─────────────────────────────────────────────────────────────────
    parts = [p for p in [pair, tf] if p]
    parts.append("TimesFM 2.5 Zero-Shot")
    parts.append(f"context → {pd.Timestamp(ctx_end).strftime('%Y-%m-%d %H:%M')}")
    parts.append(f"{len(fc_ts)}-bar horizon")
    ax1.set_title("  ·  ".join(parts), color=_TEXT, fontsize=10, pad=8, loc="left")

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [
        mlines.Line2D([], [], color=c, linewidth=lw, linestyle=ls, label=name)
        for _, c, lw, ls, name in _PLINES
    ]
    if gt_df is not None and len(gt_df) > 0:
        handles.append(
            mpatches.Patch(facecolor=_TEAL, alpha=0.38, edgecolor="none",
                           label="Ground truth")
        )

    ax1.legend(
        handles=handles,
        loc="upper left", fontsize=7.5,
        facecolor=_BG, edgecolor=_GRID,
        labelcolor=_TEXT, framealpha=0.9,
    )

    fig.tight_layout(pad=1.0)
    return fig


# ── Convenience ──────────────────────────────────────────────────────────────

def save_png(fig: plt.Figure, path: str = "timesfm_forecast.png", dpi: int = 150) -> str:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[TimesFM] Saved → {path}")
    return path


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).parent.parent))
    from Pipeline.pipeline import ForexDataLoader

    HISTDATA_DIR = _Path(__file__).parent.parent / "histdata"
    OUT_PNG      = _Path(__file__).parent.parent / "timesfm_forecast.png"

    loader   = ForexDataLoader()
    df_m1    = loader.load_and_merge(HISTDATA_DIR, "EURUSD", years=[2022, 2023])

    price_df = df_m1.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    price_df.attrs["pair"] = "EURUSD"

    forecaster = TimesFMForecaster(context_length=512)
    result     = forecaster.forecast_from_df(
        price_df,
        context_end="2023-09-01 00:00",
        prediction_length=48,
    )
    result["timeframe"] = "H1"

    fig = plot_forecast(result, context_bars_shown=50)
    save_png(fig, str(OUT_PNG))
    print("[Done]", OUT_PNG)
