"""
Kronos forecast visualizer — matplotlib dark-theme PNG.

Produces a 2-row chart:
  Row 1 — last N context candlesticks + predicted OHLCV candlesticks
           + faded ground-truth overlay (if available)
  Row 2 — predicted close line; if n_samples > 1: per-sample lines
           with min/max band shaded

Usage (script):
    python Kronos/kronos_plots.py

Usage (import):
    from Kronos.kronos_plots import plot_forecast, save_png
    fig = plot_forecast(result)
    save_png(fig, "kronos_forecast.png")
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

# ── Palette (dark TradingView theme) ─────────────────────────────────────────
_BG       = "#131722"
_PANEL    = "#161b2a"
_GRID     = "#1e222d"
_TEXT     = "#d1d4dc"
_DIM      = "#787b86"
_TEAL     = "#26a69a"
_RED      = "#ef5350"
_GOLD     = "#ffc107"
_BLUE     = "#2962ff"
_ORANGE   = "#ff8a65"

# forecast candles drawn in distinct colours to separate from context
_FC_BULL = "#00897b"   # darker teal for bullish forecast bars
_FC_BEAR = "#c62828"   # darker red for bearish forecast bars


# ── Candlestick helper ────────────────────────────────────────────────────────

def _draw_candles(
    ax: plt.Axes,
    df: pd.DataFrame,
    alpha: float = 1.0,
    bull_color: str = _TEAL,
    bear_color: str = _RED,
) -> None:
    """Draw OHLCV candlesticks on `ax`. df must have DatetimeIndex + open/high/low/close."""
    if df.empty:
        return
    xs = mdates.date2num(df.index.to_pydatetime())
    bar_w = 0.55 * (xs[1] - xs[0]) if len(xs) >= 2 else 0.0004

    for xi, row in zip(xs, df.itertuples()):
        is_bull = row.close >= row.open
        color   = bull_color if is_bull else bear_color
        ax.plot([xi, xi], [row.low, row.high],
                color=color, linewidth=0.9, alpha=alpha, zorder=2, solid_capstyle="round")
        body_lo = min(row.open, row.close)
        body_hi = max(row.open, row.close)
        body_h  = max(body_hi - body_lo, abs(row.close) * 1e-5)
        rect = mpatches.Rectangle(
            (xi - bar_w / 2, body_lo), bar_w, body_h,
            facecolor=color, edgecolor=color, linewidth=0, alpha=alpha, zorder=3,
        )
        ax.add_patch(rect)


# ── Main chart ────────────────────────────────────────────────────────────────

def plot_forecast(
    result: dict,
    *,
    context_bars_shown: int = 50,
    figsize: tuple = (15, 9),
) -> plt.Figure:
    """Build a matplotlib forecast chart from a KronosForecaster result dict.

    Parameters
    ----------
    result             : dict from forecast_from_df() or forecast_from_pipeline()
    context_bars_shown : number of context candles to show before the forecast
    figsize            : (width, height) in inches

    Returns
    -------
    matplotlib.figure.Figure
    """
    ctx_df    = result["context_df"]
    mean_fc   = result["mean_forecast"]       # pd.DataFrame (pred_len, 6)
    gt_df     = result.get("ground_truth_df")
    fc_ts     = result["forecast_timestamps"]
    pair      = result.get("pair", "")
    tf        = result.get("timeframe", "")
    ctx_end   = result["context_end"]
    n_samples = result.get("n_samples", 1)
    samples   = result.get("samples", [])     # list[DataFrame]

    last_close = float(ctx_df["close"].iloc[-1])
    vis_ctx    = ctx_df.iloc[-context_bars_shown:]

    fc_x   = mdates.date2num(pd.DatetimeIndex(fc_ts).to_pydatetime())
    ctx_x  = mdates.date2num(vis_ctx.index.to_pydatetime())
    now_x  = mdates.date2num(pd.Timestamp(ctx_end).to_pydatetime())

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize,
        gridspec_kw={"height_ratios": [7, 3], "hspace": 0.04},
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

    # ── Forecast OHLCV candlesticks ───────────────────────────────────────────
    mean_fc_plot = mean_fc[["open", "high", "low", "close"]].copy()
    mean_fc_plot.index = pd.DatetimeIndex(fc_ts)
    _draw_candles(ax1, mean_fc_plot, alpha=1.0, bull_color=_FC_BULL, bear_color=_FC_BEAR)

    # ── Ground truth candles (faded overlay) ─────────────────────────────────
    if gt_df is not None and len(gt_df) > 0:
        gt_plot = gt_df[["open", "high", "low", "close"]].copy()
        _draw_candles(ax1, gt_plot, alpha=0.35)

    # ── "Now" vertical separator ──────────────────────────────────────────────
    ax1.axvline(now_x, color=_GOLD, linestyle="--", linewidth=1.3, alpha=0.85, zorder=6)
    ax2.axvline(now_x, color=_GOLD, linestyle="--", linewidth=1.3, alpha=0.85, zorder=6)
    ax1.text(now_x, 0.97, " now",
             transform=ax1.get_xaxis_transform(),
             color=_GOLD, fontsize=8, va="top", ha="left")

    # ── Close-price subplot ───────────────────────────────────────────────────
    close_vals = mean_fc["close"].values
    end_color  = _TEAL if close_vals[-1] >= last_close else _RED

    if n_samples > 1 and len(samples) > 1:
        # thin sample lines
        for s in samples:
            s_close = s["close"].values
            ax2.plot(fc_x, s_close, color=_BLUE, linewidth=0.5, alpha=0.25, zorder=3)
        # min/max band
        all_close = np.stack([s["close"].values for s in samples], axis=0)
        ax2.fill_between(fc_x, all_close.min(0), all_close.max(0),
                         color=end_color, alpha=0.18, linewidth=0, zorder=2)

    ax2.plot(fc_x, close_vals, color=end_color, linewidth=2.0, zorder=5, label="Forecast close")
    ax2.axhline(last_close, color=_GOLD, linestyle=":", linewidth=1.0, alpha=0.6)

    # mark final value
    fc_span = fc_x[-1] - now_x
    label_x = fc_x[-1] + fc_span * 0.01
    price_decimals = max(1, min(6, int(-np.log10(max(abs(last_close), 1e-8))) + 4))
    fmt = f"{{:.{price_decimals}f}}"
    ax2.annotate(
        f" {fmt.format(close_vals[-1])}",
        xy=(fc_x[-1], close_vals[-1]), xytext=(label_x, close_vals[-1]),
        color=end_color, fontsize=8, va="center", ha="left", annotation_clip=False,
    )
    ax2.set_ylabel("Pred close", color=_DIM, fontsize=9)

    # ── X-axis formatting ─────────────────────────────────────────────────────
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

    # ── Y tick formatting ─────────────────────────────────────────────────────
    price_range = float(vis_ctx["high"].max() - vis_ctx["low"].min())
    decimals    = max(1, min(6, int(-np.log10(price_range)) + 3)) if price_range > 0 else 5
    fmt_str     = f"{{:.{decimals}f}}"
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: fmt_str.format(x)))
    ax1.tick_params(axis="y", colors=_DIM, labelsize=7)

    # ── Title ─────────────────────────────────────────────────────────────────
    parts = [p for p in [pair, tf] if p]
    parts.append("Kronos OHLCV Forecast")
    parts.append(f"context → {pd.Timestamp(ctx_end).strftime('%Y-%m-%d %H:%M')}")
    parts.append(f"{len(fc_ts)}-bar horizon")
    if n_samples > 1:
        parts.append(f"n_samples={n_samples}")
    ax1.set_title("  ·  ".join(parts), color=_TEXT, fontsize=10, pad=8, loc="left")

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(facecolor=_TEAL, edgecolor="none", label="Context (bull)"),
        mpatches.Patch(facecolor=_RED,  edgecolor="none", label="Context (bear)"),
        mpatches.Patch(facecolor=_FC_BULL, edgecolor="none", label="Forecast (bull)"),
        mpatches.Patch(facecolor=_FC_BEAR, edgecolor="none", label="Forecast (bear)"),
    ]
    if gt_df is not None and len(gt_df) > 0:
        handles.append(mpatches.Patch(facecolor=_TEAL, alpha=0.35, edgecolor="none", label="Ground truth"))
    ax1.legend(
        handles=handles, loc="upper left", fontsize=7.5,
        facecolor=_BG, edgecolor=_GRID, labelcolor=_TEXT, framealpha=0.9,
    )

    fig.tight_layout(pad=1.0)
    return fig


# ── Convenience ──────────────────────────────────────────────────────────────

def save_png(fig: plt.Figure, path: str = "kronos_forecast.png", dpi: int = 150) -> str:
    """Save figure to PNG and return the path."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# ── Standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    _PROJ = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_PROJ))

    from Pipeline.pipeline import ForexDataLoader
    from Kronos.kronos_inference import KronosForecaster

    loader = ForexDataLoader()
    df_m1  = loader.load_and_merge(_PROJ / "histdata", pair="EURUSD", years=[2020])
    df_h1  = df_m1.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()

    fc = KronosForecaster(max_context=512)
    fc.load()
    result = fc.forecast_from_df(df_h1, context_end="2020-06-01 16:00", pred_len=20)

    fig  = plot_forecast(result, context_bars_shown=50)
    path = save_png(fig, str(_PROJ / "kronos_forecast.png"))
    print(f"Saved → {path}")
