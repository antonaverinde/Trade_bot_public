"""
Forex Chart Visualizer
======================
Interactive Plotly chart built from a ForexPipeline results dict.

Usage (script):
    uv run python visualize.py

Usage (Jupyter / REPL):
    from visualize import plot

    results = pipeline.run(df_m1, "H1")

    plot(results)                                        # price only, all data
    plot(results, split="val")                           # val period only
    plot(results, split="val",
         features=["rsi_14", "adx_14"])                 # with feature subplots
    plot(results, start="2023-01-01", end="2023-06-01",
         features=["rsi_14", "adx_14", "bb_pct_b"])     # explicit date range
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from Pipeline.pipeline import resample_ohlcv, TIMEFRAMES


# ──────────────────────────────────────────────────────────────
# Feature reference lines: shown as dotted horizontal guides
# ──────────────────────────────────────────────────────────────

_REFERENCE_LINES: dict[str, list[tuple[float, str]]] = {
    "rsi_14":   [(70, "rgba(239,83,80,0.45)"),   (30, "rgba(38,166,154,0.45)")],
    "rsi_21":   [(70, "rgba(239,83,80,0.45)"),   (30, "rgba(38,166,154,0.45)")],
    "adx_14":   [(25, "rgba(255,193,7,0.45)")],
    "bb_pct_b": [(1.0, "rgba(239,83,80,0.35)"),  (0.5, "rgba(150,150,150,0.35)"),
                 (0.0, "rgba(38,166,154,0.35)")],
}

_SPLIT_COLORS: dict[str, str] = {
    "train": "rgba(38, 166, 154, 0.07)",
    "val":   "rgba(41, 98, 255, 0.07)",
    "test":  "rgba(239, 83, 80, 0.07)",
}


# ──────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────

def plot(
    results: dict,
    *,
    split: str | None = None,
    start: str | None = None,
    end: str | None = None,
    features: list[str] | None = None,
    normalized: bool = False,
    height: int = 900,
) -> go.Figure:
    """
    Build an interactive multi-panel Plotly chart from pipeline results.

    Parameters
    ----------
    results    : dict returned by ForexPipeline.run()
    split      : "train" | "val" | "test" | None — restrict to that split's date range
    start      : ISO date string (e.g. "2023-06-01") — further restrict window start
    end        : ISO date string — further restrict window end
    features   : list of feature names from results["feature_cols"];
                 each gets its own subplot row below the price panel.
                 Pass None or [] to show price only.
    normalized : if True, plot the pipeline-normalized close (fracdiff / log_returns / raw)
                 as a line instead of the raw OHLCV candlestick.
    height     : figure height in pixels

    Returns
    -------
    go.Figure — call .show() or .write_html("out.html")

    Available features
    ------------------
    results["feature_cols"]  — full list
    """
    # ── Validate ────────────────────────────────────────────────
    valid_splits = {"train", "val", "test", None}
    if split not in valid_splits:
        raise ValueError(f"split must be one of {sorted(s for s in valid_splits if s)!r} or None, got {split!r}")

    if features:
        unknown = set(features) - set(results["feature_cols"])
        if unknown:
            raise ValueError(
                f"Unknown feature(s): {sorted(unknown)!r}\n"
                f"Available: {results['feature_cols']}"
            )

    # ── Build price DataFrame (real OHLCV price levels) ─────────
    tf = results["timeframe"]
    if tf == "M1":
        price_df = results["raw_m1"].copy()
    else:
        price_df = resample_ohlcv(results["raw_m1"], TIMEFRAMES[tf])

    # ── Combined features (unscaled indicators) ─────────────────
    feat_df = pd.concat(
        [results["train_raw"], results["val_raw"], results["test_raw"]]
    ).sort_index()

    # ── Split boundaries ─────────────────────────────────────────
    split_bounds = {
        "train": (results["train_raw"].index[0], results["train_raw"].index[-1]),
        "val":   (results["val_raw"].index[0],   results["val_raw"].index[-1]),
        "test":  (results["test_raw"].index[0],  results["test_raw"].index[-1]),
    }

    # ── Window bounds ────────────────────────────────────────────
    win_start = feat_df.index[0]
    win_end   = feat_df.index[-1]

    if split is not None:
        win_start, win_end = split_bounds[split]

    if start is not None:
        win_start = max(win_start, pd.Timestamp(start))
    if end is not None:
        win_end   = min(win_end,   pd.Timestamp(end))

    price_window = price_df.loc[win_start:win_end]
    feat_window  = feat_df.loc[win_start:win_end]

    if len(price_window) == 0:
        raise ValueError(
            f"No price data in range {win_start.date()} → {win_end.date()} "
            f"for {results['pair']} {tf}"
        )

    print(
        f"[plot] {results['pair']} {tf}: {len(price_window):,} bars  "
        f"{price_window.index[0].date()} → {price_window.index[-1].date()}"
    )

    # ── Row layout ───────────────────────────────────────────────
    feat_list    = features or []
    n_feat_rows  = len(feat_list)
    n_rows       = 1 + n_feat_rows

    if n_feat_rows == 0:
        row_heights = [1.0]
    else:
        feat_h      = 0.5 / n_feat_rows
        row_heights = [0.5] + [feat_h] * n_feat_rows

    subplot_titles = [
        f"{results['pair']} — {tf}  ({win_start.date()} → {win_end.date()})"
    ] + feat_list

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    for ann in fig.layout.annotations:
        ann.font.color = "#d1d4dc"
        ann.font.size  = 11

    # ── Split-region shading ─────────────────────────────────────
    for split_name, (s_start, s_end) in split_bounds.items():
        shade_start = max(s_start, win_start)
        shade_end   = min(s_end,   win_end)
        if shade_start >= shade_end:
            continue
        for row in range(1, n_rows + 1):
            fig.add_vrect(
                x0=shade_start,
                x1=shade_end,
                fillcolor=_SPLIT_COLORS[split_name],
                opacity=1,
                layer="below",
                line_width=0,
                annotation_text=split_name if row == 1 else "",
                annotation_position="top left",
                annotation=dict(font_size=10, font_color="#787b86"),
                row=row,
                col=1,
            )

    # ── Row 1: price panel ────────────────────────────────────────
    if normalized:
        norm_series = feat_window["close"].dropna()
        fig.add_trace(go.Scatter(
            x=norm_series.index,
            y=norm_series.values,
            name=f"Close ({results.get('norm_method', 'normalized')})",
            mode="lines",
            line=dict(color="#d1d4dc", width=1.5),
        ), row=1, col=1)
    else:
        fig.add_trace(go.Candlestick(
            x=price_window.index,
            open=price_window["open"],
            high=price_window["high"],
            low=price_window["low"],
            close=price_window["close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a",
            decreasing_fillcolor="#ef5350",
        ), row=1, col=1)
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)

    # ── Feature rows ─────────────────────────────────────────────
    for row_idx, feat_name in enumerate(feat_list):
        current_row = row_idx + 2

        if feat_name not in feat_window.columns:
            print(f"[plot] Warning: '{feat_name}' not found in window — skipping")
            continue

        series = feat_window[feat_name].dropna()

        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            name=feat_name,
            mode="lines",
            line=dict(width=1.2),
        ), row=current_row, col=1)

        for y_val, color in _REFERENCE_LINES.get(feat_name, []):
            fig.add_hline(
                y=y_val,
                line_dash="dot",
                line_color=color,
                row=current_row,
                col=1,
            )

    # ── Global styling ───────────────────────────────────────────
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#131722",
        plot_bgcolor="#131722",
        height=height,
        margin=dict(l=60, r=40, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            font=dict(color="#d1d4dc"),
        ),
        hovermode="x unified",
    )

    for row in range(1, n_rows + 1):
        fig.update_yaxes(
            gridcolor="#1e222d",
            zeroline=False,
            tickfont=dict(color="#787b86"),
            row=row,
            col=1,
        )

    fig.update_xaxes(
        gridcolor="#1e222d",
        tickfont=dict(color="#787b86"),
        showspikes=True,
        spikecolor="#787b86",
        spikethickness=1,
        #rangebreaks=[dict(bounds=["sat", "mon"])] if results.get("weekends", "nogap") == "nogap" else [],
    )

    # Range selector on the bottom-most x-axis
    bottom_xaxis = "xaxis" if n_rows == 1 else f"xaxis{n_rows}"
    fig.update_layout(**{
        bottom_xaxis: dict(
            rangeselector=dict(
                buttons=[
                    dict(count=7,  label="1W",  step="day",   stepmode="backward"),
                    dict(count=1,  label="1M",  step="month", stepmode="backward"),
                    dict(count=3,  label="3M",  step="month", stepmode="backward"),
                    dict(count=6,  label="6M",  step="month", stepmode="backward"),
                    dict(step="all", label="All"),
                ],
                bgcolor="#1e222d",
                activecolor="#2962ff",
                font=dict(color="#d1d4dc"),
            ),
        )
    })

    return fig


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from Pipeline.pipeline import ForexDataLoader, ForexPipeline

    HISTDATA_DIR = Path(__file__).parent / "histdata"

    loader   = ForexDataLoader()
    pipeline = ForexPipeline(
        lags=[1, 2, 5, 10],
        target_horizons=[1, 5, 15],
        norm_method="fracdiff",fracdiff_d=0.3,
        target_type="triple_barrier", k_up=2.0, k_down=1.0, horizon_bars=10,threshold=6*1e-4,scaling="none",window_size=500
    )
    df_m1   = loader.load_and_merge(HISTDATA_DIR, "EURUSD", weekends="filled")
    results = pipeline.run(df_m1, timeframe="H1")

    NORMALIZED = True   # False → raw OHLCV candlestick, True → fracdiff/log_returns line

    plot(
        results,
        split="val",
        features=["rsi_14", "adx_14", "bb_pct_b"],
        normalized=NORMALIZED,
    ).write_html("chart.html", auto_open=False)
    print("[Done] chart.html")
