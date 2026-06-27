"""
Forex Triple Barrier Visualizer
================================
Interactive Plotly chart showing NORMALIZED price candles with the exact
TP/SL barrier lines overlaid, colored by triple barrier label outcome.

How to read this chart
----------------------
Candles show normalized (fracdiff / log-return) OHLC values — the same
price series the pipeline used to compute labels.

Dashed lines at each bar show the barrier levels set at that bar's close:
  GREEN dashed  upper = close[t] × (1 + vol_ewma[t] × k_up)   — take-profit
  RED   dashed  lower = close[t] × (1 - vol_ewma[t] × k_down) — stop-loss

Candle color = label outcome (forward-looking: describes what happens AFTER bar t):
  GREEN candle  label = +1  price crossed the upper barrier first (within horizon_bars)
  RED   candle  label = -1  price crossed the lower barrier first
  GRAY  candle  label =  0  neither barrier was reached — timeout

Usage (script):
    source ~/Trade_bot/.venv/bin/activate
    python visualize_targets.py

Usage (REPL / Jupyter):
    from visualize_targets import plot_triple_barrier
    results = pipeline.run(df_m1, "H1")   # must use target_type="triple_barrier"
    plot_triple_barrier(results, split="val", k_up=2.0, k_down=1.0).show()
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


_SPLIT_COLORS: dict[str, str] = {
    "train": "rgba(38, 166, 154, 0.07)",
    "val":   "rgba(41, 98, 255, 0.07)",
    "test":  "rgba(239, 83, 80, 0.07)",
}

# Each label class: legend name + candlestick colors (increasing / decreasing body)
_LABEL_STYLES: dict[int, dict] = {
     1: dict(
         name="TP  (label +1)",
         inc="#26a69a", dec="#1a7a70",
     ),
    -1: dict(
         name="SL  (label -1)",
         inc="#ef5350", dec="#b03b38",
     ),
     0: dict(
         name="Timeout  (label 0)",
         inc="#4a4e5c", dec="#3a3d4a",
     ),
}


def plot_triple_barrier(
    results: dict,
    *,
    split: str | None = None,
    start: str | None = None,
    end: str | None = None,
    k_up: float = 2.0,
    k_down: float = 1.0,
    height: int = 900,
) -> go.Figure:
    """
    Build an interactive chart: normalized OHLC candles + TP/SL barrier lines,
    candles colored by triple barrier label outcome.

    Parameters
    ----------
    results : dict returned by ForexPipeline.run() with target_type="triple_barrier"
    split   : "train" | "val" | "test" | None — restrict to that split's date range
    start   : ISO date string — further restrict window start
    end     : ISO date string — further restrict window end
    k_up    : upper barrier multiplier — must match what the pipeline used
    k_down  : lower barrier multiplier — must match what the pipeline used
    height  : figure height in pixels
    """
    # ── Validate ────────────────────────────────────────────────
    valid_splits = {"train", "val", "test", None}
    if split not in valid_splits:
        raise ValueError(
            f"split must be one of {sorted(s for s in valid_splits if s)!r} or None"
        )

    feat_df_full = pd.concat(
        [results["train_raw"], results["val_raw"], results["test_raw"]]
    ).sort_index()

    if "tb_label" not in feat_df_full.columns:
        raise ValueError(
            "tb_label not found — run the pipeline with target_type='triple_barrier'"
        )

    tf = results["timeframe"]

    # ── Split boundaries ─────────────────────────────────────────
    split_bounds = {
        "train": (results["train_raw"].index[0], results["train_raw"].index[-1]),
        "val":   (results["val_raw"].index[0],   results["val_raw"].index[-1]),
        "test":  (results["test_raw"].index[0],  results["test_raw"].index[-1]),
    }

    # ── Window bounds ────────────────────────────────────────────
    win_start = feat_df_full.index[0]
    win_end   = feat_df_full.index[-1]

    if split is not None:
        win_start, win_end = split_bounds[split]
    if start is not None:
        win_start = max(win_start, pd.Timestamp(start))
    if end is not None:
        win_end   = min(win_end,   pd.Timestamp(end))

    feat_window = feat_df_full.loc[win_start:win_end]

    if len(feat_window) == 0:
        raise ValueError(
            f"No data in range {win_start.date()} → {win_end.date()} "
            f"for {results['pair']} {tf}"
        )

    label_aligned = feat_window["tb_label"]

    # ── Barrier levels (exact values used during labeling) ───────
    upper_barrier = feat_window["close"] * (1 + feat_window["vol_ewma"] * k_up)
    lower_barrier = feat_window["close"] * (1 - feat_window["vol_ewma"] * k_down)

    n_tp  = (label_aligned == 1).sum()
    n_sl  = (label_aligned == -1).sum()
    n_to  = (label_aligned == 0).sum()
    total = len(label_aligned.dropna())
    print(
        f"[plot] {results['pair']} {tf}: {len(feat_window):,} bars  "
        f"{feat_window.index[0].date()} → {feat_window.index[-1].date()}\n"
        f"       TP={n_tp} ({100*n_tp/total:.1f}%)  "
        f"SL={n_sl} ({100*n_sl/total:.1f}%)  "
        f"Timeout={n_to} ({100*n_to/total:.1f}%)"
    )

    # ── Figure ───────────────────────────────────────────────────
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=[
            f"{results['pair']} — {tf}  |  "
            f"{win_start.date()} → {win_end.date()}  |  "
            f"TP {n_tp} · SL {n_sl} · Timeout {n_to}"
        ],
    )
    fig.layout.annotations[0].font.update(color="#d1d4dc", size=12)

    # ── Split shading ─────────────────────────────────────────────
    for split_name, (s_start, s_end) in split_bounds.items():
        shade_start = max(s_start, win_start)
        shade_end   = min(s_end,   win_end)
        if shade_start >= shade_end:
            continue
        fig.add_vrect(
            x0=shade_start, x1=shade_end,
            fillcolor=_SPLIT_COLORS[split_name],
            opacity=1, layer="below", line_width=0,
            annotation_text=split_name,
            annotation_position="top left",
            annotation=dict(font_size=10, font_color="#787b86"),
        )

    # ── One candlestick trace per label class (normalized OHLC) ──
    for label_val in [1, -1, 0]:
        style = _LABEL_STYLES[label_val]
        mask  = label_aligned == label_val
        if not mask.any():
            continue
        fw = feat_window[mask]
        fig.add_trace(go.Candlestick(
            x=fw.index,
            open=fw["open"],
            high=fw["high"],
            low=fw["low"],
            close=fw["close"],
            name=style["name"],
            increasing_line_color=style["inc"],
            decreasing_line_color=style["dec"],
            increasing_fillcolor=style["inc"],
            decreasing_fillcolor=style["dec"],
            hovertext=[f"label={label_val}"] * len(fw),
        ))

    # ── Barrier lines ─────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=upper_barrier.index,
        y=upper_barrier.values,
        name=f"Upper barrier (k={k_up})",
        mode="lines",
        line=dict(color="rgba(38,166,154,0.55)", width=1, dash="dot"),
        hovertemplate="upper=%{y:.6f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=lower_barrier.index,
        y=lower_barrier.values,
        name=f"Lower barrier (k={k_down})",
        mode="lines",
        line=dict(color="rgba(239,83,80,0.55)", width=1, dash="dot"),
        hovertemplate="lower=%{y:.6f}<extra></extra>",
    ))

    fig.update_xaxes(rangeslider_visible=False)

    # ── Styling ───────────────────────────────────────────────────
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#131722",
        plot_bgcolor="#131722",
        height=height,
        margin=dict(l=60, r=40, t=70, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left",   x=0,
            font=dict(color="#d1d4dc"),
        ),
        hovermode="x unified",
    )
    fig.update_yaxes(gridcolor="#1e222d", zeroline=False, tickfont=dict(color="#787b86"))
    fig.update_xaxes(
        gridcolor="#1e222d",
        tickfont=dict(color="#787b86"),
        showspikes=True, spikecolor="#787b86", spikethickness=1,
        rangebreaks=[dict(bounds=["sat", "mon"])] if results.get("weekends", "nogap") == "nogap" else [],
        rangeselector=dict(
            buttons=[
                dict(count=7,  label="1W", step="day",   stepmode="backward"),
                dict(count=1,  label="1M", step="month", stepmode="backward"),
                dict(count=3,  label="3M", step="month", stepmode="backward"),
                dict(count=6,  label="6M", step="month", stepmode="backward"),
                dict(step="all", label="All"),
            ],
            bgcolor="#1e222d", activecolor="#2962ff",
            font=dict(color="#d1d4dc"),
        ),
    )

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
        norm_method="fracdiff",
        target_type="triple_barrier", k_up=2.0, k_down=1.0, horizon_bars=10,barrier_price="hl"
    )
    df_m1   = loader.load_and_merge(HISTDATA_DIR, "EURUSD")
    results = pipeline.run(df_m1, timeframe="H1")

    plot_triple_barrier(results, split="val").write_html("chart_preds.html", auto_open=False)
    print("[Done] chart_preds.html")
