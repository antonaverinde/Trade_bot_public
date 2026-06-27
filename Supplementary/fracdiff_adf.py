"""
Optimal Fracdiff d — ADF Stationarity Sweep
============================================
Sweeps fractional differentiation order d over [0, 1] and tests stationarity
of the resulting close series with the Augmented Dickey-Fuller (ADF) test.

Goal: find the MINIMUM d that achieves stationarity (ADF p < sig_level),
      preserving as much price memory as possible.

How to read the chart
---------------------
Row 1 — ADF statistic vs d
    The series is stationary where the line dips BELOW a critical value line.
    More negative = stronger rejection of unit root = more stationary.

Row 2 — p-value vs d  (log scale)
    Drops below the significance threshold (dashed line) at the optimal d.

Row 3 — Correlation with raw close vs d
    Shows the memory-stationarity tradeoff: d=0 keeps all price memory (corr≈1),
    d=1 (full diff) loses most of it.  Pick the lowest d that still passes ADF.

Usage:
    source ~/Trade_bot/.venv/bin/activate
    python Supplementary/fracdiff_adf.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller

def _find_root() -> Path:
    """Find the Trade_bot project root by looking for histdata/ + Pipeline/."""
    candidates = []
    try:
        candidates.append(Path(__file__).resolve().parent.parent)
    except NameError:
        pass
    p = Path.cwd()
    for _ in range(6):
        candidates.append(p)
        p = p.parent
    for c in candidates:
        if (c / "histdata").is_dir() and (c / "Pipeline").is_dir():
            return c
    raise RuntimeError(
        "Cannot locate Trade_bot project root. "
        "Run from inside the Trade_bot directory or set _ROOT manually."
    )

_ROOT = _find_root()
sys.path.insert(0, str(_ROOT))
from Pipeline.pipeline import ForexDataLoader, normalize_prices, resample_ohlcv, TIMEFRAMES

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

PAIR      = "EURUSD"
TIMEFRAME = "H1"
YEARS     = None       # None = all available years
D_MIN     = 0.0
D_MAX     = 1.0
D_STEP    = 0.05
SIG_LEVEL = 0.1       # ADF significance threshold
#5% (α=0.05)	95% confidence that the series is stationary.	The industry standard. It provides a good balance between rejecting non-stationary data and retaining necessary information.
#10% (α=0.10)	90% confidence that the series is stationary.	Use this if your sample size is very small, or if you prefer a more lenient threshold to preserve as much memory as possible.

# ──────────────────────────────────────────────────────────────
# ADF sweep
# ──────────────────────────────────────────────────────────────

def sweep(df: pd.DataFrame, d_values: np.ndarray, sig_level: float) -> pd.DataFrame:
    """
    For each d, fracdiff the close series and run ADF.

    Returns a DataFrame with columns:
        d, adf_stat, p_val, crit_1, crit_5, crit_10, corr
    """
    raw_close = df["close"].copy()
    rows = []

    for d in d_values:
        d = round(float(d), 6)
        df_norm  = normalize_prices(df.copy(), method="fracdiff", d=d)
        series   = df_norm["close"].dropna()

        adf_stat, p_val, _, _, crit_vals, _ = adfuller(series, autolag="AIC", regression="c")
        corr = series.corr(raw_close.reindex(series.index))

        rows.append(dict(
            d       = d,
            adf_stat= adf_stat,
            p_val   = p_val,
            crit_1  = crit_vals["1%"],
            crit_5  = crit_vals["5%"],
            crit_10 = crit_vals["10%"],
            corr    = corr,
        ))
        status = "✓" if p_val < sig_level else "✗"
        print(f"  d={d:.2f}  ADF={adf_stat:8.3f}  p={p_val:.4f}  corr={corr:.4f}  {status}")

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────

def plot(res: pd.DataFrame, optimal_d: float | None,
         pair: str, timeframe: str, sig_level: float) -> go.Figure:

    title = (
        f"{pair} {timeframe} — "
        + (f"Optimal fracdiff d = {optimal_d:.2f}  (ADF p < {sig_level})"
           if optimal_d is not None
           else f"No d in range achieves ADF p < {sig_level}")
    )

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=["ADF Statistic", "p-value  (log scale)", "Correlation with raw close"],
    )

    for ann in fig.layout.annotations:
        ann.font.update(color="#d1d4dc", size=11)

    d = res["d"]

    # ── Row 1: ADF statistic ─────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=d, y=res["adf_stat"],
        name="ADF statistic",
        mode="lines+markers",
        marker=dict(size=5),
        line=dict(color="#2962ff", width=2),
    ), row=1, col=1)

    for label, col, color in [
        ("1%",  "crit_1",  "rgba(239,83,80,0.70)"),
        ("5%",  "crit_5",  "rgba(255,193,7,0.70)"),
        ("10%", "crit_10", "rgba(38,166,154,0.70)"),
    ]:
        # critical values are roughly constant — just use the mean
        cv = res[col].mean()
        fig.add_hline(
            y=cv, line_dash="dot", line_color=color,
            annotation_text=f"crit {label} ({cv:.2f})",
            annotation_position="right",
            annotation_font=dict(color=color, size=10),
            row=1, col=1,
        )

    # ── Row 2: p-value (log scale) ───────────────────────────────
    fig.add_trace(go.Scatter(
        x=d, y=res["p_val"],
        name="p-value",
        mode="lines+markers",
        marker=dict(size=5),
        line=dict(color="#ab47bc", width=2),
    ), row=2, col=1)

    fig.add_hline(
        y=sig_level, line_dash="dot",
        line_color="rgba(255,193,7,0.80)",
        annotation_text=f"p = {sig_level}",
        annotation_position="right",
        annotation_font=dict(color="rgba(255,193,7,0.90)", size=10),
        row=2, col=1,
    )
    fig.update_yaxes(type="log", row=2, col=1)

    # ── Row 3: correlation with raw close ────────────────────────
    fig.add_trace(go.Scatter(
        x=d, y=res["corr"],
        name="corr w/ raw close",
        mode="lines+markers",
        marker=dict(size=5),
        line=dict(color="#26a69a", width=2),
    ), row=3, col=1)

    # ── Optimal d vertical line on all rows ──────────────────────
    if optimal_d is not None:
        for row in range(1, 4):
            fig.add_vline(
                x=optimal_d,
                line_dash="dash",
                line_color="rgba(255,255,255,0.45)",
                annotation_text=f"d*={optimal_d:.2f}" if row == 1 else "",
                annotation_position="top right",
                annotation_font=dict(color="#d1d4dc", size=11),
                row=row, col=1,
            )

    # ── Styling ───────────────────────────────────────────────────
    fig.update_layout(
        title=dict(text=title, font=dict(color="#d1d4dc", size=13)),
        template="plotly_dark",
        paper_bgcolor="#131722",
        plot_bgcolor="#131722",
        height=800,
        margin=dict(l=60, r=120, t=80, b=50),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0,
            font=dict(color="#d1d4dc"),
        ),
        hovermode="x unified",
        showlegend=True,
    )
    for row in range(1, 4):
        fig.update_yaxes(gridcolor="#1e222d", zeroline=False,
                         tickfont=dict(color="#787b86"), row=row, col=1)
        fig.update_xaxes(gridcolor="#1e222d", tickfont=dict(color="#787b86"),
                         row=row, col=1)

    fig.update_xaxes(title_text="fracdiff d", title_font=dict(color="#787b86"),
                     row=3, col=1)

    return fig


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[ADF sweep] {PAIR} {TIMEFRAME}  d ∈ [{D_MIN}, {D_MAX}] step={D_STEP}")

    loader = ForexDataLoader()
    df_m1  = loader.load_and_merge(
        _ROOT / "histdata", PAIR, years=YEARS
    )
    df = resample_ohlcv(df_m1, TIMEFRAMES[TIMEFRAME]) if TIMEFRAME != "M1" else df_m1
    print(f"[Data] {len(df):,} {TIMEFRAME} bars\n")

    d_values = np.arange(D_MIN, D_MAX + D_STEP / 2, D_STEP)
    res      = sweep(df, d_values, SIG_LEVEL)

    passing  = res.loc[res["p_val"] < SIG_LEVEL, "d"]
    optimal_d = float(passing.min()) if not passing.empty else None

    print()
    if optimal_d is not None:
        print(f"[Result] Optimal d = {optimal_d:.2f}  "
              f"(minimum d where ADF p < {SIG_LEVEL})")
    else:
        print(f"[Result] No d in [{D_MIN}, {D_MAX}] achieves ADF p < {SIG_LEVEL}")

    print("\n── Full results ──────────────────────────────────")
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    out = _ROOT / "Supplementary" / "fracdiff_adf.html"
    plot(res, optimal_d, PAIR, TIMEFRAME, SIG_LEVEL).write_html(str(out), auto_open=False)
    print(f"\n[Done] {out}")
