"""
kronos_feature_plot.py
======================
Compare Kronos feature predictions from a featdata parquet against the
pipeline's normalized close price for the same period.

All parameters are parsed from the parquet filename — no manual re-entry.
Feature columns are discovered dynamically from the parquet schema.

Two modes are auto-detected from the parquet columns:

  OHLC mode  (n_samples=1 during generation)
  ───────────────────────────────────────────
  Columns: close_h{H}, high_h{H}, low_h{H}, spread_h{H}
  Row 1 (55%) — actual (grey) + pred high (teal dashed) + pred low (red dashed)
                + pred close (gold)
  Row 2 (30%) — spread_h{horizon}: log(pred_high/pred_low) — volatility proxy
  Row 3 (15%) — staleness

  Probabilistic mode  (n_samples>1 during generation)
  ─────────────────────────────────────────────────────
  Columns: q{pp}_h{H}  (no OHLC columns)
  Row 1 (55%) — actual (grey) + quantile fan (outer band fill + q50 gold line)
  Row 2 (30%) — q50_h{horizon} − actual error fill
  Row 3 (15%) — staleness

Usage (CLI):
    python Kronos/kronos_feature_plot.py featdata/EURUSD_H1_kron_ctx512_int10_h5-10-20_logret_wfilled_snone_2023.parquet

Usage (notebook / REPL):
    from Kronos.kronos_feature_plot import plot_file
    fig = plot_file("featdata/....parquet", plot_horizon=10)
    fig.show()

Output: HTML file written next to the input parquet (_plot suffix).
"""

import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Path setup ────────────────────────────────────────────────────────────────

def _find_proj() -> Path:
    try:
        start: Path = Path(__file__).resolve()
    except NameError:
        start = Path.cwd() / "_notebook_"
    p = start if start.is_dir() else start.parent
    while p != p.parent:
        if (p / "Pipeline").is_dir() and (p / "Kronos").is_dir():
            return p
        p = p.parent
    return start.parent.parent


_PROJ = _find_proj()
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from Pipeline.pipeline import ForexDataLoader, ForexPipeline  # noqa: E402

# ── Colour palette ────────────────────────────────────────────────────────────

_BG       = "#131722"
_GRID     = "#1e222d"
_TICK_CLR = "#787b86"
_ACTUAL   = "#d1d4dc"
_CLOSE    = "#ffc107"   # gold — predicted close (median)
_HIGH_CLR = "#26a69a"   # teal — predicted high
_LOW_CLR  = "#ef5350"   # red  — predicted low
_SPREAD   = "#2962ff"   # blue — spread area fill
_RERUN    = "rgba(120,123,134,0.20)"

# ── Column schema discovery ───────────────────────────────────────────────────

_CLOSE_RE  = re.compile(r"^close_h(\d+)$")
_HIGH_RE   = re.compile(r"^high_h(\d+)$")
_LOW_RE    = re.compile(r"^low_h(\d+)$")
_SPREAD_RE = re.compile(r"^spread_h(\d+)$")
_Q_RE      = re.compile(r"^q(\d+)_h(\d+)$")


def _discover_schema(df: pd.DataFrame) -> tuple[dict[int, dict], bool]:
    """Scan df.columns for Kronos feature columns.

    Returns (schema, prob_mode) where:
      schema    — {horizon: {"close": col|None, "high": col|None, "low": col|None,
                              "spread": col|None, "quantiles": [sorted q cols]}}
      prob_mode — True when only quantile columns are present (n_samples > 1 generation)
    """
    horizons: dict[int, dict] = {}

    # OHLC columns (point-forecast mode)
    for col in df.columns:
        for regex, key in [(_CLOSE_RE, "close"), (_HIGH_RE, "high"),
                           (_LOW_RE, "low"), (_SPREAD_RE, "spread")]:
            m = regex.match(col)
            if m:
                h = int(m.group(1))
                horizons.setdefault(h, {"close": None, "high": None, "low": None,
                                         "spread": None, "quantiles": []})
                horizons[h][key] = col
                break

    # Quantile columns (probabilistic mode, or legacy mixed mode)
    for col in df.columns:
        m = _Q_RE.match(col)
        if m:
            h = int(m.group(2))
            horizons.setdefault(h, {"close": None, "high": None, "low": None,
                                     "spread": None, "quantiles": []})
            horizons[h]["quantiles"].append(col)

    for h in horizons:
        horizons[h]["quantiles"] = sorted(horizons[h]["quantiles"])

    # Detect mode: probabilistic if no horizon has any OHLC column
    prob_mode = all(
        info["close"] is None and info["high"] is None and info["low"] is None
        for info in horizons.values()
    )

    return horizons, prob_mode


# ── Filename parser ───────────────────────────────────────────────────────────

_FNAME_RE = re.compile(
    r"^(?P<pair>[A-Z]+)_(?P<tf>[A-Z0-9]+)"
    r"_kron"
    r"_ctx(?P<ctx>\d+)"
    r"_int(?P<intv>\d+)"
    r"(?:_h(?P<horizons>[\d]+(?:-[\d]+)*))?"
    r"_(?P<norm>logret|fdiff[\d.]+|raw)"
    r"(?:_(?P<wknd>wfilled|wnogap|wgaps))?"
    r"(?:_(?P<scale>snone|sroll\d+|sglob))?"
    r"(?:_s(?P<nsamples>\d+))?"
    r"_(?P<year_tag>[\d_]+(?:-[\d_]+)?)\.parquet$"
)


def _parse_fname(fname: str) -> dict:
    m = _FNAME_RE.match(fname)
    if not m:
        raise ValueError(
            f"Cannot parse parameters from filename {fname!r}.\n"
            f"Expected format: PAIR_TF_kron_ctx<N>_int<I>_h<H1>-<H2>-..._<norm>_<wknd>_<scale>_YEAR.parquet"
        )
    g = m.groupdict()

    norm_raw = g["norm"]
    if norm_raw == "logret":
        norm_method, fracdiff_d = "log_returns", 0.4
    elif norm_raw.startswith("fdiff"):
        norm_method, fracdiff_d = "fracdiff", float(norm_raw[5:])
    else:
        norm_method, fracdiff_d = "raw", 0.4

    year_tag = g["year_tag"]
    parts    = year_tag.split("_")
    if all(len(p) == 4 and p.isdigit() for p in parts):
        years = [int(p) for p in parts]
        start = end = None
    else:
        years = None
        start = f"{parts[0][:4]}-{parts[0][4:6]}-{parts[0][6:8]}" if len(parts[0]) == 8 else None
        end   = f"{parts[1][:4]}-{parts[1][4:6]}-{parts[1][6:8]}" if len(parts) > 1 and len(parts[1]) == 8 else None

    wknd_map = {"wfilled": "filled", "wnogap": "nogap", "wgaps": "gaps"}
    weekends = wknd_map.get(g["wknd"] or "wfilled", "filled")

    scale_raw = g["scale"] or "snone"
    if scale_raw == "snone":
        scaling, scaling_window = "none", 200
    elif scale_raw == "sglob":
        scaling, scaling_window = "global", 200
    else:
        scaling, scaling_window = "rolling", int(scale_raw[5:])

    horizons_raw = g.get("horizons")
    horizons = [int(x) for x in horizons_raw.split("-")] if horizons_raw else None

    return dict(
        pair           = g["pair"],
        timeframe      = g["tf"],
        context_length = int(g["ctx"]),
        calc_interval  = int(g["intv"]),
        horizons       = horizons,
        n_samples      = int(g["nsamples"]) if g.get("nsamples") else 1,
        norm_method    = norm_method,
        fracdiff_d     = fracdiff_d,
        weekends       = weekends,
        scaling        = scaling,
        scaling_window = scaling_window,
        years          = years,
        start          = start,
        end            = end,
    )


# ── Pipeline runner ───────────────────────────────────────────────────────────

def _run_pipeline(params: dict, histdata_dir: Path, threshold: float = 6e-4) -> pd.Series:
    """Return normalized close series for the full date range (fracdiff warm-up intact)."""
    loader = ForexDataLoader()
    df_m1  = loader.load_and_merge(
        str(histdata_dir), pair=params["pair"],
        years=params["years"], weekends=params["weekends"],
    )
    pipeline = ForexPipeline(
        norm_method = params["norm_method"],
        fracdiff_d  = params["fracdiff_d"],
        threshold   = threshold,
        target_type = "lag",
        weekends    = params["weekends"],
    )
    results = pipeline.run(df_m1, timeframe=params["timeframe"])
    full_norm = pd.concat(
        [results["train_raw"], results["val_raw"], results["test_raw"]]
    ).sort_index()["close"]

    scaling = params.get("scaling", "none")
    if scaling == "rolling":
        w   = params["scaling_window"]
        mu  = full_norm.rolling(w, min_periods=1).mean()
        std = full_norm.rolling(w, min_periods=1).std().fillna(1.0).replace(0.0, 1.0)
        full_norm = (full_norm - mu) / std
    elif scaling == "global":
        mu  = float(full_norm.mean())
        std = float(full_norm.std()) or 1.0
        full_norm = (full_norm - mu) / std

    return full_norm


# ── Figure builder ────────────────────────────────────────────────────────────

def _build_figure(
    df: pd.DataFrame,
    params: dict,
    schema: dict[int, dict],
    prob_mode: bool,
    plot_horizon: int,
) -> go.Figure:
    """Build 3-panel Plotly figure for Kronos features vs actual.

    Dispatches to OHLC layout or probabilistic (quantile fan) layout based on prob_mode.
    """
    info       = schema[plot_horizon]
    close_col  = info["close"]
    high_col   = info["high"]
    low_col    = info["low"]
    spread_col = info["spread"]
    q_cols     = info["quantiles"]    # sorted list of q{pp}_h{H} column names

    n_samp = params.get("n_samples", 1)
    horizons_str = ", ".join(str(h) for h in sorted(schema.keys()))
    title = (
        f"{params['pair']} {params['timeframe']} — "
        f"Kronos ctx={params['context_length']} "
        f"int={params['calc_interval']} "
        f"horizons=[{horizons_str}]  "
        f"plotting h={plot_horizon}  "
        f"[{params['norm_method']}]"
        + (f"  n_samples={n_samp}" if n_samp > 1 else "")
    )

    row2_label = (
        f"q50_h{plot_horizon} − actual  (prediction error)"
        if prob_mode else
        f"Spread  (log high/low at h={plot_horizon})  — predicted volatility"
    )

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.55, 0.30, 0.15],
        subplot_titles=[title, row2_label, "Staleness"],
    )

    for ann in fig.layout.annotations:
        ann.font.color = "#d1d4dc"
        ann.font.size  = 11

    x = df.index

    # ── Row 1 ────────────────────────────────────────────────────────────────
    if "actual" in df.columns:
        fig.add_trace(go.Scatter(
            x=x, y=df["actual"], name="Actual",
            mode="lines", line=dict(color=_ACTUAL, width=1.5),
        ), row=1, col=1)

    if prob_mode:
        # Quantile fan: outer band fill + median gold line
        if len(q_cols) >= 2:
            fig.add_trace(go.Scatter(
                x=x, y=df[q_cols[-1]], name=f"{q_cols[-1]} (h{plot_horizon})",
                mode="lines", line=dict(width=0), showlegend=False,
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=x, y=df[q_cols[0]], name=f"{q_cols[0]}–{q_cols[-1]} band",
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(255,193,7,0.10)",
            ), row=1, col=1)
            # inner band (25th–75th) if available
            if len(q_cols) >= 4:
                inner_hi = q_cols[len(q_cols) - 2]
                inner_lo = q_cols[1]
                fig.add_trace(go.Scatter(
                    x=x, y=df[inner_hi], name="inner hi",
                    mode="lines", line=dict(width=0), showlegend=False,
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=x, y=df[inner_lo], name="inner band",
                    mode="lines", line=dict(width=0),
                    fill="tonexty", fillcolor="rgba(255,193,7,0.18)",
                    showlegend=False,
                ), row=1, col=1)
        # median line
        q50_candidates = [c for c in q_cols if c.startswith("q50_")]
        q_mid = q50_candidates[0] if q50_candidates else (q_cols[len(q_cols) // 2] if q_cols else None)
        if q_mid and q_mid in df.columns:
            fig.add_trace(go.Scatter(
                x=x, y=df[q_mid], name=f"q50 h{plot_horizon}",
                mode="lines", line=dict(color=_CLOSE, width=2.0),
            ), row=1, col=1)
    else:
        # OHLC mode: high/low dashed + close gold
        if high_col and high_col in df.columns:
            fig.add_trace(go.Scatter(
                x=x, y=df[high_col], name=f"Pred high h{plot_horizon}",
                mode="lines", line=dict(color=_HIGH_CLR, width=0.9, dash="dash"),
            ), row=1, col=1)
        if low_col and low_col in df.columns:
            fig.add_trace(go.Scatter(
                x=x, y=df[low_col], name=f"Pred low h{plot_horizon}",
                mode="lines", line=dict(color=_LOW_CLR, width=0.9, dash="dash"),
            ), row=1, col=1)
        if close_col and close_col in df.columns:
            fig.add_trace(go.Scatter(
                x=x, y=df[close_col], name=f"Pred close h{plot_horizon}",
                mode="lines", line=dict(color=_CLOSE, width=2.0),
            ), row=1, col=1)

    # ── Row 2 ────────────────────────────────────────────────────────────────
    if prob_mode:
        q50_col = next((c for c in q_cols if c.startswith("q50_")), None)
        if q50_col and q50_col in df.columns and "actual" in df.columns:
            err = df[q50_col] - df["actual"]
            fig.add_trace(go.Scatter(
                x=x, y=err, name=f"q50−actual (h{plot_horizon})",
                mode="lines",
                line=dict(color=_CLOSE, width=1.2),
                fill="tozeroy",
                fillcolor="rgba(255,193,7,0.15)",
            ), row=2, col=1)
    else:
        if spread_col and spread_col in df.columns:
            fig.add_trace(go.Scatter(
                x=x, y=df[spread_col], name=f"Spread h{plot_horizon}",
                mode="lines",
                line=dict(color=_SPREAD, width=1.2),
                fill="tozeroy",
                fillcolor="rgba(41,98,255,0.15)",
            ), row=2, col=1)

    # ── Row 3: staleness ──────────────────────────────────────────────────────
    if "staleness" in df.columns:
        fig.add_trace(go.Scatter(
            x=x, y=df["staleness"], name="Staleness",
            mode="lines", line=dict(color="#787b86", width=1.0),
        ), row=3, col=1)

    # ── Re-run boundary markers ───────────────────────────────────────────────
    if "run_id" in df.columns:
        rerun_times = df.index[df["run_id"].diff().fillna(0) != 0].tolist()
        if rerun_times:
            shapes = []
            for ts in rerun_times:
                for yref in ("y domain", "y2 domain"):
                    shapes.append(dict(
                        type="line", xref="x", yref=yref,
                        x0=ts, x1=ts, y0=0, y1=1,
                        line=dict(color=_RERUN, dash="dot", width=1),
                    ))
            fig.update_layout(shapes=shapes)

    # ── Global layout ─────────────────────────────────────────────────────────
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        height=900,
        margin=dict(l=60, r=40, t=60, b=40),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
            font=dict(color="#d1d4dc"),
        ),
        hovermode="x unified",
    )

    for row in range(1, 4):
        fig.update_yaxes(
            gridcolor=_GRID, zeroline=False,
            tickfont=dict(color=_TICK_CLR), row=row, col=1,
        )

    rangebreaks = [dict(bounds=["sat", "mon"])] if params.get("weekends") == "nogap" else []
    fig.update_xaxes(
        gridcolor=_GRID, tickfont=dict(color=_TICK_CLR),
        showspikes=True, spikecolor=_TICK_CLR, spikethickness=1,
        rangebreaks=rangebreaks,
    )

    fig.update_layout(xaxis3=dict(
        rangeselector=dict(
            buttons=[
                dict(count=7,  label="1W",  step="day",   stepmode="backward"),
                dict(count=1,  label="1M",  step="month", stepmode="backward"),
                dict(count=3,  label="3M",  step="month", stepmode="backward"),
                dict(count=6,  label="6M",  step="month", stepmode="backward"),
                dict(step="all", label="All"),
            ],
            bgcolor=_GRID, activecolor="#2962ff",
            font=dict(color="#d1d4dc"),
        ),
    ))

    return fig


# ── Public entry point ────────────────────────────────────────────────────────

def plot_file(
    parquet_path: str | Path,
    plot_start: str | None = None,
    plot_end: str | None = None,
    plot_horizon: int | None = None,
    threshold: float = 6e-4,
    histdata_dir: str | Path | None = None,
    save_dir: str | Path | None = None,
) -> go.Figure:
    """Load a Kronos features parquet, run the matching pipeline, build the figure.

    Parameters
    ----------
    parquet_path : path to the parquet (full path or filename — PROJ/featdata/ assumed)
    plot_start   : ISO date string — restrict plot window start
    plot_end     : ISO date string — restrict plot window end
    plot_horizon : which horizon to display (e.g. 10 shows close_h10, high_h10, low_h10).
                   Defaults to smallest horizon present in the parquet.
    threshold    : fracdiff weight truncation threshold (must match generation).
                   Default 6e-4. Not encoded in filename — pass explicitly if non-default.
    histdata_dir : M1 CSV directory (defaults to PROJ/histdata/)
    save_dir     : where to write HTML (defaults to same directory as parquet)

    Returns
    -------
    go.Figure — also written to <stem>_h{plot_horizon}_plot.html in save_dir
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.is_absolute() and not parquet_path.exists():
        parquet_path = _PROJ / "featdata" / parquet_path.name
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    if histdata_dir is None:
        histdata_dir = _PROJ / "histdata"
    if save_dir is None:
        save_dir = parquet_path.parent

    params = _parse_fname(parquet_path.name)
    print(f"[Params] {params}")

    kronos_df = pd.read_parquet(parquet_path)
    print(f"[Loaded] {parquet_path.name}  ({len(kronos_df):,} rows)")

    schema, prob_mode = _discover_schema(kronos_df)
    if not schema:
        raise ValueError(
            f"No Kronos feature columns found in {parquet_path.name}. "
            f"Columns present: {list(kronos_df.columns)}"
        )

    print(f"[Mode] {'probabilistic (quantile fan)' if prob_mode else 'OHLC (point forecast)'}")

    available_horizons = sorted(schema.keys())
    if plot_horizon is None:
        plot_horizon = available_horizons[0]
    if plot_horizon not in schema:
        raise ValueError(
            f"plot_horizon={plot_horizon} not found. Available: {available_horizons}"
        )

    norm_close = _run_pipeline(params, Path(histdata_dir), threshold=threshold)

    df = kronos_df.join(norm_close.rename("actual"), how="left")
    df["actual"] = df["actual"].bfill()

    if plot_start is not None:
        df = df.loc[plot_start:]
    if plot_end is not None:
        df = df.loc[:plot_end]

    if len(df) == 0:
        raise ValueError(f"No data in plot window {plot_start} → {plot_end}")

    period_tag = ""
    if plot_start or plot_end:
        period_tag = f"_{(plot_start or '').replace('-', '')}_{(plot_end or '').replace('-', '')}"

    fig = _build_figure(df, params, schema, prob_mode, plot_horizon)
    stem = parquet_path.stem
    out  = Path(save_dir) / f"{stem}_h{plot_horizon}{period_tag}_plot.html"
    fig.write_html(str(out))
    print(f"[Saved] {out}")
    return fig


# ── Script entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":

    PARQUET      = "EURUSD_H1_kron_ctx512_int5_h5-10-15-20_fdiff0.3_wfilled_snone_2023.parquet"
    PLOT_HORIZON = 5      # which horizon to display; None = smallest available

    PLOT_START = "2023-06-01"   # ISO date e.g. "2020-06-01" or "full"
    PLOT_END   = "2023-06-18"   # ISO date e.g. "2020-09-30" or "full"

    fig = plot_file(
        PARQUET,
        plot_horizon = PLOT_HORIZON,
        plot_start   = None if PLOT_START == "full" else PLOT_START,
        plot_end     = None if PLOT_END   == "full" else PLOT_END,
    )
    fig.show()
