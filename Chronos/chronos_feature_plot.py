"""
chronos_feature_plot.py
=======================
Compare Chronos quantile predictions from a featdata parquet against the
pipeline's normalized close price for the same period.

All parameters are parsed from the parquet filename — no manual re-entry.
Quantile columns are discovered dynamically from the parquet schema.

Usage (CLI):
    python Chronos/chronos_feature_plot.py featdata/EURUSD_H1_ctx504_int10_h5-10-15-20_logret_wfilled_snone_2023.parquet

Usage (notebook / REPL):
    from Chronos.chronos_feature_plot import plot_file
    fig = plot_file("featdata/EURUSD_H1_ctx504_int10_h5-10-15-20_logret_wfilled_snone_2023.parquet",
                    plot_horizon=10)
    fig.show()

Output: HTML file written next to the input parquet (same directory, _plot suffix).

Layout
------
  Row 1 (55%) — normalized close (actual) + quantile lines for plot_horizon
  Row 2 (30%) — forecast error: q50_h{plot_horizon} − actual
  Row 3 (15%) — staleness (bars since last Chronos run, 0 = fresh)

Faint vertical dotted lines mark every Chronos re-run boundary (run_id change).
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
        if (p / "Pipeline").is_dir() and (p / "Chronos").is_dir():
            return p
        p = p.parent
    return start.parent.parent

_PROJ = _find_proj()
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from Pipeline.pipeline import ForexDataLoader, ForexPipeline  # noqa: E402

# ── Colour palette ────────────────────────────────────────────────────────────

# Fixed 5-slot palette ordered lowest → highest percentile (pessimistic → optimistic)
_RANK_STYLES: list[dict[str, Any]] = [
    dict(color="#ef5350", width=0.8, dash="dot"),    # lowest  (pessimistic)
    dict(color="#ff8a65", width=1.0, dash="dash"),
    dict(color="#ffc107", width=2.0, dash="solid"),  # middle  (median)
    dict(color="#80cbc4", width=1.0, dash="dash"),
    dict(color="#26a69a", width=0.8, dash="dot"),    # highest (optimistic)
]

_BG       = "#131722"
_GRID     = "#1e222d"
_TICK_CLR = "#787b86"
_ACTUAL   = "#d1d4dc"
_ERR_POS  = "rgba(38,166,154,0.35)"
_ERR_NEG  = "rgba(239,83,80,0.35)"
_RERUN    = "rgba(120,123,134,0.20)"

# ── Column schema discovery ───────────────────────────────────────────────────

_COL_RE = re.compile(r"^q(\d+)_h(\d+)$")


def _discover_schema(df: pd.DataFrame) -> dict[int, list[str]]:
    """
    Scan df.columns for quantile columns of the form q{pp}_h{hh}.
    Returns {horizon: [col, col, ...]} sorted by percentile value within each horizon.
    """
    schema: dict[int, list[tuple[int, str]]] = {}
    for col in df.columns:
        m = _COL_RE.match(col)
        if m:
            pct, h = int(m.group(1)), int(m.group(2))
            schema.setdefault(h, []).append((pct, col))
    return {h: [col for _, col in sorted(cols)] for h, cols in schema.items()}


def _rank_styles(n: int) -> list[dict[str, Any]]:
    """
    Return n style dicts from _RANK_STYLES.  If n == 5 use directly.
    For other sizes pick from the palette by evenly spaced indices.
    """
    if n == len(_RANK_STYLES):
        return _RANK_STYLES
    indices = [round(i * (len(_RANK_STYLES) - 1) / (n - 1)) for i in range(n)] if n > 1 else [2]
    return [_RANK_STYLES[i] for i in indices]


# ── Filename parser ───────────────────────────────────────────────────────────

_FNAME_RE = re.compile(
    r"^(?P<pair>[A-Z]+)_(?P<tf>[A-Z0-9]+)"
    r"_ctx(?P<ctx>\d+)"
    r"(?:_pred(?P<pred>\d+))?"                           # optional: old-format pred tag
    r"_int(?P<intv>\d+)"
    r"(?:_h(?P<horizons>[\d]+(?:-[\d]+)*))?"             # optional: new-format horizons tag
    r"_(?P<norm>logret|fdiff[\d.]+|raw)"
    r"(?:_(?P<wknd>wfilled|wnogap|wgaps))?"              # optional; old files default wfilled
    r"(?:_(?P<scale>snone|sroll\d+|sglob))?"             # optional; old files default snone
    r"_(?P<year_tag>[\d_]+)\.parquet$"
)


def _parse_fname(fname: str) -> dict:
    m = _FNAME_RE.match(fname)
    if not m:
        raise ValueError(
            f"Cannot parse parameters from filename {fname!r}.\n"
            f"Expected format: PAIR_TF_ctxN_intN_h<H1>-<H2>-..._<norm>_YEAR.parquet"
        )
    g = m.groupdict()

    norm_raw = g["norm"]
    if norm_raw == "logret":
        norm_method = "log_returns"
        fracdiff_d  = 0.4
    elif norm_raw.startswith("fdiff"):
        norm_method = "fracdiff"
        fracdiff_d  = float(norm_raw[5:])
    else:
        norm_method = "raw"
        fracdiff_d  = 0.4

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
        pair             = g["pair"],
        timeframe        = g["tf"],
        context_length   = int(g["ctx"]),
        prediction_length= int(g["pred"]) if g.get("pred") else None,  # old format only
        calc_interval    = int(g["intv"]),
        horizons         = horizons,
        norm_method      = norm_method,
        fracdiff_d       = fracdiff_d,
        weekends         = weekends,
        scaling          = scaling,
        scaling_window   = scaling_window,
        years            = years,
        start            = start,
        end              = end,
    )


# ── Pipeline runner ───────────────────────────────────────────────────────────

def _run_pipeline(params: dict, histdata_dir: Path, threshold: float = 6e-4) -> pd.Series:
    """
    Return normalized close series covering the full available date range.

    Intentionally does NOT slice df_m1 to params["start"/"end"] before running
    the pipeline.  The fracdiff normalization needs warm-up bars before any
    requested start date; slicing early would produce NaN throughout the
    warm-up window.  The join in plot_file() restricts the output to the
    chronos_df date range, so there is no correctness issue with returning
    more data than needed.
    """
    loader   = ForexDataLoader()
    df_m1    = loader.load_and_merge(
        str(histdata_dir),
        pair     = params["pair"],
        years    = params["years"],
        weekends = params["weekends"],
    )
    # Do NOT slice df_m1 here — pipeline needs prior bars for fracdiff warm-up.
    # The join in plot_file aligns to the chronos_df date range automatically.

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
    schema: dict[int, list[str]],
    plot_horizon: int,
) -> go.Figure:
    """
    df must have columns: q{pp}_h{plot_horizon} (for each percentile),
                          staleness, run_id, actual
    """
    q_cols  = schema[plot_horizon]
    n_q     = len(q_cols)
    styles  = _rank_styles(n_q)
    q50_col = next((c for c in q_cols if c.startswith("q50_")), q_cols[n_q // 2])

    horizons_str = (
        ", ".join(str(h) for h in sorted(schema.keys()))
        if params.get("horizons") else "?"
    )
    title = (
        f"{params['pair']} {params['timeframe']} — "
        f"Chronos ctx={params['context_length']} "
        f"int={params['calc_interval']} "
        f"horizons=[{horizons_str}]  "
        f"plotting h={plot_horizon}  "
        f"[{params['norm_method']}]"
    )

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.55, 0.30, 0.15],
        subplot_titles=[title, f"Forecast error  ({q50_col} − actual)", "Staleness"],
    )

    for ann in fig.layout.annotations:
        ann.font.color = "#d1d4dc"
        ann.font.size  = 11

    x = df.index

    # ── Row 1: actual + quantile lines for selected horizon ───────────────────
    fig.add_trace(go.Scatter(
        x=x, y=df["actual"],
        name="Actual",
        mode="lines",
        line=dict(color=_ACTUAL, width=1.5),
    ), row=1, col=1)

    for col, style in zip(q_cols, styles):
        pct_label = col.split("_")[0].upper()   # e.g. "Q05"
        fig.add_trace(go.Scatter(
            x=x, y=df[col],
            name=f"{pct_label} h{plot_horizon}",
            mode="lines",
            line=dict(color=style["color"], width=style["width"], dash=style["dash"]),
        ), row=1, col=1)

    # ── Row 2: forecast error ─────────────────────────────────────────────────
    err = df[q50_col] - df["actual"]
    fig.add_trace(go.Scatter(
        x=x, y=err,
        name=f"{q50_col} error",
        mode="lines",
        line=dict(color="#ffc107", width=1.0),
        fill="tozeroy",
        fillcolor=_ERR_NEG,
    ), row=2, col=1)
    pos_err = err.clip(lower=0)
    fig.add_trace(go.Scatter(
        x=x, y=pos_err,
        name="error (+)",
        mode="lines",
        line=dict(color="#ffc107", width=0),
        fill="tozeroy",
        fillcolor=_ERR_POS,
        showlegend=False,
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color=_TICK_CLR, row=2, col=1)

    # ── Row 3: staleness ──────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x, y=df["staleness"],
        name="Staleness",
        mode="lines",
        line=dict(color="#2962ff", width=1.0),
    ), row=3, col=1)

    # ── Re-run boundary markers ───────────────────────────────────────────────
    rerun_times = df.index[df["run_id"].diff().fillna(0) != 0].tolist()
    if rerun_times:
        shapes = []
        for ts in rerun_times:
            for yref in ("y domain", "y2 domain"):
                shapes.append(dict(
                    type="line",
                    xref="x", yref=yref,
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
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            font=dict(color="#d1d4dc"),
        ),
        hovermode="x unified",
    )

    for row in range(1, 4):
        fig.update_yaxes(
            gridcolor=_GRID,
            zeroline=False,
            tickfont=dict(color=_TICK_CLR),
            row=row,
            col=1,
        )

    rangebreaks = [dict(bounds=["sat", "mon"])] if params.get("weekends") == "nogap" else []
    fig.update_xaxes(
        gridcolor=_GRID,
        tickfont=dict(color=_TICK_CLR),
        showspikes=True,
        spikecolor=_TICK_CLR,
        spikethickness=1,
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
            bgcolor=_GRID,
            activecolor="#2962ff",
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
    """
    Load a Chronos features parquet, run the matching pipeline, and build the
    comparison figure.

    Parameters
    ----------
    parquet_path : path to the parquet file (full path or just filename —
                   in that case PROJ/featdata/ is assumed)
    plot_start   : ISO date string — restrict plot window start
    plot_end     : ISO date string — restrict plot window end
    plot_horizon : which horizon to display (e.g. 5 shows q05_h5…q95_h5).
                   Defaults to the smallest horizon present in the parquet.
    threshold    : fracdiff weight truncation threshold passed to ForexPipeline.
                   Must match the value used when generating the Chronos features.
                   Default 6e-4 (pipeline default).  Not encoded in the filename —
                   pass explicitly if you used a non-default value.
    histdata_dir : M1 CSV directory (defaults to PROJ/histdata/)
    save_dir     : where to write the HTML (defaults to same directory as parquet)

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

    chronos_df = pd.read_parquet(parquet_path)
    print(f"[Loaded] {parquet_path.name}  ({len(chronos_df):,} rows)")

    schema = _discover_schema(chronos_df)
    if not schema:
        raise ValueError(
            f"No quantile columns (q{{pp}}_h{{hh}}) found in {parquet_path.name}. "
            f"Columns present: {list(chronos_df.columns)}"
        )

    available_horizons = sorted(schema.keys())
    if plot_horizon is None:
        plot_horizon = available_horizons[0]
    if plot_horizon not in schema:
        raise ValueError(
            f"plot_horizon={plot_horizon} not found. "
            f"Available horizons: {available_horizons}"
        )

    norm_close = _run_pipeline(params, Path(histdata_dir), threshold=threshold)

    df = chronos_df.join(norm_close.rename("actual"), how="left")
    df["actual"] = df["actual"].bfill()

    if plot_start is not None:
        df = df.loc[plot_start:]
    if plot_end is not None:
        df = df.loc[:plot_end]

    if len(df) == 0:
        raise ValueError(f"No data in plot window {plot_start} → {plot_end}")

    period_tag = ""
    if plot_start or plot_end:
        s = plot_start or str(df.index[0].date())
        e = plot_end   or str(df.index[-1].date())
        period_tag = f"_{s}_{e}"

    fig      = _build_figure(df, params, schema, plot_horizon)
    out_path = Path(save_dir) / (parquet_path.stem + f"_h{plot_horizon}" + period_tag + "_plot.html")
    fig.write_html(str(out_path), auto_open=False)
    print(f"[Saved]  {out_path}")

    return fig


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Configure here ────────────────────────────────────────────────────────
    PARQUET      = "EURUSD_H1_ctx504_int10_h5-10-15-20_fdiff0.3_wfilled_snone_2023.parquet"
    PLOT_HORIZON = 5      # which horizon to display; None = smallest available

    PLOT_START = "2023-06-01"  # ISO date e.g. "2023-06-01" or "full"
    PLOT_END   = "2023-06-18"  # ISO date e.g. "2023-06-18" or "full"
    # ─────────────────────────────────────────────────────────────────────────

    plot_file(
        PARQUET,
        plot_horizon = PLOT_HORIZON,
        plot_start   = None if PLOT_START == "full" else PLOT_START,
        plot_end     = None if PLOT_END   == "full" else PLOT_END,
    )
