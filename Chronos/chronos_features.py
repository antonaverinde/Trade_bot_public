"""
chronos_features.py
===================
Chronos-2 quantile feature generator for XGBoost.

Slides a context window along historical data, running Chronos every
calc_interval bars, and produces a feature DataFrame aligned to the
same datetime index as ForexPipeline output (ready for pd.concat/join).

Output columns
--------------
For each combination of percentile p and horizon h:
    q{pp}_h{h}          normalised predicted price at horizon h bars ahead
                        (pp = int(p*100) zero-padded, e.g. q05, q50, q95)
Additional columns:
    run_id              which Chronos run produced this row
    staleness           bars since the last Chronos run (0 = fresh run bar,
                        1…calc_interval-1 = copied from previous run)

Example: horizons=[5,10,15], percentiles=[0.05,0.25,0.50,0.75,0.95]
    → columns q05_h5  q25_h5  q50_h5  q75_h5  q95_h5
              q05_h10 q25_h10 q50_h10 q75_h10 q95_h10
              q05_h15 q25_h15 q50_h15 q75_h15 q95_h15
              run_id  staleness

calc_interval behaviour
-----------------------
Chronos re-runs every calc_interval bars.  The same predictions are
copied verbatim to the next calc_interval-1 bars; staleness encodes
how many bars ago the last run was.

  Example: horizons=[5,10], calc_interval=5
    run at bar 504 (context = closes[0:504]):
        predictions for bars 504+5=509 and 504+10=514 are recorded.
        These values are assigned to bars 504,505,506,507,508
        with staleness 0,1,2,3,4 respectively.
    run at bar 509: predictions for bars 514,519 → assigned to 509…513.

Output filename
---------------
  {pair}_{tf}_ctx{C}_int{I}_h{h1}-{h2}-..._{norm}_{wknd}_{scale}_{year}.parquet
  saved inside  <project_root>/featdata/

Usage (script)
--------------
    python Chronos/chronos_features.py

Usage (import)
--------------
    from Chronos.chronos_features import generate
    df = generate(pair="EURUSD", years=[2023], timeframe="H1",
                  horizons=[5, 10, 15], percentiles=[0.05, 0.25, 0.50, 0.75, 0.95])
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _find_proj() -> Path:
    """Walk up from __file__ (or cwd in notebooks) to find the project root."""
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


PROJ = _find_proj()
sys.path.insert(0, str(PROJ))

from Pipeline.pipeline import (          # noqa: E402
    ForexDataLoader, resample_ohlcv, TIMEFRAMES, _fracdiff_weights,
)
from Chronos.chronos_inference import ChronosForecaster  # noqa: E402


def _pname(p: float) -> str:
    """Convert quantile float to two-digit column prefix: 0.05 → 'q05'."""
    return f"q{int(round(p * 100)):02d}"


# ─────────────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(
    raw: np.ndarray,
    context_close: float,
    actual_history: np.ndarray,
    norm_method: str,
    d: float,
    threshold: float,
) -> np.ndarray:
    """
    Normalize a (n_quantiles, pred_len) matrix of raw Chronos prices.

    raw            : (n_q, pred_len) — Chronos prices, each row one quantile path
    context_close  : last actual close price (actual_history[-1])
    actual_history : all actual closes up to context_close, oldest→newest
    norm_method    : "log_returns" | "fracdiff" | "raw"
    d              : fracdiff order (ignored unless norm_method="fracdiff")

    Returns array of same shape as raw.
    """
    if norm_method == "raw":
        return raw.copy()

    if norm_method == "log_returns":
        return np.log(raw / context_close)

    # fracdiff
    weights = _fracdiff_weights(d=d, threshold=threshold)   # shape (W,) oldest first
    W = len(weights)
    n_q, pred_len = raw.shape
    result = np.zeros_like(raw, dtype=np.float64)

    for q in range(n_q):
        for h in range(pred_len):
            pred_path = raw[q, : h + 1]
            n_actual  = W - len(pred_path)
            if n_actual > 0:
                hist_slice = actual_history[-n_actual:]
                window = np.concatenate([hist_slice, pred_path])
            else:
                window = pred_path[-W:]

            if len(window) < W:
                pad    = np.full(W - len(window), window[0])
                window = np.concatenate([pad, window])

            result[q, h] = float(np.dot(weights, window))

    return result.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Plot helper
# ─────────────────────────────────────────────────────────────────────────────

def _save_run_plot(
    df: pd.DataFrame,
    fc: dict,
    context_end_idx: int,
    max_horizon: int,
    plot_dir: Path,
    run_id: int,
    pair: str,
    timeframe: str,
) -> None:
    try:
        from Chronos.chronos_plots import plot_forecast, save_png
    except ImportError:
        return

    if len(df.index) < 2:
        return

    bar_td         = df.index[1] - df.index[0]
    context_end_ts = df.index[context_end_idx]
    pred_ts = pd.date_range(
        start=context_end_ts + bar_td,
        periods=max_horizon,
        freq=bar_td,
    )
    context_df = df.iloc[max(0, context_end_idx - 49): context_end_idx + 1]

    after_idx = context_end_idx + 1
    gt = df.iloc[after_idx: after_idx + max_horizon] if after_idx < len(df) else None

    plot_result = {
        "context_df":          context_df,
        "forecast":            fc,
        "ground_truth_df":     gt,
        "forecast_timestamps": pred_ts,
        "context_end":         context_end_ts,
        "forecast_start":      pred_ts[0],
        "prediction_length":   max_horizon,
        "pair":                pair,
        "timeframe":           timeframe,
    }

    fname = plot_dir / f"run{run_id:05d}_{context_end_ts.strftime('%Y%m%d_%H%M')}.png"
    fig   = plot_forecast(plot_result, context_bars_shown=50)
    save_png(fig, str(fname))


# ─────────────────────────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────────────────────────

def generate(
    pair: str = "EURUSD",
    years: list[int] | None = None,
    start: str | None = None,
    end: str | None = None,
    timeframe: str = "H1",
    context_length: int = 504,
    horizons: list[int] | None = None,
    percentiles: list[float] | None = None,
    calc_interval: int = 1,
    norm_method: str = "log_returns",
    fracdiff_d: float = 0.4,
    threshold: float = 6e-4,
    weekends: str = "filled",
    scaling: str = "none",
    scaling_window: int = 200,
    histdata_dir: str | Path | None = None,
    save_dir: str | Path | None = None,
    device: str = "auto",
    save_plots: bool = False,
) -> pd.DataFrame:
    """
    Generate Chronos quantile features and save to parquet.

    Parameters
    ----------
    pair             : currency pair, e.g. "EURUSD"
    years            : list of years to load; None = all CSVs in histdata_dir
    start / end      : ISO date strings for sub-range slicing after year filter
    timeframe        : "M1" | "M5" | "M15" | "H1" | "H4" | "D1"
    context_length   : bars of history fed to Chronos per run (≤ 8192)
    horizons         : list of prediction horizons to record, e.g. [5, 10, 15, 20].
                       Chronos is run up to max(horizons) bars ahead.
                       Hard max is 1024. Defaults to [5, 10, 15, 20].
    percentiles      : quantile levels to record, e.g. [0.05, 0.25, 0.50, 0.75, 0.95].
                       Must be in [0, 1].  Defaults to [0.05, 0.25, 0.50, 0.75, 0.95].
    calc_interval    : re-run Chronos every N bars; copy the same predictions to
                       the next N-1 bars (staleness 1…N-1). Default 1 = run every bar.
    norm_method      : "log_returns" | "fracdiff" | "raw"
    fracdiff_d       : fractional diff order (fracdiff mode only)
    threshold        : fracdiff weight truncation threshold
    weekends         : "filled" | "nogap" | "gaps"
    scaling          : "none" | "rolling" | "global" — z-score normalisation
                       applied to all quantile columns after generation
    scaling_window   : look-back bars for rolling z-score
    histdata_dir     : path to HistData M1 CSV folder
    save_dir         : output root for parquet (and plots/ subfolder)
    device           : "auto" | "cuda" | "cpu"
    save_plots       : save a Chronos fan-chart PNG per run into save_dir/plots/

    Returns
    -------
    pd.DataFrame
        DatetimeIndex aligned to the resampled timeframe bars.
        Columns: q{pp}_h{h} for each percentile × horizon combination,
                 plus run_id (int) and staleness (int, 0…calc_interval-1).
        First context_length bars have no predictions (absent from the index).
    """
    if horizons is None:
        horizons = [5, 10, 15, 20]
    if percentiles is None:
        percentiles = [0.05, 0.25, 0.50, 0.75, 0.95]

    horizons    = sorted(set(horizons))
    percentiles = sorted(set(percentiles))
    max_horizon = max(horizons)

    assert max_horizon <= 1024, "Chronos hard max is 1024 bars"
    assert calc_interval >= 1,  "calc_interval must be ≥ 1"
    assert all(h >= 1 for h in horizons), "All horizons must be ≥ 1"
    assert all(0.0 < p < 1.0 for p in percentiles), "Percentiles must be in (0, 1)"

    pnames = [_pname(p) for p in percentiles]
    col_order = [f"{pn}_h{h}" for h in horizons for pn in pnames]

    if histdata_dir is None:
        histdata_dir = PROJ / "histdata"
    if save_dir is None:
        save_dir = PROJ / "featdata"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Load and resample ─────────────────────────────────────────────────────
    loader = ForexDataLoader()
    df_m1  = loader.load_and_merge(str(histdata_dir), pair=pair, years=years, weekends=weekends)

    if start or end:
        df_m1 = df_m1[start:end]

    df = df_m1.copy() if timeframe == "M1" else resample_ohlcv(df_m1, TIMEFRAMES[timeframe])
    print(f"[Chronos-feats] {pair} {timeframe}: {len(df):,} bars  "
          f"{df.index[0]} → {df.index[-1]}")

    closes     = df["close"].values.astype(np.float32)
    timestamps = df.index
    n          = len(closes)

    if n < context_length + calc_interval:
        raise ValueError(
            f"Not enough bars: need ≥ {context_length + calc_interval}, got {n}."
        )

    # ── Chronos quantile levels to request ───────────────────────────────────
    # We ask Chronos for all required percentile levels.
    from Chronos.chronos_inference import MODEL_QUANTILES  # noqa: E402
    # MODEL_QUANTILES are the fixed quantile levels the model always outputs.
    # Map requested percentiles to the closest available model quantile.
    def _nearest_q(p: float) -> float:
        return min(MODEL_QUANTILES, key=lambda q: abs(q - p))

    model_qs = [_nearest_q(p) for p in percentiles]

    # ── Load Chronos ──────────────────────────────────────────────────────────
    forecaster = ChronosForecaster(
        context_length=context_length,
        device=device,
        dtype=torch.bfloat16,
    )
    forecaster.load()

    # ── Plot dir ──────────────────────────────────────────────────────────────
    plot_dir: Path | None = None
    if save_plots:
        plot_dir = save_dir / "plots" / f"{pair}_{timeframe}"
        plot_dir.mkdir(parents=True, exist_ok=True)

    # ── Main loop ─────────────────────────────────────────────────────────────
    rows          = []
    run_positions = list(range(context_length, n, calc_interval))
    n_runs        = len(run_positions)
    log_every     = max(1, n_runs // 20)

    for run_id, i in enumerate(run_positions):
        steps = min(calc_interval, n - i)
        if steps <= 0:
            break

        actual_history = closes[:i]
        context_close  = float(closes[i - 1])

        # ── Chronos inference ─────────────────────────────────────────────────
        fc  = forecaster.forecast(actual_history, prediction_length=max_horizon)
        qs  = fc["quantiles"]   # dict {float: np.ndarray(max_horizon)}

        # (n_q, max_horizon) — raw Chronos prices at requested quantile levels
        raw_matrix = np.stack([qs[mq] for mq in model_qs], axis=0)

        # (n_q, max_horizon) — normalised
        norm_matrix = _normalize(
            raw_matrix, context_close, actual_history, norm_method, fracdiff_d, threshold
        )

        # ── Emit rows (one per bar in this run's block) ───────────────────────
        for k in range(steps):
            ts  = timestamps[i + k]
            row = {"datetime": ts, "run_id": run_id, "staleness": k}
            for h in horizons:
                for qi, pn in enumerate(pnames):
                    row[f"{pn}_h{h}"] = float(norm_matrix[qi, h - 1])
            rows.append(row)

        # ── Optional plot ─────────────────────────────────────────────────────
        if save_plots and plot_dir is not None:
            _save_run_plot(
                df=df, fc=fc, context_end_idx=i - 1,
                max_horizon=max_horizon,
                plot_dir=plot_dir, run_id=run_id, pair=pair, timeframe=timeframe,
            )

        if (run_id + 1) % log_every == 0 or run_id == n_runs - 1:
            pct = 100 * (run_id + 1) / n_runs
            print(f"  run {run_id+1:>5}/{n_runs}  ({pct:4.0f}%)  @ {timestamps[i]}")

    if not rows:
        raise RuntimeError("No prediction rows generated — check date range vs context_length.")

    result = (
        pd.DataFrame(rows)
        .set_index("datetime")
    )
    result.index.name = "datetime"
    # enforce deterministic column order
    result = result[col_order + ["run_id", "staleness"]]

    # ── Scale quantile columns ────────────────────────────────────────────────
    q_cols = col_order
    if scaling == "rolling":
        for col in q_cols:
            s   = result[col]
            mu  = s.rolling(scaling_window, min_periods=1).mean()
            std = s.rolling(scaling_window, min_periods=1).std().fillna(1.0).replace(0.0, 1.0)
            result[col] = ((s - mu) / std).astype(np.float32)
    elif scaling == "global":
        for col in q_cols:
            s   = result[col]
            mu  = float(s.mean())
            std = float(s.std()) or 1.0
            result[col] = ((s - mu) / std).astype(np.float32)

    # ── Output filename ───────────────────────────────────────────────────────
    if years:
        year_tag = "_".join(str(y) for y in sorted(years))
    elif start or end:
        s_tag = (start or str(timestamps[context_length].date())).replace("-", "")
        e_tag = (end   or str(timestamps[-1].date())).replace("-", "")
        year_tag = f"{s_tag}_{e_tag}"
    else:
        year_tag = f"{timestamps[0].year}_{timestamps[-1].year}"

    norm_tag  = (
        "logret"              if norm_method == "log_returns"
        else f"fdiff{fracdiff_d}" if norm_method == "fracdiff"
        else "raw"
    )
    h_tag     = "h" + "-".join(str(h) for h in horizons)
    wknd_tag  = "w" + weekends
    scale_tag = (
        "snone"                       if scaling == "none"
        else f"sroll{scaling_window}" if scaling == "rolling"
        else "sglob"
    )
    fname = (
        f"{pair}_{timeframe}"
        f"_ctx{context_length}_int{calc_interval}_{h_tag}"
        f"_{norm_tag}_{wknd_tag}_{scale_tag}_{year_tag}.parquet"
    )
    out_path = save_dir / fname
    result.to_parquet(out_path)
    print(f"[Saved] {out_path}  ({len(result):,} rows, {len(result.columns)} cols)")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    generate(
        pair           = "EURUSD",
        years          = [2023],
        timeframe      = "H1",
        context_length = 512,
        horizons       = [5, 10, 15, 20],
        percentiles    = [0.05, 0.25, 0.50, 0.75, 0.95],
        calc_interval  = 20,
        norm_method    = "fracdiff",
        fracdiff_d     = 0.3,
        threshold      = 6e-4,
        weekends       = "filled",
        scaling        = "none",
        histdata_dir   = PROJ / "histdata",
        save_dir       = PROJ / "featdata",
        device         = "auto",
        save_plots     = False,
    )
