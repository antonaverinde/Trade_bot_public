"""
kronos_features.py
==================
Kronos OHLCV feature generator for XGBoost.

Slides a context window along historical data, running Kronos every
calc_interval bars, and produces a feature DataFrame aligned to the
same datetime index as ForexPipeline output (ready for pd.concat/join).

Output columns depend on n_samples:

n_samples == 1  (point-forecast mode)
--------------------------------------
    close_h{h}      normalised predicted close at h bars ahead
    high_h{h}       normalised predicted high at h bars ahead
    low_h{h}        normalised predicted low at h bars ahead
    spread_h{h}     log(pred_high / pred_low) — intra-candle volatility proxy

n_samples > 1   (probabilistic mode)
--------------------------------------
    q{pp}_h{h}      percentile pp of predicted close across n_samples paths
                    e.g. q05_h10, q50_h10, q95_h10
                    Percentiles controlled by the `percentiles` argument.
                    OHLC columns are NOT written in this mode.

Additional columns (both modes):
    run_id          which Kronos run produced this row
    staleness       bars since the last Kronos run (0 = fresh run bar)

Normalisation
-------------
    log_returns  : log(pred_X / context_close)    for close / high / low
    fracdiff     : fracdiff on close (same as Chronos);
                   high/low expressed as log(pred_X / pred_close)
    raw          : raw predicted prices, no transform

Output filename
---------------
  {pair}_{tf}_kron_ctx{C}_int{I}_h{h1}-{h2}-..._{norm}_{wknd}_{scale}[_s{N}]_{year}.parquet
  saved inside  <project_root>/featdata/

Usage (script)
--------------
    python Kronos/kronos_features.py

Usage (import)
--------------
    from Kronos.kronos_features import generate
    df = generate(pair="EURUSD", years=[2023], timeframe="H1",
                  horizons=[5, 10, 15, 20], calc_interval=10)
    # probabilistic mode:
    df = generate(pair="EURUSD", years=[2023], timeframe="H1",
                  horizons=[5, 10, 15, 20], n_samples=20,
                  percentiles=[0.05, 0.25, 0.50, 0.75, 0.95])
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


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


PROJ = _find_proj()
sys.path.insert(0, str(PROJ))

from Pipeline.pipeline import (          # noqa: E402
    ForexDataLoader, resample_ohlcv, TIMEFRAMES, _fracdiff_weights,
)
from Kronos.kronos_inference import KronosForecaster, OHLC_COLS  # noqa: E402

OHLCV_COLS = ["open", "high", "low", "close"]


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _norm_close(
    raw_close: float,
    context_close: float,
    actual_history: np.ndarray,
    norm_method: str,
    d: float,
    threshold: float,
    h: int,
    pred_close_path: np.ndarray,
) -> float:
    """Normalise a single predicted close price."""
    if norm_method == "raw":
        return float(raw_close)
    if norm_method == "log_returns":
        return math.log(raw_close / context_close + 1e-12)

    # fracdiff: same approach as Chronos
    weights = _fracdiff_weights(d=d, threshold=threshold)
    W = len(weights)
    pred_path = pred_close_path[: h]      # closes at steps 0…h-1
    n_actual  = W - len(pred_path)
    if n_actual > 0:
        hist_slice = actual_history[-n_actual:]
        window = np.concatenate([hist_slice, pred_path])
    else:
        window = pred_path[-W:]

    if len(window) < W:
        pad    = np.full(W - len(window), window[0])
        window = np.concatenate([pad, window])

    return float(np.dot(weights, window))


def _norm_ohlc_component(
    raw_x: float,
    raw_close: float,
    context_close: float,
    norm_method: str,
) -> float:
    """Normalise a predicted open/high/low value.

    log_returns:  log(X / context_close)
    fracdiff:     log(X / pred_close)  — candle shape, scale-free
    raw:          raw price
    """
    if norm_method == "raw":
        return float(raw_x)
    if norm_method == "log_returns":
        return math.log(raw_x / context_close + 1e-12)
    # fracdiff: express relative to the predicted close at the same horizon
    return math.log(raw_x / raw_close + 1e-12)


def _spread(raw_high: float, raw_low: float) -> float:
    """Intra-candle range: log(high/low). Always ≥ 0, scale-free."""
    return math.log(max(raw_high, raw_low + 1e-10) / (raw_low + 1e-10))


# ─────────────────────────────────────────────────────────────────────────────
# Plot helper
# ─────────────────────────────────────────────────────────────────────────────

def _save_run_plot(
    df: pd.DataFrame,
    result: dict,
    context_end_idx: int,
    plot_dir: Path,
    run_id: int,
) -> None:
    try:
        from Kronos.kronos_plots import plot_forecast, save_png
    except ImportError:
        return
    if len(df.index) < 2:
        return

    context_end_ts = df.index[context_end_idx]
    fname = plot_dir / f"run{run_id:05d}_{context_end_ts.strftime('%Y%m%d_%H%M')}.png"
    fig   = plot_forecast(result, context_bars_shown=50)
    save_png(fig, str(fname))


# ─────────────────────────────────────────────────────────────────────────────
# Filename builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_fname(
    pair: str, timeframe: str, context_length: int, calc_interval: int,
    horizons: list[int], norm_method: str, fracdiff_d: float,
    weekends: str, scaling: str, scaling_window: int,
    n_samples: int, years: list[int] | None,
    start: str | None, end: str | None,
) -> str:
    if norm_method == "log_returns":
        norm_tag = "logret"
    elif norm_method == "fracdiff":
        norm_tag = f"fdiff{fracdiff_d}"
    else:
        norm_tag = "raw"

    h_tag    = "h" + "-".join(str(h) for h in horizons)
    wknd_tag = "w" + weekends
    if scaling == "none":
        scale_tag = "snone"
    elif scaling == "rolling":
        scale_tag = f"sroll{scaling_window}"
    else:
        scale_tag = "sglob"

    samp_tag = f"_s{n_samples}" if n_samples > 1 else ""

    if years:
        year_tag = "_".join(str(y) for y in sorted(years))
    elif start or end:
        year_tag = f"{(start or '').replace('-', '')}-{(end or '').replace('-', '')}"
    else:
        year_tag = "all"

    return (
        f"{pair}_{timeframe}_kron_ctx{context_length}_int{calc_interval}"
        f"_{h_tag}_{norm_tag}_{wknd_tag}_{scale_tag}{samp_tag}_{year_tag}.parquet"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────────────────────────

def generate(
    pair:           str = "EURUSD",
    years:          list[int] | None = None,
    start:          str | None = None,
    end:            str | None = None,
    timeframe:      str = "H1",
    context_length: int = 512,
    horizons:       list[int] | None = None,
    calc_interval:  int = 1,
    norm_method:    str = "log_returns",
    fracdiff_d:     float = 0.4,
    threshold:      float = 6e-4,
    weekends:       str = "filled",
    scaling:        str = "none",
    scaling_window: int = 200,
    n_samples:      int = 1,
    percentiles:    list[float] | None = None,
    temperature:    float = 1.0,
    top_p:          float = 0.9,
    histdata_dir:   str | Path | None = None,
    save_dir:       str | Path | None = None,
    device:         str = "auto",
    save_plots:     bool = False,
) -> pd.DataFrame:
    """Generate Kronos OHLCV features and save to parquet.

    Parameters
    ----------
    pair            : currency pair, e.g. "EURUSD"
    years           : list of years to load; None = all CSVs in histdata_dir
    start / end     : ISO date strings for sub-range slicing after year filter
    timeframe       : "M1" | "M5" | "M15" | "H1" | "H4" | "D1"
    context_length  : bars of history fed to Kronos per run (≤ 512)
    horizons        : prediction horizons to record, e.g. [5, 10, 15, 20]
    calc_interval   : re-run Kronos every N bars; copy the same predictions to
                      the next N-1 bars (staleness 1…N-1). Default 1 = run every bar.
    norm_method     : "log_returns" | "fracdiff" | "raw"
    fracdiff_d      : fractional diff order (fracdiff mode only)
    threshold       : fracdiff weight truncation threshold
    weekends        : "filled" | "nogap" | "gaps"
    scaling         : "none" | "rolling" | "global"
    scaling_window  : look-back bars for rolling z-score (scaling="rolling")
    n_samples       : number of stochastic Kronos paths per run.
                      n_samples=1  → point-forecast mode (OHLC feature columns).
                      n_samples>1  → probabilistic mode (percentile columns only; no OHLC).
                      Paths are batched automatically (VRAM auto-detected on load).
    percentiles     : quantile levels for probabilistic mode, e.g. [0.05, 0.25, 0.50, 0.75, 0.95].
                      Ignored when n_samples=1. Defaults to [0.05, 0.25, 0.50, 0.75, 0.95].
    temperature     : Kronos sampling temperature (> 0 = stochastic, 0 → greedy)
    top_p           : nucleus sampling threshold
    histdata_dir    : path to HistData M1 CSV folder
    save_dir        : output root for parquet (and plots/ subfolder)
    device          : "auto" | "cuda" | "cpu"
    save_plots      : save a Kronos PNG per run into save_dir/plots/

    Returns
    -------
    pd.DataFrame with DatetimeIndex aligned to the resampled timeframe.
    """
    if horizons is None:
        horizons = [5, 10, 15, 20]
    if percentiles is None:
        percentiles = [0.05, 0.25, 0.50, 0.75, 0.95]

    horizons    = sorted(set(horizons))
    max_horizon = max(horizons)
    prob_mode   = n_samples > 1

    assert max_horizon <= 512,  "Kronos max context and pred_len is 512"
    assert calc_interval >= 1,  "calc_interval must be ≥ 1"
    assert all(h >= 1 for h in horizons), "All horizons must be ≥ 1"
    if context_length > 512:
        print(f"[kronos_features] clamping context_length {context_length} → 512")
        context_length = 512

    histdata_dir = Path(histdata_dir) if histdata_dir else PROJ / "histdata"
    save_dir     = Path(save_dir)     if save_dir     else PROJ / "featdata"
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Load & resample ───────────────────────────────────────────────────────
    loader = ForexDataLoader()
    df_m1  = loader.load_and_merge(histdata_dir, pair=pair, years=years, weekends=weekends)
    if start:
        df_m1 = df_m1[start:]
    if end:
        df_m1 = df_m1[:end]

    df = df_m1 if timeframe == "M1" else resample_ohlcv(df_m1, TIMEFRAMES[timeframe])
    df.attrs["pair"]      = pair
    df.attrs["timeframe"] = timeframe

    n       = len(df)
    closes  = df["close"].values.astype(np.float64)
    timestamps = df.index

    print(f"[kronos_features] {pair} {timeframe}  n={n:,}  "
          f"context={context_length}  horizons={horizons}  "
          f"calc_interval={calc_interval}  n_samples={n_samples}")

    # ── Load Kronos ───────────────────────────────────────────────────────────
    forecaster = KronosForecaster(max_context=context_length, device=device)
    forecaster.load()

    # ── Plots directory ───────────────────────────────────────────────────────
    plot_dir: Path | None = None
    if save_plots:
        plot_dir = save_dir / "plots" / f"{pair}_{timeframe}_kronos"
        plot_dir.mkdir(parents=True, exist_ok=True)

    # ── Main loop ─────────────────────────────────────────────────────────────
    rows: list[dict] = []
    run_positions = range(context_length, n, calc_interval)
    total_runs    = len(run_positions)

    for run_id, i in enumerate(run_positions):
        steps = min(calc_interval, n - i)
        if steps <= 0:
            break

        if run_id % max(1, total_runs // 20) == 0:
            pct = 100 * run_id / total_runs
            print(f"  run {run_id}/{total_runs} ({pct:.0f}%)  bar {i}/{n}", flush=True)

        ctx_start    = max(0, i - context_length)
        context_ohlc = df[OHLC_COLS].iloc[ctx_start: i].copy()
        actual_hist  = closes[ctx_start: i]
        context_close = float(closes[i - 1])

        x_timestamp  = pd.Series(timestamps[ctx_start: i])

        # y_timestamp: use actual future index where possible, pad near end
        avail_end = min(i + max_horizon, n)
        y_idx_avail = timestamps[i: avail_end]
        if len(y_idx_avail) < max_horizon:
            from Kronos.kronos_inference import _make_future_timestamps
            from datetime import timedelta
            if len(timestamps) >= 2:
                bar_td = timestamps[-1] - timestamps[-2]
            else:
                bar_td = timedelta(hours=1)
            last_ts = timestamps[avail_end - 1] if avail_end > i else timestamps[i - 1]
            extra   = _make_future_timestamps(last_ts, bar_td, max_horizon - len(y_idx_avail))
            y_idx   = y_idx_avail.append(pd.DatetimeIndex(extra))
        else:
            y_idx = y_idx_avail[:max_horizon]

        y_timestamp = pd.Series(pd.DatetimeIndex(y_idx))

        # ── Kronos inference ──────────────────────────────────────────────────
        result = forecaster.forecast(
            ohlcv_df=context_ohlc,
            pred_len=max_horizon,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            temperature=temperature,
            top_p=top_p,
            n_samples=n_samples,
            percentiles=percentiles if prob_mode else None,
            verbose=False,
        )
        pred_df     = result["mean_forecast"]      # (max_horizon, 6)
        close_samps = result["close_samples"]      # (n_samples, max_horizon)

        if save_plots and plot_dir is not None:
            result.update({
                "context_df":         context_ohlc,
                "ground_truth_df":    df.iloc[i: i + max_horizon] if i + max_horizon <= n else None,
                "context_end":        timestamps[i - 1],
                "forecast_start":     y_idx[0],
                "prediction_length":  max_horizon,
                "pair":               pair,
                "timeframe":          timeframe,
            })
            _save_run_plot(df, result, i - 1, plot_dir, run_id)

        # ── Emit calc_interval rows ───────────────────────────────────────────
        # Build a 1D close path for fracdiff (cumulative predicted closes)
        pred_close_path = pred_df["close"].values.astype(np.float64)

        for k in range(steps):
            bar_ts = timestamps[i + k]
            row: dict = {"run_id": run_id, "staleness": k}

            if prob_mode:
                # probabilistic mode: percentile columns only
                for h in horizons:
                    col_samples = close_samps[:, h - 1]  # (n_samples,)
                    for p in percentiles:
                        pp  = f"q{int(round(p * 100)):02d}"
                        row[f"{pp}_h{h}"] = float(np.percentile(col_samples, p * 100))
            else:
                # point-forecast mode: OHLC columns
                for h in horizons:
                    idx   = h - 1
                    raw_c = float(pred_df["close"].iloc[idx])
                    raw_h = float(pred_df["high"].iloc[idx])
                    raw_l = float(pred_df["low"].iloc[idx])

                    # guard against degenerate candles (tokeniser rounding)
                    raw_h = max(raw_h, raw_c, raw_l)
                    raw_l = min(raw_l, raw_c, raw_h)

                    norm_c = _norm_close(
                        raw_c, context_close, actual_hist, norm_method,
                        fracdiff_d, threshold, idx + 1, pred_close_path,
                    )
                    norm_h = _norm_ohlc_component(raw_h, raw_c, context_close, norm_method)
                    norm_l = _norm_ohlc_component(raw_l, raw_c, context_close, norm_method)

                    row[f"close_h{h}"]  = norm_c
                    row[f"high_h{h}"]   = norm_h
                    row[f"low_h{h}"]    = norm_l
                    row[f"spread_h{h}"] = _spread(raw_h, raw_l)

            rows.append((bar_ts, row))

    # ── Build DataFrame ───────────────────────────────────────────────────────
    index = pd.DatetimeIndex([r[0] for r in rows], name="datetime")
    data  = [r[1] for r in rows]
    out   = pd.DataFrame(data, index=index)

    # Deterministic column order
    feat_cols: list[str] = []
    if prob_mode:
        for h in horizons:
            for p in percentiles:
                pp = f"q{int(round(p * 100)):02d}"
                feat_cols.append(f"{pp}_h{h}")
    else:
        for h in horizons:
            feat_cols += [f"close_h{h}", f"high_h{h}", f"low_h{h}", f"spread_h{h}"]
    for col in feat_cols:
        if col not in out.columns:
            out[col] = np.nan
    meta_cols = ["run_id", "staleness"]
    out = out[feat_cols + meta_cols]

    # ── Optional scaling ──────────────────────────────────────────────────────
    if scaling != "none":
        scale_cols = feat_cols  # exclude run_id and staleness
        if scaling == "global":
            mu  = out[scale_cols].mean()
            std = out[scale_cols].std().replace(0, 1)
            out[scale_cols] = (out[scale_cols] - mu) / std
        elif scaling == "rolling":
            for col in scale_cols:
                roll_mu  = out[col].rolling(scaling_window, min_periods=1).mean()
                roll_std = out[col].rolling(scaling_window, min_periods=1).std().fillna(1).replace(0, 1)
                out[col] = (out[col] - roll_mu) / roll_std

    # ── Save parquet ──────────────────────────────────────────────────────────
    fname  = _build_fname(
        pair, timeframe, context_length, calc_interval, horizons,
        norm_method, fracdiff_d, weekends, scaling, scaling_window,
        n_samples, years, start, end,
    )
    fpath  = save_dir / fname
    out.to_parquet(fpath)
    print(f"[kronos_features] Saved {len(out):,} rows → {fpath}")

    return out


# ── Standalone entry-point ────────────────────────────────────────────────────

if __name__ == "__main__":
    df = generate(
        pair="EURUSD",
        years=[2023],
        timeframe="H1",
        context_length=512,
        horizons=[5, 10, 15, 20],
        #percentiles    = [0.1, 0.3, 0.5, 0.7, 0.9],#for probabilistic mode; ignored when n_samples=1
        n_samples      = 1,  # 1 = point-forecast mode (OHLC features); >1 → probabilistic mode (percentiles only)
        calc_interval=20,
        norm_method    = "fracdiff",
        fracdiff_d     = 0.3,
        threshold      = 6e-4,
        weekends="filled",
    )
    print(df.head())
    print("columns:", df.columns.tolist())
