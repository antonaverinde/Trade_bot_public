# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Trading bot: XGBoost + Chronos-2 zero-shot forecasting on HistData M1 forex CSVs.

## Environment

Virtual environment is at `/home/anton/Trade_bot/.venv`.

**Important:** Shell state does not persist between Bash tool invocations in Claude Code. Each Bash call starts in a fresh shell. To run commands in the venv, **prefix them with the activation**:

```bash
source /home/anton/Trade_bot/.venv/bin/activate && <your-command>
```

For example:
```bash
source /home/anton/Trade_bot/.venv/bin/activate && python -m pip list
source /home/anton/Trade_bot/.venv/bin/activate && jupyter notebook
```

Always use the full absolute path (not `~`).

## Architecture

### Core modules

- **`Pipeline/pipeline.py`** — end-to-end pipeline: load → normalize → features → split → scale
- **`main.py`** — runs XGBoost on all timeframes + Chronos zero-shot forecast
- **`histdata/`** — HistData M1 CSVs per pair per year (`DAT_ASCII_EURUSD_M1_2020.csv`)
- **`Chronos/chronos_inference.py`** — `ChronosForecaster`: model loading, forecasting, quantile/CDF helpers
- **`Chronos/chronos_plots.py`** — Matplotlib fan chart (dark TradingView theme), PNG output
- **`Chronos/chronos_features.py`** — rolling Chronos predictions as XGBoost feature columns → `featdata/*.parquet`
- **`Chronos/chronos_feature_plot.py`** — Plotly visualization: Chronos quantiles vs. pipeline-normalized close
- **`Kronos/kronos_inference.py`** — `KronosForecaster`: model loading, OHLCV forecasting
- **`Kronos/kronos_plots.py`** — Matplotlib candlestick forecast chart (dark TradingView theme), PNG output
- **`Kronos/kronos_features.py`** — rolling Kronos OHLCV predictions as XGBoost feature columns → `featdata/*.parquet`
- **`Kronos/kronos_feature_plot.py`** — Plotly visualization: Kronos predicted OHLC vs. pipeline-normalized close
- **`Supplementary/fracdiff_adf.py`** — sweeps fracdiff `d` over [0,1], ADF test to find optimal d
- **`xgboost_experiment.ipynb`** — 3-class XGBoost with MLflow logging
- **`xgboost_chronos_experiment.ipynb`** — XGBoost + Chronos features
- **`xgboost_experimen_bin.ipynb`** — Binary classification (long/short, neutral removed) with MLflow logging
- **`xgboost_experiment_bin_conf.ipynb`** — Binary classification with confidence/uncertainty analysis
- **`xgboost_nns_experiment.ipynb`** — XGBoost with neural network predictor features

**Model weights:** `Chronos/model/` is gitignored (~477 MB). `ChronosForecaster.load()` auto-downloads from HuggingFace on first use.

### Pipeline usage

```python
from Pipeline.pipeline import ForexDataLoader, ForexPipeline

loader   = ForexDataLoader()
pipeline = ForexPipeline(norm_method="log_returns", target_type="lag", fracdiff_d=0.3, threshold=3e-4)
df_m1    = loader.load_and_merge("histdata/", pair="EURUSD", years=[2023])
df_m1    = df_m1["2023-11-01":"2023-12-25"]   # optional date slice

results  = pipeline.run(df_m1, timeframe="M15")
X_train, y_train = pipeline.get_xy(results["train"], "direction_1", results["feature_cols"])
```

**Key constructor args:**
- `norm_method`: `"log_returns"` | `"fracdiff"` | `"raw"`
- `target_type`: `"lag"` → cols `direction_1/5/15` | `"triple_barrier"` → col `tb_label`
- `barrier_on_raw`: `True` (default) — use raw price levels for triple-barrier arithmetic; `False` uses normalized data (broken, for comparison only)
- `fracdiff_d`: differentiation order (default 0.4; optimal H1 ≈ 0.3)
- `threshold`: weight truncation (default 6e-4; lower = longer warm-up, more history)
- `scaling`: `"none"` (recommended for XGBoost) | `"rolling"` | `"global"`
- `weekends`: `"nogap"` (default) | `"gaps"` | `"filled"`
- `lags`: default `[1, 2, 5, 10]` — creates `close_lagN` features

**Features (32 total):** RSI 14+21 (speed, accel, crosses) · ADX (di_diff, adx_delta) · dist_ema200 · atr_rel · bb_pct_b · time sin/cos · body/shadow/gap · skew/kurt/vol_ewma · close lags

**Scaling notes:**
- XGBoost is scale-invariant — use `scaling="none"` to avoid production distribution shift
- Rolling scaling (window=200 bars ≈ 8 H1 trading days) is for neural nets / linear models only
- fracdiff and scaling are orthogonal: fracdiff stationarizes, scaling normalizes variance

**Fracdiff warm-up by d (threshold=6e-4):** d=0.3 → ~98 bars warm-up, 90.5% weight captured. Use `threshold=3e-4` for slightly longer warm-up; `threshold=1e-5` for strict (d=0.3 → ~700 bars).

**Triple-barrier target:** use `target_col="tb_label"` (not `"direction_1"`) in `get_xy()`.

**Triple-barrier labeling mechanics (with raw-price fix):** For each bar `t`, three barriers are set using raw price levels and volatility computed from raw returns:

**Key fix (barrier_on_raw=True, default):**
- Before feature engineering, inject raw price columns: `_raw_close`, `_raw_high`, `_raw_low` (from original df), and `_raw_sigma = pct_change().ewm(span=100).std()` computed on raw closes
- Triple barrier arithmetic uses these raw values, NOT normalized prices (log-returns/fracdiff are ~0.0001, making barrier math meaningless)
- `_raw_*` columns are excluded from `feature_cols` (no data leakage); they are dropped after labeling
- This fix produces correct label distributions (e.g., `{-1: 731, 0: 119, 1: 508}`) instead of all-neutral

**Barrier calculation** (with raw values):

- **Upper** = `close_raw[t] × (1 + sigma_raw[t] × k_up)` — take-profit barrier
- **Lower** = `close_raw[t] × (1 - sigma_raw[t] × k_down)` — stop-loss barrier  
- **Time** = bar `t + horizon_bars` — timeout barrier

The next `horizon_bars` closes (or high/low depending on `barrier_price`) are scanned to see which barrier is hit first:

| Result | `tb_label` | `tb_ret` |
|---|---|---|
| Upper hit first (or tie) | `+1` (long) | log return from `close[t]` to exit bar |
| Lower hit first | `-1` (short) | log return from `close[t]` to exit bar |
| Neither hit within time | `0` (neutral/hold) | NaN |

**`tb_ret` use cases:**
- **Regression targets** — train a model to predict expected return magnitude (continuous), not just direction (discrete 3-class)
- **Position sizing** — weight trades by expected payoff magnitude
- **Backtest P&L** — compute realized returns without re-running the full simulation
- Key distinction from `tb_label`: the label tells you *which barrier was hit first*, `tb_ret` tells you *the actual return at that exit bar*
  - With `barrier_price="hl"` the scan uses high/low but `tb_ret` is always computed from close-to-close

**Key insight:** Barrier width scales with **current volatility** — wide when volatile, tight when calm. This makes the same `k_up`/`k_down` multipliers produce consistent risk/reward across volatility regimes. Default parameters: `k_up=2.0`, `k_down=1.0`, `horizon_bars=10`.

**Weekend modes:**
- `"nogap"` — Mon–Fri data only, Plotly `rangebreaks` hide visual gap
- `"gaps"` — 7-day grid with NaN weekends
- `"filled"` — forward-fill Friday close through weekend (required for Chronos continuous context)

### Chronos usage (standalone)

```python
from Pipeline.pipeline import ForexDataLoader
from Chronos.chronos_inference import ChronosForecaster
from Chronos.chronos_plots import plot_forecast, save_png
import torch

loader = ForexDataLoader()
df_m1  = loader.load_and_merge("histdata/", pair="EURUSD", years=[2023])
df_h1  = df_m1.resample('1h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()

forecaster = ChronosForecaster(context_length=512, device='auto', dtype=torch.bfloat16)
forecaster.load()
result = forecaster.forecast_from_df(df_h1, context_end='2023-09-01 00:00', prediction_length=48)

fig = plot_forecast(result, context_bars_shown=50)
save_png(fig, "chronos_forecast.png")

prob_up = ChronosForecaster.prob_above(result["forecast"], threshold=result["context_df"]["close"].iloc[-1])
```

**Quality guideline:** `prediction_length / context_length ≤ 0.25`. Hard max prediction = 1024 bars.

| context_length | Reliable prediction_length |
|---|---|
| 512 | ≤ 128 |
| 1024 | ≤ 256 |
| 2048 | ≤ 512 |
| 8192 | ≤ 1024 |

Chronos-2 takes **raw close prices** (not log returns). Output quantiles keyed by float: `qs[0.01]`, `qs[0.50]`, etc.

### Chronos features (XGBoost integration)

```python
from Chronos.chronos_features import generate

chronos_df = generate(
    pair="EURUSD", years=[2023], timeframe="H1",
    horizons=[5, 10, 15, 20],                          # prediction horizons to record
    percentiles=[0.05, 0.25, 0.50, 0.75, 0.95],       # quantile levels
    calc_interval=10,                                   # rerun every 10 bars; copy for 9
)
# → featdata/EURUSD_H1_ctx504_int10_h5-10-15-20_logret_wfilled_snone_2023.parquet

# Join with pipeline features:
feat_df  = pd.concat([results["train_raw"], results["val_raw"], results["test_raw"]])
chronos  = pd.read_parquet("featdata/...parquet")
q_cols   = [c for c in chronos.columns if c.startswith("q") and "_h" in c]
combined = feat_df.join(chronos[q_cols], how="left")
# First context_length bars → NaN in Chronos cols
```

**Key `generate()` params:**
- `horizons`: list of prediction horizons to record (default `[5, 10, 15, 20]`); internally sets `prediction_length = max(horizons)`
- `percentiles`: quantile levels (default `[0.05, 0.25, 0.50, 0.75, 0.95]`)
- `context_length` (default 504), `calc_interval` (default 1 = run every bar; set higher to copy and reduce computation)
- `norm_method`: `"log_returns"` | `"fracdiff"` | `"raw"` (default `"log_returns"`)
- `weekends`: `"filled"` (default) | `"nogap"` | `"gaps"`
- `scaling`: `"none"` (default) | `"rolling"` | `"global"`
- `scaling_window`: rolling z-score lookback (default 200; ignored if `scaling` ≠ `"rolling"`)

**Output columns:** `q{pp}_h{H}` for each percentile × horizon (e.g. `q05_h5`, `q50_h10`, `q95_h15`) · `run_id` · `staleness` (0 = fresh run, 1…calc_interval-1 = copied)

**Column naming:** `q{int(p*100):02d}_h{horizon}` — e.g. percentile 0.05 at horizon 10 → `q05_h10`

**Normalization per bar:** each predicted price is transformed to match `norm_method`:
- `"log_returns"` → `log(pred / last_actual_close)`
- `"fracdiff"` → applies `_fracdiff_weights` on `[...actual_history, pred_0...pred_h]`
- `"raw"` → no transform

**Filename tags:** `_ctx{N}_int{I}_h{h1}-{h2}-..._w<weekends>_s<scale>_<year>` (e.g. `_ctx504_int10_h5-10-15-20_wfilled_snone_2023`)

**Feature data uses Parquet** (not CSV): preserves dtypes, DatetimeIndex, 5–10× smaller, selective column reads.

### Chronos feature visualization

```bash
python Chronos/chronos_feature_plot.py featdata/EURUSD_H1_ctx504_int10_h5-10-15-20_fdiff0.3_wfilled_snone_2023.parquet
```

Or from Python:
```python
from Chronos.chronos_feature_plot import plot_file
fig = plot_file("featdata/...parquet", plot_horizon=10,
                plot_start="2023-06-01", plot_end="2023-09-30",
                threshold=6e-4)  # Match generation params if non-default
fig.show()
```

`plot_horizon` selects which horizon to display (defaults to smallest available). Quantile columns for that horizon (e.g. `q05_h10`…`q95_h10`) are auto-discovered from the parquet schema.

**Critical: `threshold` parameter.** If you generated the features with a non-default `threshold` (e.g., `chronos_features.generate(..., threshold=3e-4)`), you **must** pass the same value to `plot_file(threshold=...)`. The parquet filename does NOT encode the threshold; mismatched thresholds produce incorrect fracdiff warm-up lengths and shift the normalized values. Pass `threshold=None` to skip normalization entirely and compare raw prices.

**Fracdiff warm-up and `plot_start`.** The pipeline loads **all available years** of M1 data (or the specified `years`), runs fracdiff on the full series (no pre-slicing), and the join with `chronos_df` automatically restricts the output to dates in the parquet. This ensures fracdiff has enough historical context to warm up before your `plot_start` date. If `plot_start="2023-06-01"` but the parquet contains 2023 Q1–Q4 data, the pipeline still loads and warms up on 2023 Q1 bars, then only displays from 2023-06-01 onward. **Do not manually slice `df_m1` to `plot_start` before calling `plot_file()` — the function handles date alignment internally.**

3-panel layout: (1) normalized close + quantile lines for selected horizon + rerun boundaries, (2) `q50_h{H} − actual` error fill, (3) staleness (0 = fresh run, counts up to calc_interval-1).

**Leading NaN backfill:** after joining, `df["actual"] = df["actual"].bfill()` fills fracdiff warm-up gap for display only.

**Plotly shapes performance:** batch rerun boundary lines via `fig.update_layout(shapes=[...])` — do NOT loop `add_vline()` (O(n²)).

### Kronos usage (standalone)

Kronos is a generative OHLCV foundation model (decoder-only Transformer, 102.3M params).
Unlike Chronos, it forecasts full candlestick data (open/high/low/close) rather than close-price quantiles.

**Setup (one-time):**
```bash
cd /home/anton/Trade_bot/Kronos
git clone https://github.com/shiyu-coder/Kronos.git kronos_repo
# Dependencies already in venv: einops, safetensors, tqdm
```

Model weights (`Kronos/model/`) and `Kronos/kronos_repo/` are gitignored.

```python
from Pipeline.pipeline import ForexDataLoader
from Kronos.kronos_inference import KronosForecaster

loader = ForexDataLoader()
df_m1  = loader.load_and_merge("histdata/", pair="EURUSD", years=[2020])
df_h1  = df_m1.resample('1h').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()

fc = KronosForecaster(max_context=512)   # auto-downloads weights on first load()
fc.load()
result = fc.forecast_from_df(df_h1, context_end='2020-06-01 16:00', pred_len=20)

from Kronos.kronos_plots import plot_forecast, save_png
fig = plot_forecast(result, context_bars_shown=50)
save_png(fig, "kronos_forecast.png")
```

**`result` dict keys:**
- `mean_forecast` — pd.DataFrame (pred_len, 6) with open/high/low/close/volume/amount
- `context_df` — last max_context bars fed to model
- `ground_truth_df` — actual OHLCV after context_end (if available in df)
- `close_samples` — np.ndarray (n_samples, pred_len) of close prices across stochastic runs
- `samples` — list of individual prediction DataFrames

### Kronos features (XGBoost integration)

```python
from Kronos.kronos_features import generate

# Point-forecast mode (n_samples=1, default) → OHLC columns
kronos_df = generate(
    pair="EURUSD", years=[2020], timeframe="H1",
    horizons=[5, 10, 15, 20],
    calc_interval=10,
    norm_method="log_returns",
    weekends="filled",
)
# → featdata/EURUSD_H1_kron_ctx512_int10_h5-10-15-20_logret_wfilled_snone_2020.parquet

# Probabilistic mode (n_samples>1) → percentile columns only (Chronos-compatible naming)
kronos_prob = generate(
    pair="EURUSD", years=[2020], timeframe="H1",
    horizons=[5, 10, 15, 20],
    calc_interval=10,
    n_samples=10,
    percentiles=[0.05, 0.25, 0.50, 0.75, 0.95],
    temperature=1.0,
)
# → featdata/EURUSD_H1_kron_ctx512_int10_h5-10-15-20_logret_wfilled_snone_s10_2020.parquet
# Columns: q05_h5, q25_h5, q50_h5, q75_h5, q95_h5, ... q95_h20, run_id, staleness
```

**Output columns depend on n_samples:**

`n_samples=1` — point-forecast mode (OHLC columns):
- `close_h{H}` — normalized predicted close (analogous to Chronos q50)
- `high_h{H}` — normalized predicted high
- `low_h{H}` — normalized predicted low
- `spread_h{H}` — log(pred_high / pred_low), scale-free volatility proxy
- `run_id`, `staleness`

`n_samples>1` — probabilistic mode (percentile columns **only**, no OHLC):
- `q{pp}_h{H}` — percentile pp of close across n_samples stochastic paths
  e.g. `q05_h10`, `q50_h10`, `q95_h10` (naming identical to Chronos — drop-in compatible)
- `run_id`, `staleness`
- Use `percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]` to control which quantiles are written

**Key differences from Chronos:**
- **Input requirements:** OHLC are **required** (raises ValueError if missing); `volume` and `amount` are **optional** — Kronos auto-fills both with 0.0 internally if absent. For forex (e.g., EURUSD with meaningless reported volume), pass only OHLC; letting Kronos zero-fill is correct.
- **Context limit is 512** (vs 8192 for Chronos) — use `context_length=512`
- **Point forecast** (OHLCV) not quantile distribution — richer geometry per bar
- **n_samples > 1 is slow** — N full autoregressive passes; use `calc_interval >= 10`
- **Normalization:** log_returns applies to all 4 columns vs context_close; fracdiff applies to close only, high/low expressed as log(X/pred_close) (candle shape)

**Parquet filename format:** `{pair}_{tf}_kron_ctx{C}_int{I}_h{h1}-{h2}-..._{norm}_{wknd}_{scale}[_s{N}]_{year}.parquet`
The `_kron_` infix distinguishes from Chronos parquets.

**Multiple stochastic samples (n_samples > 1):** The `KronosForecaster.forecast()` method supports parallel sampling of N stochastic paths in a **single GPU forward pass**. All N samples are processed together with expanded batch dims, avoiding N sequential inference calls.

**Overhead vs. n_samples:**
| n_samples | Time overhead | VRAM cost | Use case |
|-----------|---------------|-----------|----------|
| 1 | 1× (baseline ~0.26s for pred_len=10) | baseline | Fast point forecast, feature generation |
| 5 | ~4× | moderate | Good CI estimate, feature generation with percentiles (optimal for calc_interval ≥ 10) |
| 10 | ~7.5× | higher | Tight confidence bands, standalone plots/fan charts |
| 20+ | avoid | OOM risk | Diminishing returns |

The parallel approach is ~25% faster than N sequential `predict()` calls but still scales roughly linearly (each autoregressive step processes N sequences in parallel).

**Result dict keys with n_samples > 1:**
- `all_samples_raw` — np.ndarray (n_samples, pred_len, 6) raw OHLCVA across all paths
- `close_samples` — np.ndarray (n_samples, pred_len) close prices; use `np.percentile(..., axis=0)` to compute quantiles per bar
- `samples` — list[pd.DataFrame], one per sample, each with full OHLCV
- `mean_forecast` — column-wise mean across all samples

**Percentile extraction:**
```python
result = fc.forecast_from_df(df_h1, context_end='2020-06-01', pred_len=20, n_samples=10)
close_paths = result["close_samples"]  # (10, 20)
p05 = np.percentile(close_paths, 5, axis=0)   # pessimistic close path
p50 = np.percentile(close_paths, 50, axis=0)  # median
p95 = np.percentile(close_paths, 95, axis=0)  # optimistic close path
```

**Critical:** `temperature=1.0` (default) is required for meaningful variance between samples. At `temperature=0` all paths are identical (greedy decoding) and percentiles are useless.

**Recommendation:**
- `n_samples=1` (default): fast OHLC point forecast. Use when geometric features (high/low/spread) are the signal.
- `n_samples=5, calc_interval=10`: probabilistic mode at ~4× overhead amortized over 10 bars. Produces `q05/q25/q50/q75/q95` columns identical in naming to Chronos — plug directly into Chronos-trained XGBoost code.
- `n_samples=10`: standalone plots with proper fan chart.
- Both modes coexist in `featdata/` — use filename `_s{N}` tag to distinguish.

**Automatic GPU memory management (RTX 4060 8GB):**

When `n_samples` exceeds available parallel capacity, `KronosForecaster.forecast()` automatically chunks into sequential batches via `_generate_samples()`:

**Benchmarking on load():**
```
Total VRAM: 8585 MB (RTX 4060)
After model load: 430 MB
Per-sample overhead: ~28 MB per sample
Fixed inference overhead: 41 MB
Max parallel recommendation: min(250, 64) = 64 samples per batch
```

The conservative cap is `_VRAM_PRACTICAL_CAP = 64` — beyond this, wall-time overhead (chunking overhead, synchronization) increases faster than statistical benefit from additional samples.

**How automatic chunking works:**
```python
fc = KronosForecaster()  # max_parallel=0 → auto-detect → typically 64 on RTX 4060
fc.load()
result = fc.forecast_from_df(df_h1, pred_len=20, n_samples=100)
# Internally: ceil(100/64) = 2 GPU forward passes
#   Batch 1: 64 samples
#   Batch 2: 36 samples
# Progress: "[samples] batch 1/2  (64 paths)" then "[samples] batch 2/2  (36 paths)"
```

**Override auto-detection:**
```python
fc = KronosForecaster(max_parallel=128)  # skip benchmark, use 128 per batch
fc.load()
result = fc.forecast_from_df(..., n_samples=100)  # single batch if ≤128
```

Pass `max_parallel=0` (default) for auto-detect, or any concrete value (e.g., 64, 128) to override. The chunking is transparent to the caller — `result["close_samples"]` and percentiles are computed across **all** samples regardless of batch count.

**No OOM with automatic chunking:** `n_samples=100` safely runs on RTX 4060 8GB with 2 sequential batches and no GPU memory overflow.

**Join with pipeline features (same as Chronos):**
```python
kron = pd.read_parquet("featdata/EURUSD_H1_kron_ctx512_int10_h5-10-15-20_logret_wfilled_snone_2020.parquet")
kron_cols = [c for c in kron.columns if c not in ("run_id", "staleness")]
combined = feat_df.join(kron[kron_cols], how="left")
# First context_length bars → NaN in Kronos cols
```

**Visualization:**
```bash
python Kronos/kronos_feature_plot.py featdata/EURUSD_H1_kron_ctx512_int10_h5-10-15-20_logret_wfilled_snone_2020.parquet
```

Or from Python:
```python
from Kronos.kronos_feature_plot import plot_file
fig = plot_file("featdata/...parquet", plot_horizon=10, plot_start="2020-06-01")
fig.show()
```

**Auto-detection of mode:** `kronos_feature_plot.py` automatically detects whether a parquet contains point-forecast (OHLC) or probabilistic (percentile) columns:
- Checks for presence of OHLC columns (`close_h*`, `high_h*`, `low_h*`)
- Sets `prob_mode=False` if OHLC found, `prob_mode=True` if only percentile columns found
- Renders **point-forecast layout** (OHLC mode): high/low dashed + close gold + spread volatility panel
- Renders **probabilistic layout** (percentile mode): quantile fan with inner/outer bands + q50−actual error panel
- Parquet filename `_s{N}` tag is informational only — visualization auto-detects based on column presence

### TimeseriesFM usage (standalone)

TimeseriesFM is Google's zero-shot time series forecaster (2.5B-200m variant). Like Chronos, it predicts quantile distributions; unlike Chronos, it uses a state-space architecture and is optimized for shorter contexts.

**Setup (one-time):**
```bash
source /home/anton/Trade_bot/.venv/bin/activate && pip install timesfm[torch]==2.0.0
# Model (~1.2 GB) auto-downloads to Timeseriesfm/model/ on first load()
```

**Standalone forecast:**
```python
from Pipeline.pipeline import ForexDataLoader
from Timeseriesfm.timesfm_inference import TimesFMForecaster
import torch

loader = ForexDataLoader()
df_m1  = loader.load_and_merge("histdata/", pair="EURUSD", years=[2023])
df_h1  = df_m1.resample('1h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()

forecaster = TimesFMForecaster(context_length=512, device='auto', dtype=torch.bfloat16)
forecaster.load()
result = forecaster.forecast_from_df(df_h1, context_end='2023-09-01 00:00', prediction_length=48)

from Timeseriesfm.timesfm_plots import plot_forecast, save_png
fig = plot_forecast(result, context_bars_shown=50)
save_png(fig, "timesfm_forecast.png")

prob_up = TimesFMForecaster.prob_above(result["forecast"], threshold=result["context_df"]["close"].iloc[-1])
```

**Key parameters:**
- `context_length` (default 512) — TimesFM sweet spot for forex (Chronos handles up to 8192)
- `prediction_length` (max ~256 reliable; TimesFM degrades faster than Chronos for long horizons)
- Input: **raw close prices** (not log returns)
- Output quantiles: **10 levels** `[0.0, 0.1, 0.2, ..., 0.9]` (coarser than Chronos's 21; 0.0 = mean, not median)

**Quality guideline:** `prediction_length / context_length ≤ 0.5` (more permissive than Chronos). Hard max prediction ≈ 256 bars.

### TimeseriesFM features (XGBoost integration)

```python
from Timeseriesfm.timesfm_features import generate

timesfm_df = generate(
    pair="EURUSD", years=[2023], timeframe="H1",
    horizons=[5, 10, 15, 20],                          # prediction horizons to record
    percentiles=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # TimesFM's 10 quantiles (default)
    calc_interval=5,                                    # rerun every 5 bars; copy for 4
    context_length=512,
    prediction_length=20,  # must be ≥ max(horizons)
)
# → featdata/EURUSD_H1_tfm_ctx512_int5_h5-10-15-20_logret_wfilled_snone_2023.parquet
```

**Key `generate()` params:**
- `horizons`: list of prediction horizons (default `[5, 10, 15, 20]`)
- `percentiles`: quantile levels (default all 10: `[0.0, 0.1, ..., 0.9]`; can filter, e.g., `[0.1, 0.5, 0.9]`)
- `context_length`, `prediction_length` — TimesFM tuning; `prediction_length ≥ max(horizons)`
- `calc_interval` — rerun every N bars; copy for N-1 (lower than Chronos due to faster speed; default 5)
- `norm_method`, `weekends`, `scaling` — same as Chronos

**Output columns:** `tfm_q{pp}_h{H}` for each percentile × horizon (e.g. `tfm_q00_h5`, `tfm_q50_h10`, `tfm_q90_h15`) · `run_id` · `staleness`

**Column naming:** percentile as int (0=p0.0, 1=p0.1, ..., 9=p0.9) → `tfm_q{int(p*10):02d}_h{horizon}` (e.g. p=0.0 → `q00`, p=0.5 → `q50`, p=0.9 → `q90`)

**Parquet filename tag:** `_tfm_` (e.g. `EURUSD_H1_tfm_ctx512_int5_h5-10-15-20_logret_wfilled_snone_2023.parquet`) — distinct from Chronos's no-infix and Kronos's `_kron_`

**Join with pipeline features:**
```python
tfm = pd.read_parquet("featdata/EURUSD_H1_tfm_ctx512_int5_h5-10-15-20_logret_wfilled_snone_2023.parquet")
tfm_cols = [c for c in tfm.columns if c.startswith("tfm_q") and "_h" in c]
combined = feat_df.join(tfm[tfm_cols], how="left")
# First context_length bars → NaN in TimeseriesFM cols
```

### TimeseriesFM feature visualization

```bash
python Timeseriesfm/timesfm_feature_plot.py featdata/EURUSD_H1_tfm_ctx512_int5_h5-10-15-20_logret_wfilled_snone_2023.parquet
```

Or from Python:
```python
from Timeseriesfm.timesfm_feature_plot import plot_file
fig = plot_file("featdata/...parquet", plot_horizon=10,
                plot_start="2023-06-01", plot_end="2023-09-30",
                threshold=6e-4)  # Match generation params if non-default
fig.show()
```

Same 3-panel layout as Chronos: (1) normalized close + quantile lines + rerun boundaries, (2) `q50_h{H} − actual` error fill, (3) staleness.

### Fracdiff ADF sweep

```bash
python Supplementary/fracdiff_adf.py
```
Sweeps d over [0,1], finds minimum d for stationarity (ADF p < 0.1). Configure `PAIR`, `TIMEFRAME`, `YEARS`, `D_MIN/MAX/STEP`, `SIG_LEVEL` at top of script.

### Project root detection

Use `_find_proj()` in any module that runs as a script or in notebooks:

```python
from pathlib import Path

def _find_proj() -> Path:
    try:
        start = Path(__file__).resolve()
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
```

Then: `histdata_dir = PROJ / "histdata"`, `featdata_dir = PROJ / "featdata"`.

### XGBoost 3-class pattern

**Label encoding** (`tb_label` → 0/1/2):
```python
def extract_xy(results, split, target_col, feature_cols):
    X, y = pipeline.get_xy(results[split], target_col, feature_cols)
    if target_col == "tb_label":
        y = (y + 1).astype(int)  # -1→0, 0→1, 1→2
    return X, y.astype(int)
```

**XGBoost config:**
```python
XGB_PARAMS = {"num_class": 3, "objective": "multi:softprob", "eval_metric": "mlogloss", ...}
```

**Metrics — two suites:**
```python
proba      = model.predict_proba(X)          # shape (n,3) — DO NOT SLICE for multi-class AUC
proba_long = proba[:, 2]                     # P(long) for binary metrics
y_long     = (y == 2).astype(int)

m = {
    "auc":           roc_auc_score(y, proba, multi_class="ovr", average="macro"),
    "auc_long":      roc_auc_score(y_long, proba_long),
    "avg_precision": average_precision_score(y, proba, average="macro"),
    "logloss":       log_loss(y, proba),
    "brier_long":    brier_score_loss(y_long, proba_long),
    "f1":            f1_score(y, pred, average="macro", zero_division=0),
    "balanced_acc":  balanced_accuracy_score(y, pred),
    "mcc":           matthews_corrcoef(y, pred),
}
```

**Class labels:**
```python
CLASS_LABELS = {0: "short", 1: "neutral", 2: "long"}
```

**Classification reports with class names and distribution (notebooks):**
When printing `classification_report()`, always use `target_names` and precede with class distribution summary:
```python
_target_names = [CLASS_LABELS[i] for i in range(3)]   # ["short", "neutral", "long"]

for sname, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
    pred = model.predict(X)
    
    # Class distribution summary
    total = len(y)
    unique, counts = np.unique(y, return_counts=True)
    dist = {CLASS_LABELS[int(k)]: int(v) for k, v in zip(unique, counts)}
    print(f"\n── {sname}  (n={total:,}) ──")
    print("  Class counts: " + "  ".join(f"{lbl}: {cnt:,} ({cnt/total:.1%})"
                                        for lbl, cnt in dist.items()))
    
    print(classification_report(y, pred, target_names=_target_names))
```
This replaces the 0/1/2 row labels with semantic names and shows raw counts+percentages. Enables quick diagnosis: if recall is weak on a class, check if it has fewer samples (sampling issue vs. model weakness).

### XGBoost + Chronos notebook (`xgboost_chronos_experiment.ipynb`)

**Configuration:**
1. `DATA_CFG` is single source of truth — all shared params (pair, years, timeframe, norm_method, weekends, scaling, fracdiff_d, threshold)
2. `CHRONOS_CFG` holds only Chronos-specific params; structure:
   ```python
   CHRONOS_CFG = {
       "context_length": 504,          # match generate() call
       "calc_interval": 10,            # match generate() call
       "horizons": [5, 10, 15, 20],    # critical: filter quantile cols to these horizons
       "use_staleness": False,         # optionally include staleness as a feature
   }
   ```
3. Parquet filename is derived from both CFGs via `_build_chronos_fname()` — replicate the exact same logic as `chronos_features.generate()` internal naming

**Column discovery & joining (Cell 11+):**
- Quantile columns follow pattern `q{pp}_h{hh}` (e.g., `q05_h10`, `q50_h20`)
- Discover dynamically via regex and **filter to horizons in `CHRONOS_CFG["horizons"]`** (critical!)
- Sort by (horizon, percentile) for consistency
- If `use_staleness=True`, append `"staleness"` col
- Join onto pipeline splits, drop rows with NaN in any quantile col (fracdiff warm-up gap)
- Print coverage per split to verify Chronos data availability

**Features & training:**
- Feature importance: orange = Chronos cols, blue = pipeline cols
- MLflow experiment: `xgboost_chronos_forex` (separate from base `xgboost_forex`)
- Pass extended feature list to XGBoost as usual

### Per-class prediction quality analysis

Both `xgboost_experiment.ipynb` and `xgboost_chronos_experiment.ipynb` include a "Per-class Prediction Quality" section that generates two plots per split (val/test), logged to MLflow:

**Per-class ROC curves** (`per_class_roc.png`)
- 3 side-by-side subplots (one per class: short, neutral, long)
- One-vs-Rest (OvR) ROC curves per class
- Validation and test curves overlaid in each subplot
- Per-class AUC displayed in legend

**Precision/Recall/F1 by class** (`per_class_prf1.png`)
- 3 grouped bar charts (Precision, Recall, F1)
- One class per subplot with val/test bars side by side
- Allows visual comparison of per-class performance across splits

Access via: `from sklearn.preprocessing import label_binarize` and sklearn metrics (`roc_curve`, `precision_recall_fscore_support`).

## XGBoost hyperparameter tuning insights

### Signal limitation with 32 pipeline features

A comprehensive hyperparameter sweep (29 configs) revealed that **32 pipeline features alone cannot generate confident predictions** for 3-class XGBoost on forex M1→M15 data:

- All configs produce P(long) in range **[0.44, 0.55]** — the model never reaches 0.57
- Regularization tuning (`max_depth`, `reg_lambda`, `learning_rate`) does **not** change this range
- This is a **feature problem, not a hyperparameter problem** — the signal (RSI, ADX, spreads, lags, TAs) is insufficient for confident predictions

**Anti-overfit config (minimizes val–test gap):**
- `years: [2021, 2022, 2023]` (multiyear training, not single-year)
- `max_depth: 4` (deeper reduces stability; default 5 → 4)
- `reg_lambda: 5.0` (primary overfit lever; 1.0 → 5.0)
- `learning_rate: 0.02` (slower convergence, smoother; 0.05 → 0.02)
- `min_child_weight: 80` (tighter splits; 50 → 80)
- `gamma: 0.3` (light split-gain penalty; 0.1 → 0.3)
- Convergence: `best_iter` ~14 (vs. ~4 on single-year), logloss gap 0.009 → 0.003

**Trap: Asymmetric barriers (e.g., k_up=2.0) appear to "fix" low confidence** — wider take-profit means fewer bars labeled long (62% become short), so the model learns the base rate and predicts P(short)>0.60 for 94% of samples. This is class-imbalance exploitation, not a learned pattern. **Do not use.**

**Path forward:** Add Chronos or Kronos features. Rolling quantile signals push P(long/short) wider and enable confident predictions. See `xgboost_chronos_experiment.ipynb`.

### FVG Fractal Strategy (`strategy_fvg_fractals/`)

**Status:** Prototype with live-timing safeguards. Do not deploy without a P&L backtest.

**Purpose:** Predict which fractal level (from an 8-level configuration: base M15 + higher H1, each with last/second, high/low) will break first following a Fair Value Gap event.

**Module structure:**
- `pipeline.py` (669 lines) — FVGFractalPipeline class; fractal detection, feature engineering, target labeling
- `xgboost_fvg_fractals.ipynb` — XGBoost 3-class classifier (1=upper, -1=lower, 0=ambiguous/none)

**Data leakage issues identified (May 2025):**

| Issue | Severity | Type | Details |
|-------|----------|------|---------|
| **Structural target geometry** | Critical | Non-temporal | Features (`*_dist_signed_atr`) measure signed ATR distance from decision price to each watched fractal level. Target (`target_first_break_dir`) is which side breaks first, so geometry remains highly predictive. `pipeline.py` now defaults to `require_unbroken_levels=True`, so levels already crossed by `decision_close` are not eligible as future breaks. This reduces an outright labeling flaw but does not make accuracy equivalent to trading edge. |
| **H1 fractal look-ahead in pipeline.py** | Fixed | Temporal | `pipeline.py` now uses fractal availability time (`confirmation_time + timeframe`) and event `decision_time`, so H1 fractals are not used before the confirming H1 candle has closed. |
| **M15 fractal confirmation timing** | Fixed | Temporal | `pipeline.py` now requires fractals to be available by `decision_time`. With a 5-bar fractal, the center is at least 3 base bars old at the decision point. |
| **Accuracy ≠ trading profitability** | Critical | Metric mismatch | 96% accuracy at predicting which fractal breaks first does not imply 96% profitable trades. If upper fractal is 1 pip away and lower is 50 pips away, the model is correct 98% of the time but earns 1 pip while risking 50. Confidence threshold sweep shows model is uniformly confident (learned geometry, not market dynamics). |

**Key findings:**
- Naive geometric predictor (closest level wins): **91.3% accuracy**
- XGBoost 3-class val/test: **95.6% accuracy** (~4 points above baseline)
- FVG direction alone correlates: bullish FVG → 80% upper fractal; bearish → 80% lower
- **The 96% AUC is explained by geometry, not trading signal.**

**Recommendation:** If pursuing FVG fractals:
1. Keep `decision_delay_bars`, `require_unbroken_levels`, and `single_timeframe` explicit in each experiment config.
2. **Reframe the target:** Instead of "which level breaks first," predict **directional reversal quality** (how far price travels after break, how clean the reversal). Requires different features & target.
3. **Separate the geometry from the signal:** Build geometry-agnostic features (e.g., FVG fill speed, retest structure) that don't directly encode distance to watched levels.
4. **Evaluate on expected return, not accuracy:** For every prediction, compute expected P&L (price × probability × risk/reward geometry) rather than classification accuracy.

## Known quirks

**tb_label + binary objective mismatch:** `extract_xy()` with `tb_label` produces labels 0/1/2 — use `objective: "multi:softprob"` with `num_class: 3`. If you remap to binary (`y == 1`), use `objective: "binary:logistic"` instead.

**sklearn roc_auc_score with 3-class y:** pass full `(n,3)` proba array with `multi_class="ovr"` — do not pass a 1D slice.

**Weekend visual gaps in Plotly:** data has no weekend rows, but Plotly's calendar x-axis shows gaps. Use `weekends="nogap"` for `rangebreaks` fix, or `"filled"` for continuous data.

**Jupyter kernel caching:** after editing `chronos_inference.py`/`chronos_plots.py`, restart kernel or:
```python
import importlib, Chronos.chronos_inference as _m
importlib.reload(_m)
from Chronos.chronos_inference import ChronosForecaster, MODEL_QUANTILES
```

**Quantiles dict float keys:** `qs[0.01]` etc. — theoretically fragile but works since keys/lookups use same literals. Switch to string keys if numeric issues arise.

**KronosPredictor device parameter:** Does not accept `device="auto"` — pass concrete device string ("cuda:0", "mps", "cpu"). `KronosForecaster.__init__(device="auto")` **does** accept "auto" and resolves it internally before passing to `KronosPredictor`. Use `KronosForecaster(device="auto")` in user code, not raw `KronosPredictor(device="auto")`.
