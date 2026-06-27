# Timeseriesfm — TimesFM 2.5 Zero-Shot Forecasting

Google's [TimesFM 2.5](https://huggingface.co/google/timesfm-2.5-200m-pytorch) (200M params, PyTorch) integrated into the Trade-bot pipeline, mirroring the `Chronos/` folder structure.

**Package:** `timesfm[torch]` 2.0.0 (installed from GitHub)  
**Model weights:** auto-downloaded to `Timeseriesfm/model/` on first `load()` call (gitignored)  
**Context limit:** up to 16 384 bars  

---

## Differences from Chronos

| | TimesFM 2.5 | Chronos-2 |
|---|---|---|
| Quantile levels | 10 (mean + deciles 0.1–0.9) | 21 (0.01–0.99) |
| Column prefix | `tfm_q{pp}_h{H}` | `q{pp}_h{H}` |
| Parquet tag | `_tfm_` in filename | no tag |
| Mean quantile key | `0.0` | n/a |
| Max context | 16 384 bars | 8 192 bars |

---

## Quantile layout

`quantile_matrix` has shape `(10, pred_len)`:

| Index | Key | Meaning |
|---|---|---|
| 0 | `0.0` | Arithmetic mean |
| 1 | `0.1` | 10th percentile |
| 2 | `0.2` | 20th percentile |
| … | … | … |
| 5 | `0.5` | Median (= point forecast) |
| … | … | … |
| 9 | `0.9` | 90th percentile |

---

## Files

### `timesfm_inference.py` — `TimesFMForecaster`

```python
from Timeseriesfm.timesfm_inference import TimesFMForecaster

forecaster = TimesFMForecaster(context_length=512, device="auto")
forecaster.load()

# From a raw OHLCV DataFrame
result = forecaster.forecast_from_df(price_df, context_end="2023-09-01 00:00", prediction_length=48)

# From a ForexPipeline results dict
result = forecaster.forecast_from_pipeline(results, prediction_length=48, context_end="train_end")

# P(price > threshold) at each future bar
prob_up = TimesFMForecaster.prob_above(result["forecast"], threshold=last_close)
```

**`forecast()` return dict:**
- `context` — np.ndarray, last context_length bars of input
- `quantile_matrix` — np.ndarray `(10, pred_len)`
- `model_quantiles` — `[0.0, 0.1, …, 0.9]`
- `quantiles` — `{float: np.ndarray}` — keyed by quantile level
- `median` — np.ndarray `(pred_len,)` — alias for `quantiles[0.5]`

**`forecast_from_df()` adds:** `context_df`, `ground_truth_df`, `forecast_timestamps`, `context_end`, `forecast_start`, `prediction_length`, `pair`

---

### `timesfm_plots.py` — Matplotlib fan chart

Dark TradingView theme, 2-row layout:
- **Row 1 (3:1):** last N candlesticks + 5 quantile lines (P10/P30/P50/P70/P90) + optional ground truth overlay (faded)
- **Row 2 (1:1):** P(price > last close) — green/red fill around 50%

```python
from Timeseriesfm.timesfm_plots import plot_forecast, save_png

fig = plot_forecast(result, context_bars_shown=50)
save_png(fig, "timesfm_forecast.png")
```

Standalone script:
```bash
python Timeseriesfm/timesfm_plots.py
```

---

### `timesfm_features.py` — Rolling feature generator

Slides a context window along historical data, running TimesFM every `calc_interval` bars, and saves quantile predictions as a parquet aligned to the pipeline's datetime index.

```python
from Timeseriesfm.timesfm_features import generate

df = generate(
    pair           = "EURUSD",
    years          = [2023],
    timeframe      = "H1",
    context_length = 512,
    horizons       = [5, 10, 15, 20],
    percentiles    = [0.1, 0.3, 0.5, 0.7, 0.9],
    calc_interval  = 10,          # rerun every 10 bars; copy for 9
    norm_method    = "log_returns",
    weekends       = "filled",
    scaling        = "none",
)
# → featdata/EURUSD_H1_tfm_ctx512_int10_h5-10-15-20_logret_wfilled_snone_2023.parquet
```

**Output columns:** `tfm_q{pp}_h{H}` for each percentile × horizon + `run_id` + `staleness`

**Valid `percentiles` values:** `{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}` (use `0.0` for arithmetic mean)

**`norm_method` transforms each predicted price to match the pipeline:**
- `"log_returns"` → `log(pred / last_actual_close)`
- `"fracdiff"` → applies `_fracdiff_weights(d, threshold)` over `[…history, pred]`
- `"raw"` → no transform

**Joining with pipeline features:**
```python
feat_df = pd.concat([results["train_raw"], results["val_raw"], results["test_raw"]])
tfm     = pd.read_parquet("featdata/...parquet")
q_cols  = [c for c in tfm.columns if c.startswith("tfm_q") and "_h" in c]
combined = feat_df.join(tfm[q_cols], how="left")
```

**Parquet filename format:**
```
{PAIR}_{TF}_tfm_ctx{C}_int{I}_h{h1}-{h2}-..._{norm}_{wknd}_{scale}_{year}.parquet
```

Standalone script (edit params at bottom of file):
```bash
python Timeseriesfm/timesfm_features.py
```

---

### `timesfm_feature_plot.py` — Interactive Plotly visualization

3-panel interactive HTML comparing TimesFM predictions against pipeline-normalized close.

```python
from Timeseriesfm.timesfm_feature_plot import plot_file

fig = plot_file(
    "featdata/EURUSD_H1_tfm_ctx512_int10_h5-10-15-20_logret_wfilled_snone_2023.parquet",
    plot_horizon = 10,           # which horizon to display; None = smallest
    plot_start   = "2023-06-01",
    plot_end     = "2023-09-30",
    threshold    = 6e-4,         # must match generation params if fracdiff was used
)
fig.show()
```

**Layout:**
- **Row 1 (55%):** Normalized actual close + quantile lines for selected horizon
- **Row 2 (30%):** Forecast error (`tfm_q50_h{H}` − actual), green/red fill
- **Row 3 (15%):** Staleness (0 = fresh TimesFM run, counts up to `calc_interval − 1`)

Dotted vertical lines mark TimesFM re-run events. Output HTML saved to the same folder as the parquet.

CLI:
```bash
python Timeseriesfm/timesfm_feature_plot.py featdata/EURUSD_H1_tfm_ctx512_int10_h5-10-15-20_logret_wfilled_snone_2023.parquet
```

---

## Configuration tips

| param | recommended | note |
|---|---|---|
| `context_length` | 512 | good balance; model supports up to 16 384 |
| `prediction_length` | ≤ 128 (ctx=512) | stay within ¼ of context for reliable quantiles |
| `calc_interval` | 5–20 | higher = faster generation; lower = fresher predictions |
| `norm_method` | `"log_returns"` | matches pipeline default; use `"fracdiff"` only if pipeline uses it |
| `weekends` | `"filled"` | required for continuous context; `"nogap"` for pipeline features |
| `percentiles` | `[0.1, 0.3, 0.5, 0.7, 0.9]` | use `0.0` to add the mean column |
