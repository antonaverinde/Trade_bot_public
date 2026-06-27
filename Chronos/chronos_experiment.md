# Chronos-2 Zero-Shot Forex Forecast

Zero-shot probabilistic price forecasting using Amazon's
[Chronos-2](https://huggingface.co/amazon/chronos-2) foundation model.
No fine-tuning — the model predicts out-of-the-box from raw close prices.

Chronos-2 is a **quantile-regression model**: it outputs 21 probability levels directly in a
single forward pass, without Monte Carlo sampling. This makes it faster, smaller, and more
accurate than the older Chronos-T5 family.

---

## Files

| File | Purpose |
|---|---|
| `chronos_inference.py` | `ChronosForecaster` class — model loading, forecast, CDF helpers |
| `chronos_plots.py` | Matplotlib dark-theme chart — candlesticks + 4-band fan + P(up) subplot |
| `chronos_experiment.ipynb` | Main notebook — load data, run forecast, plot, save PNG |
| `model/` | Downloaded model weights (auto-created on first run, ~480 MB bfloat16) |

---

## Quick Start

```bash
cd ~/Trade_bot
source .venv/bin/activate
jupyter notebook Chronos/chronos_experiment.ipynb
```

Run cells top-to-bottom. Weights download to `Chronos/model/` on first run.
Output: `chronos_forecast.png` and `chronos_forecast_future.png` in the project root.

### Standalone script (no notebook)

```bash
cd ~/Trade_bot
source .venv/bin/activate
python Chronos/chronos_plots.py        # writes chronos_forecast.png to project root
```

---

## Why Chronos-2 over Chronos-T5

| | chronos-t5-large | **chronos-2** |
|---|---|---|
| Parameters | 710 M | **120 M** |
| VRAM (bfloat16) | ~2.8 GB | **~480 MB** |
| Context window | 2 048 bars | **8 192 bars** |
| Max prediction | ~512 bars | **1 024 bars** |
| Output type | Monte Carlo samples | **21 quantiles directly** |
| Inference speed | slow (N sample passes) | **fast (single pass)** |
| Zero-shot accuracy | good | **better on benchmarks** |

Chronos-2 fits easily in an 8 GB GPU and leaves ample VRAM headroom.

---

## GPU Requirements

`amazon/chronos-2` in `bfloat16` uses approximately **480 MB VRAM** — negligible on an 8 GB card.
The base model is the only Chronos-2 size currently available; there is no tiny/small/large family.

If you want to run on CPU, set `device="cpu"` and `dtype=torch.float32` in `CHRONOS_CFG`.

---

## Configuration Reference

All config lives in **Cell 2** of the notebook.

### `CFG` — single config dict

| Key | Default | Description |
|---|---|---|
| `pair` | `"EURUSD"` | Currency pair (must match a file in `histdata/`) |
| `years` | `[2022, 2023]` | Which annual M1 CSVs to load |
| `timeframe` | `"H1"` | Resample target: `M5` `M15` `M30` `H1` `H4` `D1` |
| `context_end` | `"2023-09-01 00:00"` | **Where the forecast starts** — any ISO datetime in the loaded range |
| `context_length` | `512` | Bars fed to the model (max 8 192) |
| `prediction_length` | `48` | Bars to predict ahead (48 H1 ≈ 2 trading days) |
| `context_bars_shown` | `50` | Candlesticks shown in the chart before the "now" line |
| `device` | `"auto"` | `"auto"` selects CUDA if available, else CPU |
| `dtype` | `torch.bfloat16` | Use `torch.float32` on CPU or older GPUs |

**`context_end`** accepts any ISO datetime string within the loaded data range.
Run Cell 3 first to see the exact range, then set `context_end` accordingly.

| `context_end` value | Forecast starts | Ground truth? |
|---|---|---|
| `"2023-09-01 00:00"` | 2023-09-01 01:00 | yes — bars after that date exist |
| `price_df.index[-1]` | last bar + 1 period | no — genuine future |

---

## Model Output: 21 Fixed Quantiles

Chronos-2 always outputs these 21 quantile levels — they are baked into the model weights
and cannot be changed without fine-tuning:

```
0.01  0.05  0.10  0.15  0.20  0.25  0.30  0.35  0.40  0.45
0.50
0.55  0.60  0.65  0.70  0.75  0.80  0.85  0.90  0.95  0.99
```

The `forecast()` method returns these as a dict `{float: np.ndarray(pred_len)}` and as a
`(21, pred_len)` `quantile_matrix` for direct indexing.

---

## What the Notebook Produces

### Cell 6 — Forecast Summary Table
Prints quantile values (P05 / P25 / P50 / P75 / P95) and `P(up)` at four forecast horizons:
25 %, 50 %, 75 %, and 100 % of `prediction_length`.
When ground truth exists, shows the actual close and whether it fell inside the P10–P90 band.

### Cell 7 — Fan Chart (`chronos_forecast.png`)
Two-row matplotlib chart with dark TradingView theme:

**Row 1 — Price chart**
- Teal/red candlesticks for the last `context_bars_shown` bars of history
- Faded candlesticks showing actual ground-truth prices (when available)
- Four nested blue fan bands: P01–P99 → P05–P95 → P10–P90 → P25–P75 (outermost to innermost)
- Solid blue line = P50 median forecast
- Gold dashed "now" separator
- Right-side labels: P05 / P25 / P50 / P75 / P95 values at the final forecast bar

**Row 2 — Directional probability**
- `P(price_t > last close)` at each future step via CDF interpolation
- Teal fill where P(up) > 50 %, red fill where P(up) < 50 %
- Gold dotted 50 % reference line

### Cell 8 — Future Chart (`chronos_forecast_future.png`)
Same chart but with `context_end = price_df.index[-1]` — the forecast starts after the last
bar in your data. No ground truth overlay.

---

## How P(up) Is Computed

Unlike Chronos-T5 which estimated directional probability by counting samples above the
threshold (`np.mean(samples > last_close)`), Chronos-2 uses CDF interpolation:

```python
# At each forecast step t:
# q_matrix[:, t] = 21 sorted price values (quantile ladder)
# model_qs       = [0.01, 0.05, ..., 0.99]
cdf_at_last_close = np.interp(last_close, q_matrix[:, t], model_qs)
prob_up[t] = 1.0 - cdf_at_last_close
```

This is exact (not a sample approximation) and requires no stochastic sampling.

---

## How Chronos-2 Works (Brief)

Chronos-2 uses a patch-based encoder-decoder architecture. The input time series is divided
into patches; the encoder embeds each patch into a latent representation; the decoder autoregressively
generates output patches, each producing a full quantile forecast for that patch's time steps.

Unlike Chronos-T5 (which tokenises values into discrete bins and samples token sequences),
Chronos-2 directly regresses the quantile values, making it deterministic and faster.
Context length up to 8 192 bars allows it to capture longer-term seasonality and trends.

---

## Reusing the Forecaster in Other Scripts

```python
import sys
import pandas as pd
sys.path.insert(0, "/home/anton/Trade_bot")

from Pipeline.pipeline import ForexDataLoader
from Chronos.chronos_inference import ChronosForecaster
from Chronos.chronos_plots import plot_forecast, save_png

loader   = ForexDataLoader()
df_m1    = loader.load_and_merge("histdata/", pair="EURUSD", years=[2023])

# Resample M1 → H1
price_df = (
    df_m1.resample("1h")
    .agg({"open": "first", "high": "max", "low": "min",
          "close": "last", "volume": "sum"})
    .dropna()
)
price_df.attrs["pair"] = "EURUSD"

fc = ChronosForecaster(context_length=512)
result = fc.forecast_from_df(price_df, context_end="2023-10-01 00:00", prediction_length=48)
result["timeframe"] = "H1"

# Directional probability at each step
prob_up = ChronosForecaster.prob_above(result["forecast"], threshold=result["context_df"]["close"].iloc[-1])

fig = plot_forecast(result, context_bars_shown=50)
save_png(fig, "my_forecast.png")
```

For use with XGBoost pipeline output, `forecast_from_pipeline` is also available:

```python
from Pipeline.pipeline import ForexPipeline

pipeline = ForexPipeline(norm_method="log_returns", target_type="lag")
results  = pipeline.run(df_m1, timeframe="H1")

result = fc.forecast_from_pipeline(results, prediction_length=48, context_end="train_end")
```

---

## Tips

- **Longer context**: increasing `context_length` toward 2048–4096 can improve forecast
  quality on hourly/daily timeframes where seasonality matters, at the cost of slightly
  more memory and time.
- **Longer horizons**: Chronos-2 supports up to 1024 bars of prediction. Quality degrades
  gradually beyond `prediction_length / context_length > 0.25` — keep the ratio low.
- **Multiple pairs**: call `forecast_from_pipeline` once per pair; the loaded model is reused.
- **CPU fallback**: `device="cpu"`, `dtype=torch.float32` — prediction is slower but works
  without a GPU. Reduce `context_length` to 256–512 to keep latency manageable.
