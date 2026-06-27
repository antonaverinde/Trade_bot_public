# Visualize Reference

`visualize.py` — interactive Plotly chart from a pipeline results dict.  
Flow: **pipeline results → slice window → candlestick + feature subplots → HTML / .show()**

---

## Quick Start

```python
from visualize import plot
from Pipeline.pipeline import ForexDataLoader, ForexPipeline

loader   = ForexDataLoader()
pipeline = ForexPipeline()
df_m1    = loader.load_and_merge("histdata/", pair="EURUSD")
results  = pipeline.run(df_m1, timeframe="H1")

# Price only — full dataset
plot(results).show()

# Validation period with indicator subplots
plot(results, split="val", features=["rsi_14", "adx_14", "bb_pct_b"]).show()

# Save to HTML
plot(results, split="val", features=["rsi_14"]).write_html("chart.html")
```

---

## `plot()` Parameters

```python
plot(
    results,
    *,
    split    = None,
    start    = None,
    end      = None,
    features = None,
    height   = 900,
) -> go.Figure
```

| Parameter | Default | Description |
|---|---|---|
| `results` | — | Dict returned by `pipeline.run()` |
| `split` | `None` | Restrict window to one split: `"train"`, `"val"`, `"test"`, or `None` (all data) |
| `start` | `None` | ISO date string — further restrict window start, e.g. `"2023-06-01"` |
| `end` | `None` | ISO date string — further restrict window end |
| `features` | `None` | List of feature names → one subplot row each. `None` = price panel only |
| `height` | `900` | Figure height in pixels |

`split` and `start`/`end` compose: `split="val"` sets the base window to the val date range, then `start`/`end` can narrow it further.

---

## Usage Patterns

Price only — all data:
```python
plot(results)
```

Train split, no indicators:
```python
plot(results, split="train")
```

Val split with three indicators:
```python
plot(results, split="val", features=["rsi_14", "adx_14", "bb_pct_b"])
```

Explicit date range:
```python
plot(results, start="2023-01-01", end="2023-06-01", features=["rsi_14"])
```

Val split narrowed to a specific month:
```python
plot(results, split="val", start="2023-09-01", end="2023-10-01")
```

Taller figure for many feature rows:
```python
plot(results, split="val",
     features=["rsi_14", "rsi_21", "adx_14", "bb_pct_b", "atr_rel", "vol_ewma"],
     height=1400)
```

See all available feature names:
```python
print(results["feature_cols"])
```

---

## Chart Layout

The figure is always one column with shared x-axes:

| Row | Content | Height share |
|---|---|---|
| 1 | Candlestick (real OHLCV price levels) | 50 % (or 100 % if no features) |
| 2…N | One line chart per requested feature | Remaining 50 % split equally |

**Split-region shading** — colored background bands appear in every row:

| Color | Split |
|---|---|
| Teal | train |
| Blue | val |
| Red | test |

The bands appear even when `split=None`, so you can always see where the splits fall within the visible window.

---

## Feature Reference Lines

Some features automatically get horizontal guide lines:

| Feature | Lines |
|---|---|
| `rsi_14`, `rsi_21` | 70 (overbought), 30 (oversold) |
| `adx_14` | 25 (trending threshold) |
| `bb_pct_b` | 1.0 (upper band), 0.5 (midline), 0.0 (lower band) |

Other features are plotted as plain lines with no guides.

---

## Available Features

All 32 features from `results["feature_cols"]`:

### RSI (10)
`rsi_14`, `rsi_14_speed`, `rsi_14_accel`, `rsi_14_cross_50`, `rsi_14_cross_70`  
`rsi_21`, `rsi_21_speed`, `rsi_21_accel`, `rsi_21_cross_50`, `rsi_21_cross_70`

### ADX — Trend Strength (3)
`adx_14`, `di_diff`, `adx_delta`

### Trend & Volatility (3)
`dist_ema200`, `atr_rel`, `bb_pct_b`

### Time (6)
`hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `is_monday`, `is_friday`

### Candle Structure (3)
`body_ratio`, `shadow_ratio`, `body_gap`

### Distribution (3)
`ret_skew`, `ret_kurt`, `vol_ewma`

### Lags (4)
`close_lag1`, `close_lag2`, `close_lag5`, `close_lag10`

---

## Notes

**Price levels** — the candlestick uses `results["raw_m1"]` resampled to the pipeline timeframe, so it always shows real price levels (e.g. 1.0850), not normalized log returns.

**Feature values** — indicator subplots use unscaled values from `train_raw`/`val_raw`/`test_raw`, so RSI reads 0–100, ADX reads 0–100, etc.

**Gap bars** — the 50-bar gaps between splits are excluded from `*_raw` DataFrames. Feature subplots will show a small break at the split boundaries; the price candlestick is continuous.

**Interactivity** — zoom, pan, and legend click-to-toggle are built in. A range selector (1W / 1M / 3M / 6M / All) appears on the bottom x-axis. Hovering shows unified tooltips for all panels at the cursor position.

**Invalid feature name** — `plot()` raises `ValueError` listing all valid names if an unknown feature is passed.

---

## Running as a Script

```bash
uv run python visualize.py
```

Produces `chart.html` — val split of EURUSD H1 with `rsi_14`, `adx_14`, `bb_pct_b`.
