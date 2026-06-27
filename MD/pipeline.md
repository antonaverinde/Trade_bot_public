# Pipeline Reference

`Pipeline/pipeline.py` — end-to-end forex ML pipeline.  
Flow: **load → gap-fill → normalize → features → split → scale → ready for model**

---

## Quick Start

```python
from Pipeline.pipeline import ForexDataLoader, ForexPipeline

loader   = ForexDataLoader()
pipeline = ForexPipeline(
    norm_method="log_returns",
    target_type="lag",
)

df_m1   = loader.load_and_merge("histdata/", pair="EURUSD")
results = pipeline.run(df_m1, timeframe="M15")

X_train, y_train = pipeline.get_xy(results["train"], "direction_1", results["feature_cols"])
X_val,   y_val   = pipeline.get_xy(results["val"],   "direction_1", results["feature_cols"])
```

Switch to triple barrier targets:
```python
pipeline = ForexPipeline(target_type="triple_barrier", k_up=2.0, k_down=1.0, horizon_bars=10)
# target column becomes "tb_label" (values: 1, -1, 0)
X_train, y_train = pipeline.get_xy(results["train"], "tb_label", results["feature_cols"])
```

Switch to fracdiff normalization:
```python
pipeline = ForexPipeline(norm_method="fracdiff", fracdiff_d=0.4)
```

Filter by year — pass `years` to `load_and_merge`:
```python
df_m1   = loader.load_and_merge("histdata/", pair="EURUSD", years=[2023])
results = pipeline.run(df_m1, timeframe="M15")
```

Filter by arbitrary date range — slice the DataFrame after loading:
```python
df_m1   = loader.load_and_merge("histdata/", pair="EURUSD", years=[2023])
df_m1   = df_m1["2023-11-01":"2023-12-25"]
results = pipeline.run(df_m1, timeframe="M15")
```

`load_and_merge` only filters by full years. For any sub-year range, load the relevant year(s) first, then slice. The pipeline accepts any DataFrame with a DatetimeIndex.

---

## ForexPipeline Parameters

| Parameter | Default | Description |
|---|---|---|
| `lags` | `[1, 2, 5, 10]` | Which bars back to create `close_lagN` features |
| `target_horizons` | `[1, 5, 15]` | Horizons for lag-based targets (bars ahead) |
| `gap_bars` | `50` | Gap between train/val and val/test splits to prevent label leakage |
| `scaling` | `"global"` | Scaler type — see Scalers section below |
| `window_size` | `500` | Lookback window for RollingScaler only (bars) |
| `include_raw` | `False` | Include raw OHLCV columns in features (dangerous with global scaling) |
| `seq_len` | `0` | If > 0, build 3D sliding-window arrays for LSTM/Transformer — see below |
| `norm_method` | `"log_returns"` | Price normalization before output features — see below |
| `fracdiff_d` | `0.4` | Fractional diff order, used when `norm_method="fracdiff"` or `barrier_norm_method="fracdiff"` |
| `target_type` | `"lag"` | Two options: `"lag"` or `"triple_barrier"` — see below |
| `k_up` | `2.0` | Triple barrier multiplier above the selected barrier basis |
| `k_down` | `1.0` | Triple barrier multiplier below the selected barrier basis |
| `horizon_bars` | `10` | Triple barrier time limit (bars until expiry) |
| `barrier_price` | `"close"` | Price series used to check barrier hits — `"close"` or `"hl"` (see below) |
| `barrier_norm_method` | `"raw"` | Basis for triple-barrier labels — `"raw"`, `"log_returns"`, `"fracdiff"`, or `"features"` |
| `barrier_on_raw` | `True` | Backward-compatible alias: `True` -> `barrier_norm_method="raw"`, `False` -> `"features"` |
| `weekends` | `"nogap"` | Weekend handling: `"nogap"` = no weekend rows (5-day week, default); `"gaps"` = NaN rows for Sat/Sun; `"filled"` = Sat/Sun forward-filled from Friday's close |

---

### norm_method

`norm_method` controls the model feature output in `train`/`val`/`test`. It does not force the triple-barrier label basis; use `barrier_norm_method` for that.

| Value | What it does |
|---|---|
| `"log_returns"` | `log(col_t / col_{t-1})` per OHLC column. Stationary, fast. Default. |
| `"fracdiff"` | Fractional differentiation (López de Prado). Preserves more memory than log returns while keeping stationarity. Controlled by `fracdiff_d` (closer to 0 = more memory, closer to 1 = closer to full diff). |
| `"raw"` | No transformation. Chronos always gets raw prices regardless of this setting. |

---

### target_type

Two labeling strategies available:

**`"lag"` (default)** — simple forward-looking labels:
- `direction_1/5/15` — binary: 1 if price goes up in N bars, 0 if down (classification)
- `future_ret_1/5/15` — actual log return N bars ahead (regression)

**`"triple_barrier"`** — dynamic volatility-scaled barriers (López de Prado):
For each bar, three exits compete — whichever triggers first wins:
1. Upper barrier (take profit): `price × (1 + vol_ewma × k_up)` → label **1**
2. Lower barrier (stop loss): `price × (1 - vol_ewma × k_down)` → label **-1**
3. Time barrier: `horizon_bars` bars elapsed with no hit → label **0**

Output columns: `tb_label` (1/-1/0), `tb_ret` (realized exit movement on the selected barrier basis).
Barrier width adapts to current volatility — wide in volatile regimes, tight in quiet ones.

**`barrier_norm_method`** controls which OHLC basis is used for label calculation:

| Value | Barrier basis | Barrier arithmetic |
|---|---|---|
| `"raw"` (default) | raw OHLC price levels | multiplicative: `close × (1 ± sigma × k)` |
| `"log_returns"` | log-return OHLC | additive: `close ± sigma × k` |
| `"fracdiff"` | fractionally differentiated OHLC | additive: `close ± sigma × k` |
| `"features"` | the same OHLC basis produced by `norm_method` | raw features are multiplicative, transformed features are additive |

`"raw"` with `barrier_price="hl"` is the recommended XGBoost experiment default because it labels actual TP/SL touches on price levels while still allowing model features to be fracdiff or log returns.

**`barrier_price`** controls which prices are checked against the barriers each future bar:

| Value | Upper barrier check | Lower barrier check |
|---|---|---|
| `"close"` (default) | close >= upper | close <= lower |
| `"hl"` | high >= upper | low <= lower |

`"hl"` is more realistic for intrabar execution — a bar's high may have touched the take-profit level even if close settled below it. Results in more TP hits and fewer timeouts compared to `"close"`.

---

### weekends

Controls how Saturday and Sunday are represented in the data. Also available on `ForexDataLoader.load_and_merge(weekends=...)` to apply the same choice at M1 load time.

| Value | Week shape | Weekend rows | Values |
|---|---|---|---|
| `"nogap"` (default) | 5-day | Absent — index jumps Fri → Mon | — |
| `"gaps"` | 7-day calendar | Present | All NaN |
| `"filled"` | 7-day calendar | Present | Forward-filled from Friday's last close |

**When to use each:**
- `"nogap"` — default; correct for all standard XGBoost/feature-engineering use cases.
- `"gaps"` — when the model needs a 7-day index but should be able to distinguish missing weekend bars from real data (e.g. mask-aware transformers).
- `"filled"` — when the model requires a gapless continuous index (e.g. calendar-aligned time-series models, Chronos with a uniform context window).

**Visual note:** With `"nogap"`, Plotly charts show a blank ~49-hour stretch at weekends even though the DataFrame has no weekend rows — this is Plotly rendering calendar time on the x-axis, not a data issue. The visualizers (`visualize_raw.py`, `visualize_targets.py`) automatically apply `rangebreaks` to collapse this gap when `weekends="nogap"`.

---

### Scalers

Two scalers available via the `scaling` parameter. Both use **robust scaling (median + IQR)** instead of min-max or z-score, which is better for financial data with fat tails and outliers.

**`"global"` → ForexScaler** (default)
- Fits on the **training set only**: computes median and IQR per feature from train bars
- Applies the same fixed stats to val and test — no leakage
- Formula: `(value - train_median) / train_IQR`
- Best for: standard use, when regime doesn't change dramatically

**`"rolling"` → RollingScaler**
- No fitting step — for each bar, computes median + IQR over the **preceding `window_size` bars**
- Adapts to regime changes (e.g. low-vol 2017 vs high-vol 2020 crash)
- Formula: `(value - rolling_median) / rolling_IQR`
- Best for: long date ranges where market conditions shift significantly

---

### seq_len — Sequence Arrays

By default (`seq_len=0`) the pipeline returns flat 2D DataFrames `(bars, features)` — ready for XGBoost.

When `seq_len > 0`, the pipeline additionally builds 3D sliding-window arrays for sequence models (LSTM, Transformer):

```
Example: 10 bars, 3 features, seq_len=4

Flat input (10 × 3):
  bar 0: [f1, f2, f3]
  bar 1: [f1, f2, f3]
  ...
  bar 9: [f1, f2, f3]

3D output (7 samples × 4 timesteps × 3 features):
  sample 0: bars [0,1,2,3]
  sample 1: bars [1,2,3,4]
  sample 2: bars [2,3,4,5]
  ...
  sample 6: bars [6,7,8,9]
```

Real example: 100,000 bars, 32 features, `seq_len=60` → shape `(99,940, 60, 32)`.

Available in result dict as `train_seq_X`, `val_seq_X`, `test_seq_X`, `train_seq_y`, etc.

```python
pipeline = ForexPipeline(seq_len=60)
results  = pipeline.run(df_m1, "M15")
X_train  = results["train_seq_X"]  # shape: (n_samples, 60, 32)
y_train  = results["train_seq_y"]  # shape: (n_samples,)
```

---

## pipeline.run() — Output Dict

```python
results = pipeline.run(df_m1, timeframe="M15")
```

| Key | Content |
|---|---|
| `"pair"` | Pair name string, e.g. `"EURUSD"` |
| `"timeframe"` | Timeframe string, e.g. `"M15"` |
| `"feature_cols"` | List of feature column names (32 total by default) |
| `"target_cols"` | Regression target column names |
| `"direction_cols"` | Classification target column names |
| `"train"` | Scaled train DataFrame |
| `"val"` | Scaled val DataFrame |
| `"test"` | Scaled test DataFrame |
| `"train_raw"` | Unscaled train DataFrame (use for target extraction) |
| `"val_raw"` | Unscaled val DataFrame |
| `"test_raw"` | Unscaled test DataFrame |
| `"raw_m1"` | Raw OHLCV at the chosen timeframe — pass to Chronos |
| `"scaler"` | Fitted scaler object (save for inference) |
| `"norm_method"` | Feature-output normalization method used by the run |
| `"barrier_norm_method"` | Triple-barrier label basis used by the run |
| `"barrier_price"` | Triple-barrier hit-check mode used by the run |
| `"train_seq_X"` etc. | 3D arrays `(samples, seq_len, features)` — only if `seq_len > 0` |

---

## Features (32 total, default config)

### RSI (10 features)
Computed for periods 14 and 21.

| Column | Description |
|---|---|
| `rsi_14`, `rsi_21` | RSI value (0–100) |
| `rsi_14_speed`, `rsi_21_speed` | RSI change vs previous bar |
| `rsi_14_accel`, `rsi_21_accel` | Speed change (2nd derivative) |
| `rsi_14_cross_50`, `rsi_21_cross_50` | 1 if RSI crossed 50 upward this bar |
| `rsi_14_cross_70`, `rsi_21_cross_70` | 1 if RSI crossed 70 upward this bar |

### ADX — Trend Strength (3 features)

| Column | Description |
|---|---|
| `adx_14` | ADX value (0–100), trend strength |
| `di_diff` | +DI − −DI: positive = bullish pressure, negative = bearish |
| `adx_delta` | `adx_14 - adx_14.shift(3)`: trend building (+) or fading (−) |

### Trend & Volatility (3 features)

| Column | Description |
|---|---|
| `dist_ema200` | `(close − EMA200) / close` — distance from global average |
| `atr_rel` | `ATR(14) / close` — volatility relative to price level |
| `bb_pct_b` | Bollinger %B: position within the band (0 = lower, 1 = upper), clipped to [−0.5, 1.5] |

### Time (6 features)

| Column | Description |
|---|---|
| `hour_sin`, `hour_cos` | Hour of day encoded cyclically (0–23) |
| `dow_sin`, `dow_cos` | Day of week encoded cyclically (0=Mon, 4=Fri) |
| `is_monday` | Binary flag |
| `is_friday` | Binary flag |

Note: time features are derived from the index directly (not rolling windows), so they are unaffected by `dropna()`. Weekends are already removed in `load_and_merge()` before any features are computed, so sin/cos values are always valid trading-day positions (0=Mon through 4=Fri).

### Candle Structure (3 features)

| Column | Description |
|---|---|
| `body_ratio` | `|close − open| / (high − low + 1e-10)` — 0 = doji, 1 = marubozu |
| `shadow_ratio` | `upper_wick / (high − low + 1e-10)` — upper shadow share (0→1) |
| `body_gap` | `(open − prev_close) / prev_close` — normalized gap between bars |

### Distribution (3 features)

| Column | Description |
|---|---|
| `ret_skew` | Rolling skewness of close over 100 bars |
| `ret_kurt` | Rolling kurtosis of close over 100 bars (fat tails) |
| `vol_ewma` | EWMA volatility span=100 — same metric used in triple barrier labeling |

### Lags (4 features)

| Column | Description |
|---|---|
| `close_lag1/2/5/10` | Normalized close shifted N bars back |

---

## Classes & Functions

### `ForexDataLoader`
- **`load_csv(path, pair)`** — load a single HistData CSV file
- **`load_and_merge(histdata_dir, pair, years=None)`** — merge all annual CSVs for a pair, ffill intra-week gaps, drop weekends
- **`generate_synthetic(pair, n_bars, start)`** — GARCH-like synthetic M1 data for testing

### `normalize_prices(df, method, d)`
Standalone function. Transforms OHLCV before feature engineering. See `norm_method` table above.

### `FeatureEngineer`
Internal class, used by `ForexPipeline`. Methods:
- `_rsi_features` — RSI 14 & 21 with speed, accel, crosses
- `_adx_features` — ADX, DI diff, ADX delta
- `_ma_distance` — EMA 200 distance
- `_relative_atr` — ATR / price
- `_bollinger_pct_b` — Bollinger %B
- `_time_features` — sin/cos time encoding + day flags
- `_candle_structure` — body ratio, shadow ratio, body gap
- `_distribution_features` — skew, kurtosis, EWMA vol
- `_lags` — lagged close values
- `_triple_barrier_targets` — López de Prado triple barrier labeling

### `WalkForwardSplitter`
Splits into train (60%) / val (20%) / test (20%) with `gap_bars` gap between each to prevent label leakage.

### `ForexScaler` (global)
Robust scaler: median + IQR computed from train bars only, then applied identically to val and test. Scale is centered around the train median with units of train IQR — so a scaled value of 2.0 means "2 IQRs above the training median".

### `RollingScaler` (rolling)
Normalizes each bar using median + IQR of the preceding `window_size` bars. Regime-aware, no fit step needed. Each split (train/val/test) is scaled independently using its own history.

### `ForexPipeline`
- **`run(df_m1, timeframe)`** — full pipeline, returns result dict
- **`get_xy(split, target, feature_cols)`** — extract `(X_array, y_array)` from a split

### `build_sequences(df, feature_cols, target_col, seq_len)`
Converts flat DataFrame to 3D arrays `(n_samples, seq_len, n_features)` for LSTM/Transformer. Activated automatically when `seq_len > 0`.

---

## Supported Timeframes

`"M1"`, `"M5"`, `"M15"`, `"H1"`, `"H4"`, `"D1"`

Pass to `pipeline.run(df_m1, timeframe=...)`. Resampling from M1 is automatic.

---

## Data Format (HistData CSVs)

```
20200102 170100;1.12034;1.12041;1.12034;1.12038;0
```

- Semicolon-delimited, no header
- Columns: `YYYYMMDD HHMMSS ; open ; high ; low ; close ; volume`
- Volume is always 0 in HistData — replaced with 1.0 placeholder
- File naming: `DAT_ASCII_EURUSD_M1_2020.csv`
