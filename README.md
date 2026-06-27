# Trade Bot — Forex ML Pipeline

End-to-end pipeline that turns raw M1 OHLCV forex data into model-ready features and walk-forward train/val/test splits, with XGBoost training and Chronos zero-shot forecasting.

---

## Pipeline Overview

```
Raw CSV files
     │
     ▼
[1] Load & Merge      — parse HistData CSVs, merge years per pair
     │
     ▼
[2] Resample          — M1 → M5 / M15 / H1 / H4 / D1
     │
     ▼
[3] Feature Engineer  — 40 features: returns, volatility, technicals,
     │                  microstructure, time/session, lags + targets
     ▼
[4] Walk-Forward Split — 60 / 20 / 20 with 50-bar leakage gap
     │
     ▼
[5] Scale             — Robust median/IQR, fitted on train only
     │
     ▼
Model-ready (X_train, y_train), (X_val, y_val), (X_test, y_test)
```

---

## Data

| Source | Pairs | Timespan | Granularity | Bars |
|--------|-------|----------|-------------|------|
| HistData.com | EURUSD, GBPUSD | 2020–2024 | M1 | ~1.8M per pair |

Files live in `histdata/` as annual CSVs: `DAT_ASCII_{PAIR}_M1_{YEAR}.csv`

---

## Features (40 total)

| Group | Features |
|-------|----------|
| **Returns** | `log_return`, `log_return_high`, `log_return_low` |
| **Volatility** | `atr_14`, `atr_norm`, `vol_roll_1h`, `vol_roll_1d` |
| **Technicals** | `rsi_6`, `rsi_14`, `dist_ema20/50/200`, `bb_upper`, `bb_lower`, `bb_width`, `macd_norm`, `macd_signal`, `macd_hist` |
| **Microstructure** | `hl_spread`, `log_volume`, `vol_zscore`, `body_ratio`, `candle_dir` |
| **Time / Session** | `hour_sin/cos`, `dow_sin/cos`, `sess_asia`, `sess_london`, `sess_ny`, `sess_overlap` |
| **Lags (1, 5, 15)** | `ret_lagN`, `vol_lagN`, `rsi_lagN` |

### Targets

| Column | Type | Description |
|--------|------|-------------|
| `direction_1/5/15` | Classification | 1 if close is higher N bars ahead, else 0 |
| `future_ret_1/5/15` | Regression | Log return N bars ahead |
| `tb_label` | Classification | Triple-barrier label: 1 upper hit, -1 lower hit, 0 timeout |
| `tb_ret` | Regression | Realized exit movement on the selected barrier basis |

---

## Models

### XGBoost
Trained per timeframe on pipeline targets. Triple-barrier experiments default to raw-price high/low barrier checks (`barrier_norm_method="raw"`, `barrier_price="hl"`) while feature output remains controlled by `norm_method`.

### Chronos (Amazon)
Zero-shot time-series forecast on the close price series. No training required — uses the pretrained `chronos-t5-small` model. Requires `torch` and `chronos-forecasting` to be installed.

---

## Project Structure

```
Trade_bot/
├── Pipeline/
│   └── pipeline.py     # ForexDataLoader, FeatureEngineer, WalkForwardSplitter,
│                       # ForexScaler, ForexPipeline
├── histdata/           # Raw annual CSVs from HistData.com
├── main.py             # Entry point: load → pipeline → XGBoost → Chronos
└── pyproject.toml
```

---

## Usage

```bash
# Install dependencies
uv sync

# Optional: install Chronos (large — requires PyTorch)
uv add chronos-forecasting torch

# Run smoke test (synthetic data)
uv run python Pipeline/pipeline.py

# Run full pipeline on real histdata
uv run python main.py
```

---

## Design Insights

### Why raw OHLCV is excluded from features

Raw prices are non-stationary — a model trained on 2020 EURUSD levels (~1.10) will see
meaningless numbers in 2024 (~1.08). The model would learn price thresholds that never
generalize. Every piece of information from raw OHLCV is already represented in a
stationary, normalized form:

| Raw | Stationary replacement |
|-----|----------------------|
| Close price | `log_return`, `dist_ema20/50/200` |
| High / Low | `hl_spread`, `atr_14`, `bb_upper/lower` |
| Volume | `log_volume`, `vol_zscore` |

**Exception:** with `scaling="rolling"` and `include_raw=True`, raw prices become
stationary through window normalization: `(close − rolling_median) / rolling_IQR`
represents distance from recent price level, which is meaningful and distinct from
the existing features. Never use `include_raw=True` with global scaling.

---

### Lags vs target horizons

**Lags** (`lags=[1, 5, 15]`) — how far *back* to look. Creates copies of return,
volatility, and RSI shifted back by 1, 5, and 15 bars. At bar T the model sees what
those values were at T-1, T-5, T-15 — all in the same flat row. This gives a tabular
model (XGBoost) short-term memory without needing a recurrent architecture.

**Target horizons** (`target_horizons=[1, 5, 15]`) — how far *forward* to predict.
Creates targets for 1, 5, and 15 bars ahead. You pick one column at training time.

They are independent:
- Lags → affect the **input** (width of the feature set)
- Horizons → affect the **output** (what the model is trying to predict)

Each bar in the dataset is a question-answer pair:

> **Question (X):** "Given everything observable at bar T, what is the market state?"
> **Answer (y):** "The next bar went up (1) or down (0)."

---

### Global scaling vs rolling (window) scaling

**Global scaler** fits median/IQR once on the train set (2020–2023), then applies those
fixed statistics to val and test. In 2024 the market may have shifted regime — different
ATR levels, different volatility — making the 2020 statistics stale.

**Rolling scaler** recomputes median/IQR from the last N bars at every point in time.
It is always regime-aware and is mandatory for live deployment.

| | Global | Rolling |
|---|---|---|
| Fit | Once on train | Per bar, from lookback window |
| Regime changes | Blind | Adapts |
| Live trading | Risky | Correct |
| First-pass benchmark | Fine | Fine |

Use `ForexPipeline(scaling="rolling", window_size=500)` to enable.

---

### XGBoost vs Transformer data format

**XGBoost** consumes a flat 2D table: `(bars, features)`. Temporal context is provided
manually via lag features. Use `lags=[1, 5, 15]`.

**Transformer / LSTM** consumes a 3D tensor: `(samples, timesteps, features)`. The model
learns its own temporal relationships through attention — lag features are redundant.
Use `lags=[]` and `seq_len=60` (or your chosen window size).

```python
# XGBoost
pipeline = ForexPipeline(lags=[1, 5, 15])
results  = pipeline.run(df_m1, "M15")
X, y     = pipeline.get_xy(results["train"], "direction_1", results["feature_cols"])
# X shape: (n_bars, 40)

# Transformer
pipeline = ForexPipeline(lags=[], seq_len=60)
results  = pipeline.run(df_m1, "M15")
X = results["train_seq_X"]   # shape: (n_samples, 60, 31)
y = results["train_seq_y"]   # shape: (n_samples,)
```

---

### Why split first, then build sequence windows

Windowing across split boundaries leaks future information into training.
The correct order:

```
Raw bars
    │
    ├── Train bars  →  sliding windows  →  (n_train, timesteps, features)
    ├── [gap — dropped to prevent label leakage]
    ├── Val bars    →  sliding windows  →  (n_val,   timesteps, features)
    ├── [gap]
    └── Test bars   →  sliding windows  →  (n_test,  timesteps, features)
```

The first `seq_len − 1` bars of val and test are sacrificed as warm-up context for
the first valid window. This is expected and correct.

---

## Environment

Uses the project virtualenv managed by `uv` (`requires-python = ">=3.12"`).

Key dependencies: `pandas`, `numpy`, `xgboost`, `scikit-learn`, `darts`, `chronos-forecasting`
