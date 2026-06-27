# XGBoost Important Parameters

## Fracdiff vs Scaling — they don't overlap

### Fracdiff = stationarization

Fractional differentiation (López de Prado) removes the trend/unit root from the price series so it has a stable distribution over time. The output still carries its own variance — during a high-volatility month fracdiff values are larger, during a quiet month smaller. **It does not normalize to unit variance.**

### Scaling = variance normalization

`ForexScaler` (robust, global) or `RollingScaler` (window-based) divides by IQR so all features live on a comparable numeric range.

**These two steps address different problems and must both be applied:**

| Step | Removes | Leaves |
|------|---------|--------|
| Fracdiff | Trend / unit root | Regime-scale variance intact |
| Scaling | Magnitude differences between features | Zero-mean, unit-IQR |

---

## Fracdiff order `d`

### What `d` controls

- `d = 0` — raw prices (non-stationary, unit root)
- `d = 1` — simple differences (fully stationary, but throws away all price memory)
- `d ∈ (0, 1)` — partial memory: stationary enough for ML, but retains long-range dependencies

### Empirical optimum

| Timeframe | Optimal `d` | Notes |
|-----------|------------|-------|
| H1 | ~0.17 | Passes ADF at p < 0.1 while preserving most price memory |
| M15 | ~0.17 | Same result — relatively stable across intraday frames |
| Other | run `Supplementary/fracdiff_adf.py` | Sweeps d=0→1, finds min d where ADF p < sig_level |

Values much higher than 0.17 (e.g., 0.4–0.5) are safe statistically but waste memory unnecessarily. Values below the ADF threshold produce a non-stationary series that will degrade model generalization.

---

## Fracdiff threshold parameter

`_fracdiff_weights(d, threshold=6e-4)` in `pipeline.py:175`

The threshold controls the **warm-up window length** — how many past bars are needed before a single valid output row can be produced. Lower threshold → more historical weights included → longer warm-up → more memory captured.

### Threshold vs warm-up vs memory retained

| `threshold` | Approx weights (warm-up bars) | % info captured |
|------------|-------------------------------|-----------------|
| `6e-4` (default) | ~150 | ~91% |
| `3e-4` | ~200 | 92.9% |
| `1e-4` | ~500 | 95.7% |
| `4.3e-5` | ~1000 | 97.4% |

**Practical guidance:**
- Default `6e-4` is a good starting point — short warm-up, minimal training data waste.
- If you have ≥ 2 years of H1 data (~3,300 bars per year), `3e-4` (200-bar warm-up) is essentially free.
- `1e-4` or lower wastes ~500 bars of training data per split — only justified when you have multi-year datasets and need maximum memory retention.
- The marginal gain from `3e-4` → `4.3e-5` is only 4.5% more information, rarely worth the 5× longer warm-up.

---

## Scaling: rolling window size

`RollingScaler(window_size=200)` is recommended over the default 500.

### Why 200 for H1/M15

| Window | H1 real time | Warm-up % of 3,315-bar train |
|--------|-------------|------------------------------|
| 500 (default) | ~21 trading days | ~15% wasted on partial stats |
| **200** | **~8 trading days** | **~6% wasted** |
| 100 | ~4 trading days | ~3%, but z-score stats become noisy |

- 200 bars captures enough local volatility regime (a couple of weeks of H1) to produce stable IQR estimates.
- Halves the warm-up penalty vs. 500 with negligible quality loss.
- Going below 100 risks unstable z-score statistics during regime transitions.

### Global vs rolling

| Scaler | When to use |
|--------|-------------|
| `ForexScaler` (global) | Short backtests, homogeneous volatility regime, fast iteration |
| `RollingScaler` | Long backtests spanning multiple volatility regimes (recommended) |

Global scaling bakes in train-set volatility, creating distribution shift on val/test bars recorded during quiet or explosive regimes. Rolling scaling adapts to local conditions.

---

## Weekend handling

| Mode | Data | Chronos compat | Notes |
|------|------|----------------|-------|
| `"nogap"` (default) | Mon–Fri only, no NaN | No — gaps in context | Compact charts; bad for sequence models |
| `"gaps"` | 7-day grid with NaN weekends | No — NaN breaks context | Shows real time passage |
| **`"filled"`** | **7-day grid, Friday close forward-filled** | **Yes** | Best for Chronos and any transformer needing continuous context |

`weekends="filled"` is applied at **two points** in the pipeline:
1. At load time in `ForexDataLoader.load_and_merge()` — extends raw M1 data
2. At resample time in `ForexPipeline.run()` — re-extends at target timeframe

This keeps a fully 7-day aligned grid at every timeframe. Chronos was pre-trained expecting continuous time series, so `filled` avoids silent NaN propagation into the 8K context window.

---

## Target types

| `target_type` | Column | Notes |
|--------------|--------|-------|
| `"lag"` | `direction_1`, `direction_5`, `direction_15` | Binary up/down N bars ahead |
| `"triple_barrier"` | `tb_label` (-1 / 0 / 1) | More realistic; remapped to binary for XGBoost: -1→0, 0→0, 1→1 |

When `target_type="triple_barrier"`, always pass `target_col="tb_label"` to `get_xy()` — not `"direction_1"`.

---

## Walk-forward split

Default: `train=60%, gap=50, val=20%, gap=50, test=20%`

The gap prevents label leakage from autocorrelated targets (e.g. `direction_15` looks 15 bars ahead, so train and val must be ≥ 15 bars apart). 50-bar default is conservative and safe for all horizons.

---

## Pipeline quick-reference

All fracdiff parameters are first-class `ForexPipeline` constructor args:

```python
pipeline = ForexPipeline(
    norm_method="fracdiff",        # "log_returns" | "fracdiff" | "raw"
    fracdiff_d=0.17,               # only used when norm_method="fracdiff"
    threshold=3e-4,                # fracdiff weight cutoff — ~200 warm-up bars, 92.9% info
    target_type="triple_barrier",
    scaling="rolling",             # "global" | "rolling"
    window_size=200,               # only used when scaling="rolling"
    lags=[1, 2, 5, 10],
    weekends="filled",             # "nogap" | "gaps" | "filled"
)
results = pipeline.run(df_m1, timeframe="H1")
X_train, y_train = pipeline.get_xy(results["train"], "tb_label", results["feature_cols"])
```
