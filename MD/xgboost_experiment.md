# XGBoost Forex Experiment Notebook

`xgboost_experiment.ipynb` — trains an XGBoost binary classifier on forex OHLCV data prepared by
ForexPipeline and tracks every detail of the run in MLflow (SQLite backend stored in the project folder).

---

## Running the Notebook

```bash
source ~/Trade_bot/.venv/bin/activate
cd ~/Trade_bot
jupyter notebook xgboost_experiment.ipynb
```

Run cells top-to-bottom. The notebook is fully re-runnable; each run creates a new MLflow entry.

### Browsing Results in MLflow UI

```bash
cd ~/Trade_bot
source /home/anton/Trade_bot/.venv/bin/activate
mlflow ui --backend-store-uri sqlite:///mlflow///mlflow.db --host 0.0.0.0 --port 5000
```

The SQLite file `mlflow.db` is created automatically in the project root on the first run.

---

## Configuration

All parameters live in two dicts in **Cell 2**. Edit them there; everything downstream reads from these.

### `DATA_CFG` — Pipeline Parameters

| Key | Default | Description |
|---|---|---|
| `pair` | `"EURUSD"` | Currency pair. Must match a CSV in `histdata/` |
| `years` | `[2022, 2023]` | Which annual CSVs to load and merge |
| `timeframe` | `"M15"` | Resample target: `M1 M5 M15 H1 H4 D1` |
| `norm_method` | `"log_returns"` | Price normalisation: `log_returns` \| `fracdiff` \| `raw` |
| `target_type` | `"lag"` | Label generation: `lag` (directional) \| `triple_barrier` |
| `target_col` | `"direction_1"` | Column used as the training label (see below) |
| `lags` | `[1,2,5,10]` | Close-lag feature shifts (bars) |
| `target_horizons` | `[1,5,15]` | Future-return horizons for lag targets |
| `gap_bars` | `50` | Bars dropped between train/val and val/test to prevent leakage |
| `scaling` | `"global"` | Feature scaling: `global` (ForexScaler) \| `rolling` (RollingScaler) |
| `fracdiff_d` | `0.4` | Fractional differencing order (only used when `norm_method=fracdiff`) |
| `k_up` / `k_down` | `2.0` / `1.0` | Barrier multipliers for triple-barrier labels |
| `horizon_bars` | `10` | Time barrier length for triple-barrier mode |

#### `target_col` options

With `target_type="lag"`: `direction_1`, `direction_5`, `direction_15`
— binary 0/1 indicating whether price goes up over the next 1/5/15 bars.

With `target_type="triple_barrier"`: `tb_label`
— values −1 (stop loss hit), 0 (time out), 1 (take profit hit).
The notebook remaps this to binary (1 → long signal, −1/0 → not-long).

### `XGB_PARAMS` — Model Hyperparameters

| Key | Default | Description |
|---|---|---|
| `n_estimators` | `300` | Max boosting rounds (actual may be fewer with early stopping) |
| `max_depth` | `5` | Maximum tree depth. Higher = more complex, more overfit risk |
| `learning_rate` | `0.05` | Shrinkage per round. Lower = more rounds needed, better generalisation |
| `subsample` | `0.8` | Row sampling ratio per tree (0–1) |
| `colsample_bytree` | `0.8` | Feature sampling ratio per tree (0–1) |
| `min_child_weight` | `5` | Minimum sum of instance weights in a leaf. Regularises on sparse splits |
| `gamma` | `0.1` | Minimum loss reduction required for a further partition |
| `reg_alpha` | `0.1` | L1 weight regularisation |
| `reg_lambda` | `1.0` | L2 weight regularisation |
| `early_stopping_rounds` | `20` | Stop if val logloss does not improve for 20 consecutive rounds |
| `objective` | `binary:logistic` | Loss function. Use `multi:softprob` for triple-barrier multi-class |
| `eval_metric` | `logloss` | Metric monitored for early stopping |
| `random_state` | `42` | Seed for reproducibility |

---

## Pipeline Stages

The notebook calls `ForexPipeline.run()` which executes five stages:

1. **Load** — `ForexDataLoader.load_and_merge` reads HistData M1 CSVs, deduplicates, forward-fills thin-liquidity gaps, drops weekends.
2. **Resample** — resamples from M1 to `timeframe` using OHLC aggregation.
3. **Features** — computes 32 features (see below) from normalised OHLCV.
4. **Split** — walk-forward split: 60 % train → gap → 20 % val → gap → 20 % test.
5. **Scale** — fits ForexScaler (median/IQR) on train, applies to all three splits.

### Feature Set (32 features)

| Group | Features |
|---|---|
| RSI | `rsi_14`, `rsi_21`, speed, acceleration, cross-50/70 signals |
| ADX | `adx_14`, `di_diff` (+DI−−DI), `adx_delta` (trend momentum) |
| Trend/Vol | `dist_ema200`, `atr_rel`, `bb_pct_b` (Bollinger %B) |
| Time | `hour_sin/cos`, `dow_sin/cos`, `is_monday`, `is_friday` |
| Candle | `body_ratio`, `shadow_ratio`, `body_gap` |
| Distribution | `ret_skew`, `ret_kurt`, `vol_ewma` |
| Lags | `close_lag1/2/5/10` (normalised close shifted N bars) |

---

## MLflow Logging

Everything logged under `experiment = xgboost_forex`, one run per notebook execution.

### Parameters logged

- All `DATA_CFG` values (prefixed `data_`)
- All `XGB_PARAMS` values (prefixed `xgb_`)
- Date ranges: `raw_m1_start/end`, `train/val/test_start/end`
- Sample counts and class-balance per split
- `best_iteration` (from early stopping)
- `n_features`

### Metrics logged

Per split (prefix `train_`, `val_`, `test_`):

| Metric | Why it matters |
|---|---|
| `auc` | Area under ROC — rank-ordering quality, threshold-independent |
| `avg_precision` | Area under PR curve — better than AUC when classes are imbalanced |
| `logloss` | Calibrated probability quality |
| `brier` | Mean squared probability error — penalises overconfident predictions |
| `f1` | Harmonic mean of precision and recall at the default threshold (0.5) |
| `precision` | Of all predicted longs, how many were correct |
| `recall` | Of all actual longs, how many were captured |
| `balanced_acc` | Average of per-class recall — robust to class imbalance |
| `mcc` | Matthews Correlation Coefficient — single score for binary classifiers |
| `overfit_logloss_gap` | Val minus train logloss at best iteration — quantifies overfitting |

### Artifacts logged

| Path | Content |
|---|---|
| `features.json` | Ordered list of feature names used in training |
| `plots/roc_curve.png` | ROC curves for all three splits on one axes |
| `plots/pr_curve.png` | Precision-Recall curves for val and test |
| `plots/confusion_matrices.png` | Heatmaps with counts and row-normalised % |
| `plots/learning_curves.png` | Logloss vs boosting round, overfitting region shaded |
| `plots/feature_importance.png` | Top-20 features by Gain, Weight, Cover |
| `plots/calibration_curve.png` | Reliability diagram — predicted probability vs actual frequency |
| `plots/metrics_comparison.png` | Bar charts comparing all metrics across splits |
| `reports/classification_report.csv` | Per-class precision/recall/F1 for val and test |
| `reports/feature_importance.csv` | All features ranked by all three importance types |
| `model/` | Serialised XGBoost model (loadable with `mlflow.xgboost.load_model`) |

---

## Overfitting Analysis Guide

### Learning Curves (Cell 9)
The most direct overfitting diagnostic. If the **val logloss curve turns upward while train continues downward**, the model is memorising. The **shaded red region** shows rounds after best_iteration — trees added there overfit. If there is no divergence, the model generalises well to unseen validation data.

### ROC / AUC Gap (Cell 5)
Compare train AUC vs test AUC. A gap > 0.05 is a warning sign. A gap > 0.10 suggests the model has learned noise or structural differences between train and test periods.

### Metrics Comparison (Cell 12)
Look at F1, precision, recall, balanced_accuracy across all three splits simultaneously. Sharp drops from train → val → test indicate over-specialisation to the training period.

### Calibration Curve (Cell 11)
A well-calibrated model follows the diagonal closely. Overfit models often show S-shaped or extreme curves because they push probabilities toward 0 and 1 with false confidence. Brier score penalises this directly.

### MCC (Matthews Correlation Coefficient)
Unlike accuracy, MCC accounts for all four cells of the confusion matrix. Values near 0 mean the model is no better than random; values above 0.15 on financial classification are considered meaningful.

---

## Reloading a Trained Model

```python
import mlflow.xgboost

RUN_ID = "paste-run-id-here"
model  = mlflow.xgboost.load_model(f"runs:/{RUN_ID}/model")
preds  = model.predict(X_test)
probas = model.predict_proba(X_test)[:, 1]
```
