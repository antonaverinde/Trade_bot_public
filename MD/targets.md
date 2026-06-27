# Target Design Suggestions for Selective Trading Models

The current binary setup asks the model to classify every decisive triple-barrier event as short or long. That does not match the desired trading behavior: make fewer predictions, but make them more accurately. The target design should explicitly support no-trade/abstention and should reward clean, high-quality setups.

## Current Problem

Current binary target flow:

- `tb_label = -1` -> short
- `tb_label = 0` -> dropped neutral
- `tb_label = 1` -> long

This creates a forced short-vs-long classifier on all decisive events. Many decisive events are still weak or ambiguous. The model learns a low-confidence boundary, so probabilities stay near `0.5`, calibration curves have only a few populated bins, and high thresholds like `0.8-0.9` select no useful samples.

The objective should move from "predict every bar" to "predict only when there is edge."

## Target Option 1: Three-Class Trade Decision

Create a direct trade-decision target:

- `0 = short`
- `1 = no_trade`
- `2 = long`

Suggested label logic:

- long if forward return is greater than `+min_edge`
- short if forward return is lower than `-min_edge`
- no_trade otherwise

`min_edge` should include spread/slippage plus a safety margin. For example, if the model predicts H1 bars, `min_edge` could be expressed in pips, raw return, or volatility units.

Why this helps:

- model is allowed to reject ambiguous bars
- no-trade becomes an explicit learned state
- inference can trade only when `P(long)` or `P(short)` is high enough

First implementation target name:

- `trade_label_h{horizon}`

Config idea:

```python
"target_type": "trade_threshold",
"trade_horizon_bars": 10,
"min_edge": 0.0005,
"cost_buffer": 0.0001,
```

## Target Option 2: Volatility-Normalized Trade Decision

Instead of fixed return thresholds, label by volatility-adjusted return:

```python
future_ret = log(close[t + h] / close[t])
vol = rolling_or_ewm_vol[t]
z_ret = future_ret / vol
```

Labels:

- long if `z_ret >= z_entry`
- short if `z_ret <= -z_entry`
- no_trade otherwise

Why this helps:

- adapts to changing volatility regimes
- avoids using the same pip threshold in quiet and volatile markets
- should produce cleaner labels than raw directional movement

First implementation target name:

- `trade_z_label_h{horizon}`

Config idea:

```python
"target_type": "trade_zscore",
"trade_horizon_bars": 10,
"z_entry": 1.0,
"vol_window": 100,
```

## Target Option 3: Quality-Weighted Triple Barrier

Keep triple-barrier labels, but add sample weights so stronger events matter more.

Possible weights:

- larger absolute `tb_ret` -> larger weight
- faster barrier hit -> larger weight
- timeout/noisy event -> lower weight
- symmetrical cap to avoid a few extreme rows dominating

Example weight:

```python
speed_weight = horizon_bars / max(1, bars_to_hit)
return_weight = abs(tb_ret) / rolling_vol
sample_weight = clip(speed_weight * return_weight, 0.5, 5.0)
```

Why this helps:

- model focuses on clean setups
- weak labels still exist but matter less
- can be used with binary or three-class models

Needed extra columns:

- `tb_bars_to_hit`
- `tb_exit_type`
- `tb_quality_weight`

## Target Option 4: Separate Long and Short One-vs-Rest Models

Train two independent models:

Long model target:

- `1 = good long setup`
- `0 = not a good long setup`

Short model target:

- `1 = good short setup`
- `0 = not a good short setup`

Inference:

- trade long if long model confidence passes threshold and short model does not
- trade short if short model confidence passes threshold and long model does not
- skip if neither or both are confident

Why this helps:

- long and short regimes may not be symmetric
- avoids forcing one boundary to explain both sides
- easier to tune separate thresholds for long and short

This is especially relevant because current tests found long confidence pockets, while short pockets were less stable.

## Target Option 5: Return or Edge Regression

Train regression instead of classification:

- predict forward return
- predict volatility-normalized forward return
- predict expected trade payoff after cost

Inference:

- long if predicted edge > threshold
- short if predicted edge < -threshold
- otherwise no trade

Why this helps:

- model learns magnitude, not just sign
- thresholds are natural for abstention
- calibration can be evaluated as prediction error and realized return buckets

Potential target names:

- `future_ret_h{horizon}`
- `future_zret_h{horizon}`
- `expected_payoff_h{horizon}`

## Evaluation Rules

Do not evaluate only full-set accuracy. Every target experiment should report:

- full split metrics: AUC, logloss, Brier, balanced accuracy, MCC
- threshold sweep for long, short, and either side
- selected coverage
- selected accuracy
- selected precision for long and short
- confusion matrix on selected predictions
- actual return/payoff of selected predictions if available
- train/val/test gap
- calibration curve and probability histogram

A target is only useful if:

- validation and test both show selected accuracy above random
- selected coverage is non-zero and stable
- train is not much better than val/test
- long and short thresholds can be chosen independently

## Suggested Experiment Order

1. Implement `trade_threshold` three-class labels.
2. Add threshold sweeps for `P(short)` and `P(long)` with no-trade allowed.
3. Implement `trade_zscore` labels if fixed thresholds are unstable.
4. Add quality weights to triple-barrier labels.
5. Try separate long and short one-vs-rest models.
6. Try return/edge regression if classification still produces compressed probabilities.

## Recommended First Implementation

Start with a three-class volatility-normalized trade target:

- `short = z_ret <= -1.0`
- `no_trade = -1.0 < z_ret < 1.0`
- `long = z_ret >= 1.0`

Then train multiclass XGBoost and trade only when:

- `P(long) >= threshold_long`, or
- `P(short) >= threshold_short`

Keep thresholds low at first, for example `0.50-0.70`, and let validation/test threshold sweeps determine whether higher thresholds are realistic.
