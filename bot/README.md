# Backtest Optimization Bot

Runs a cost-aware backtest and parameter search for the existing XGBoost models.

```bash
source /home/anton/Trade_bot/.venv/bin/activate
python -m bot.optimize --model both --trials 100
```

Useful options:

```bash
python -m bot.optimize --model h1 --trials 200
python -m bot.optimize --model fvg --trials 200
python -m bot.optimize --model both --level-mode model --trials 500
python -m bot.optimize --model both --spread-pips 1.2 --slippage-pips 0.3
python -m bot.optimize --model h1 --h1-run-id <mlflow-run-id>
python -m bot.optimize --model fvg --fvg-run-id <mlflow-run-id>
```

Outputs are written to `outputs/bot_runs/<timestamp>/<model>/`:

- `trials.csv`: all searched decision/risk parameters and validation metrics.
- `trades.csv`: final test-set trades using the best validation parameters.
- `equity_curve.csv`: final test-set equity and drawdown over time.
- `summary.json`: selected run IDs, best parameters, validation metrics, and test metrics.
- `equity_curve.png`, `drawdown.png`: report charts.

V1 is simulation only. It loads the latest successful MLflow runs by default, compares
the H1 and FVG/fractal models separately, enforces one open position total, and includes
spread, slippage, and commission in every closed trade.

`--level-mode model` uses model-specific levels:

- H1 uses an executable raw-price volatility barrier based on the model's `k_up/k_down`.
- FVG uses known prior fractal levels at decision time; rows without usable model levels are skipped.

`--level-mode atr` uses generic ATR-multiple stop/take distances.
