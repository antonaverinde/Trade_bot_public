"""
Chronos-2 zero-shot forex forecaster.

Uses amazon/chronos-2 (120M params, ~480 MB bfloat16).
Chronos-2 outputs 21 quantiles directly — no Monte Carlo sampling.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_DIR = Path(__file__).parent / "model"

# 21 quantile levels built into the model weights — cannot be changed
MODEL_QUANTILES = [
    0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99,
]


class ChronosForecaster:
    """
    Zero-shot Chronos-2 forecaster for raw forex price series.

    Parameters
    ----------
    model_name     : HuggingFace id — "amazon/chronos-2"
    device         : "auto" | "cuda" | "cpu"
    dtype          : torch.bfloat16 (default) or torch.float32
    context_length : bars fed to the model as history (max 8192)
    cache_dir      : where to store weights — defaults to Chronos/model/
    """

    def __init__(
        self,
        model_name: str = "amazon/chronos-2",
        device: str = "auto",
        dtype=torch.bfloat16,
        context_length: int = 1024,
        cache_dir: str | None = None,
    ):
        self.model_name     = model_name
        self.device         = device
        self.dtype          = dtype
        self.context_length = context_length
        self.cache_dir      = cache_dir or str(MODEL_DIR)
        self._pipeline      = None
        self._model_quantiles: list[float] | None = None

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self) -> "ChronosForecaster":
        from chronos import Chronos2Pipeline

        print(f"[Chronos] Loading {self.model_name}")
        print(f"          cache  → {self.cache_dir}")
        self._pipeline = Chronos2Pipeline.from_pretrained(
            self.model_name,
            device_map=self.device,
            dtype=self.dtype,
            cache_dir=self.cache_dir,
        )
        self._model_quantiles = self._pipeline.quantiles
        device_name = next(iter(self._pipeline.model.parameters())).device
        print(f"[Chronos] Ready on {device_name}")
        print(f"          quantiles : {len(self._model_quantiles)} levels  "
              f"({self._model_quantiles[0]} … {self._model_quantiles[-1]})")
        print(f"          ctx window: {self._pipeline.model_context_length}")
        return self

    def _ensure_loaded(self):
        if self._pipeline is None:
            self.load()

    # ── Core forecast ─────────────────────────────────────────────────────────

    def forecast(
        self,
        close: "np.ndarray | pd.Series",
        prediction_length: int = 96,
    ) -> dict:
        """
        Run Chronos-2 on a 1-D raw close-price series.

        Returns
        -------
        dict:
            context          np.ndarray (context_length,)
            quantile_matrix  np.ndarray (21, pred_len)
            model_quantiles  list[float]
            quantiles        dict {float: np.ndarray (pred_len,)}
            median           np.ndarray (pred_len,)
        """
        self._ensure_loaded()

        arr = close.values if isinstance(close, pd.Series) else np.asarray(close, dtype=np.float32)
        ctx = arr[-self.context_length:]

        raw = self._pipeline.predict(
            inputs=[ctx],
            prediction_length=prediction_length,
            context_length=self.context_length,
        )

        # raw[0]: (n_variates=1, n_quantiles=21, pred_len) → squeeze to (21, pred_len)
        q_matrix = raw[0].squeeze(0).numpy()
        qs = {q: q_matrix[i] for i, q in enumerate(self._model_quantiles)}

        return {
            "context":         ctx,
            "quantile_matrix": q_matrix,
            "model_quantiles": self._model_quantiles,
            "quantiles":       qs,
            "median":          qs[0.50],
        }

    # ── CDF helper ────────────────────────────────────────────────────────────

    @staticmethod
    def prob_above(fc: dict, threshold: float) -> np.ndarray:
        """
        P(price_t > threshold) at each forecast step via CDF interpolation.

        Returns np.ndarray (pred_len,) in [0, 1].
        """
        q_matrix = fc["quantile_matrix"]       # (21, pred_len)
        qs_arr   = np.array(fc["model_quantiles"])
        pred_len = q_matrix.shape[1]

        probs = np.empty(pred_len)
        for t in range(pred_len):
            cdf = np.interp(threshold, q_matrix[:, t], qs_arr)
            probs[t] = 1.0 - cdf

        return np.clip(probs, 0.0, 1.0)

    # ── Primary API: raw DataFrame ────────────────────────────────────────────

    def forecast_from_df(
        self,
        price_df: pd.DataFrame,
        context_end: "str | pd.Timestamp",
        prediction_length: int = 96,
    ) -> dict:
        """
        Forecast directly from a raw OHLCV DataFrame.

        This is the primary entry point — no pipeline needed.
        price_df must have a DatetimeIndex and columns [open, high, low, close].

        Parameters
        ----------
        price_df         : raw OHLCV at the desired timeframe (e.g. H1)
        context_end      : datetime string (e.g. "2023-06-01 16:00") or Timestamp —
                           the model receives context_length bars BEFORE this point
                           and predicts prediction_length bars AFTER it.
                           Bars after this that exist in price_df are shown as ground truth.
        prediction_length: number of bars to predict

        Returns
        -------
        dict:
            context_df          raw OHLCV DataFrame (last context_length bars of history)
            forecast            output of self.forecast()
            ground_truth_df     actual OHLCV for forecast window (or None)
            forecast_timestamps pd.DatetimeIndex
            context_end         pd.Timestamp (snapped to nearest available bar)
            forecast_start      pd.Timestamp
            prediction_length   int
            pair                str (from price_df.attrs if available)
        """
        # Snap context_end to nearest available bar
        requested = pd.Timestamp(context_end)
        idx       = price_df.index.get_indexer([requested], method="pad")[0]
        cutoff_ts = price_df.index[idx]

        history    = price_df.loc[:cutoff_ts]
        close_hist = history["close"]

        fc         = self.forecast(close_hist, prediction_length=prediction_length)
        context_df = history.iloc[-self.context_length:]

        # Ground truth: bars after cutoff that exist in the data
        after           = price_df.loc[cutoff_ts:].iloc[1 : prediction_length + 1]
        ground_truth_df = after if len(after) > 0 else None

        # Forecast timestamps
        bar_td = (
            context_df.index[-1] - context_df.index[-2]
            if len(context_df) >= 2
            else pd.Timedelta(hours=1)
        )
        forecast_timestamps = pd.date_range(
            start=cutoff_ts + bar_td,
            periods=prediction_length,
            freq=bar_td,
        )

        return {
            "context_df":           context_df,
            "forecast":             fc,
            "ground_truth_df":      ground_truth_df,
            "forecast_timestamps":  forecast_timestamps,
            "context_end":          cutoff_ts,
            "forecast_start":       forecast_timestamps[0],
            "prediction_length":    prediction_length,
            "pair":                 price_df.attrs.get("pair", ""),
        }

    # ── Pipeline wrapper (for XGBoost feature integration) ───────────────────

    def forecast_from_pipeline(
        self,
        results: dict,
        prediction_length: int = 96,
        context_end: str = "train_end",
    ) -> dict:
        """
        Convenience wrapper around forecast_from_df for use with ForexPipeline output.

        results["raw_m1"] already contains raw OHLCV at the target timeframe.

        context_end options:
            "train_end"          last bar of the training split
            "val_end"            last bar of the validation split
            "test_end"           last bar of the test split (future — no ground truth)
            "2023-06-01 16:00"   any ISO datetime in the dataset
        """
        # raw_m1 is already at the target timeframe — use it directly
        price_df = results["raw_m1"].copy()
        price_df.attrs["pair"] = results.get("pair", "")

        _split_ends = {
            "train_end": results["train_raw"].index[-1],
            "val_end":   results["val_raw"].index[-1],
            "test_end":  results["test_raw"].index[-1],
        }
        cutoff = _split_ends.get(context_end, context_end)

        result = self.forecast_from_df(price_df, context_end=cutoff,
                                       prediction_length=prediction_length)
        result["timeframe"] = results.get("timeframe", "")
        return result
