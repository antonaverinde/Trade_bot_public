"""
TimesFM 2.5 zero-shot forex forecaster.

Uses google/timesfm-2.5-200m-pytorch (200M params).
Outputs 10 levels: mean + 9 decile quantiles (q0.1 … q0.9).

Output quantile layout (last dim of quantile_matrix):
    index 0  → mean (arithmetic)
    index 1  → q0.1  (10th percentile)
    ...
    index 5  → q0.5  (median — also returned as point_forecast)
    ...
    index 9  → q0.9  (90th percentile)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_DIR = Path(__file__).parent / "model"

# 10 levels: mean + decile quantiles
# Key 0.0 encodes the mean (arithmetic); 0.1–0.9 are true percentiles
MODEL_QUANTILES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class TimesFMForecaster:
    """
    Zero-shot TimesFM 2.5 forecaster for raw forex price series.

    Parameters
    ----------
    model_name     : HuggingFace id — "google/timesfm-2.5-200m-pytorch"
    context_length : max bars fed to the model (up to 16 384; default 512)
    device         : "auto" | "gpu" | "cpu"  (TimesFM uses "gpu"/"cpu", not "cuda")
    cache_dir      : where to store weights — defaults to Timeseriesfm/model/
    """

    def __init__(
        self,
        model_name: str = "google/timesfm-2.5-200m-pytorch",
        context_length: int = 512,
        device: str = "auto",
        cache_dir: str | None = None,
    ):
        self.model_name     = model_name
        self.context_length = context_length
        self.device         = device
        self.cache_dir      = cache_dir or str(MODEL_DIR)
        self._model         = None

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self) -> "TimesFMForecaster":
        import timesfm
        from huggingface_hub import hf_hub_download

        backend = self._resolve_backend()
        print(f"[TimesFM] Loading {self.model_name}")
        print(f"          cache  → {self.cache_dir}")
        print(f"          device → {backend}")

        # Instantiate directly to avoid huggingface_hub mixin passing proxies kwarg
        # through model_kwargs into __init__ (a known bug in timesfm 2.0.0).
        wrapper = timesfm.TimesFM_2p5_200M_torch(torch_compile=False)

        weights_path = hf_hub_download(
            repo_id  = self.model_name,
            filename = "model.safetensors",
            cache_dir= self.cache_dir,
        )
        wrapper.model.load_checkpoint(weights_path, torch_compile=False)

        self._model = wrapper
        self._model.compile(timesfm.ForecastConfig(
            max_context           = self.context_length,
            max_horizon           = self.context_length,
            normalize_inputs      = False,
            force_flip_invariance = True,
            infer_is_positive     = True,
            fix_quantile_crossing = True,
        ))

        print(f"[TimesFM] Ready — ctx={self.context_length}  "
              f"quantiles: {len(MODEL_QUANTILES)} levels "
              f"({MODEL_QUANTILES[0]} … {MODEL_QUANTILES[-1]})")
        return self

    def _resolve_backend(self) -> str:
        if self.device == "auto":
            try:
                import torch
                return "gpu" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        if self.device in ("cuda", "gpu"):
            return "gpu"
        return "cpu"

    def _ensure_loaded(self):
        if self._model is None:
            self.load()

    # ── Core forecast ─────────────────────────────────────────────────────────

    def forecast(
        self,
        close: "np.ndarray | pd.Series",
        prediction_length: int = 96,
    ) -> dict:
        """
        Run TimesFM 2.5 on a 1-D raw close-price series.

        Returns
        -------
        dict:
            context          np.ndarray (up to context_length,)
            quantile_matrix  np.ndarray (10, pred_len)  — [mean, q10, …, q90]
            model_quantiles  list[float]  — MODEL_QUANTILES (len 10)
            quantiles        dict {float: np.ndarray (pred_len,)}
            median           np.ndarray (pred_len,)  — alias for quantiles[0.5]
        """
        self._ensure_loaded()

        arr = close.values if isinstance(close, pd.Series) else np.asarray(close, dtype=np.float32)
        ctx = arr[-self.context_length:]

        # forecast() pads/slices to max_context internally
        _, q_full = self._model.forecast(
            horizon=prediction_length,
            inputs=[ctx],
        )
        # q_full: (1, pred_len, 10)  → squeeze batch dim → (pred_len, 10)
        q_seq = q_full[0]                        # (pred_len, 10)
        q_matrix = q_seq.T                       # (10, pred_len)

        qs = {q: q_matrix[i] for i, q in enumerate(MODEL_QUANTILES)}

        return {
            "context":         ctx,
            "quantile_matrix": q_matrix,
            "model_quantiles": MODEL_QUANTILES,
            "quantiles":       qs,
            "median":          qs[0.5],
        }

    # ── CDF helper ────────────────────────────────────────────────────────────

    @staticmethod
    def prob_above(fc: dict, threshold: float) -> np.ndarray:
        """
        P(price_t > threshold) at each forecast step via CDF interpolation.

        Uses only the true percentile levels (0.1–0.9), skipping the mean (0.0).

        Returns np.ndarray (pred_len,) in [0, 1].
        """
        q_matrix   = fc["quantile_matrix"]       # (10, pred_len)
        # Use indices 1..9 → true quantiles 0.1..0.9
        q_sub      = q_matrix[1:]                # (9, pred_len)
        qs_arr     = np.array(MODEL_QUANTILES[1:])
        pred_len   = q_matrix.shape[1]

        probs = np.empty(pred_len)
        for t in range(pred_len):
            cdf      = np.interp(threshold, q_sub[:, t], qs_arr)
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

        price_df must have a DatetimeIndex and columns [open, high, low, close].

        Parameters
        ----------
        price_df         : raw OHLCV at the desired timeframe (e.g. H1)
        context_end      : datetime string or Timestamp — model receives
                           context_length bars BEFORE this point and predicts
                           prediction_length bars AFTER it.
        prediction_length: number of bars to predict

        Returns
        -------
        dict with keys matching ChronosForecaster.forecast_from_df() output.
        """
        requested  = pd.Timestamp(context_end)
        idx        = price_df.index.get_indexer([requested], method="pad")[0]
        cutoff_ts  = price_df.index[idx]

        history    = price_df.loc[:cutoff_ts]
        close_hist = history["close"]

        fc         = self.forecast(close_hist, prediction_length=prediction_length)
        context_df = history.iloc[-self.context_length:]

        after           = price_df.loc[cutoff_ts:].iloc[1 : prediction_length + 1]
        ground_truth_df = after if len(after) > 0 else None

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
            "context_df":          context_df,
            "forecast":            fc,
            "ground_truth_df":     ground_truth_df,
            "forecast_timestamps": forecast_timestamps,
            "context_end":         cutoff_ts,
            "forecast_start":      forecast_timestamps[0],
            "prediction_length":   prediction_length,
            "pair":                price_df.attrs.get("pair", ""),
        }

    # ── Pipeline wrapper ──────────────────────────────────────────────────────

    def forecast_from_pipeline(
        self,
        results: dict,
        prediction_length: int = 96,
        context_end: str = "train_end",
    ) -> dict:
        """
        Convenience wrapper around forecast_from_df for ForexPipeline output.

        context_end options:
            "train_end" / "val_end" / "test_end"  — split boundary
            "2023-06-01 16:00"                    — any ISO datetime
        """
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
