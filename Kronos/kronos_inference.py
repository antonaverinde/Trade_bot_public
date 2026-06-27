import sys
import math
from pathlib import Path
from datetime import timedelta
from typing import Sequence

import numpy as np
import pandas as pd
import torch

KRONOS_REPO = Path(__file__).parent / "kronos_repo"
MODEL_DIR   = Path(__file__).parent / "model"
OHLC_COLS   = ["open", "high", "low", "close"]
MAX_CONTEXT = 512

_VRAM_PRACTICAL_CAP = 64   # never auto-set above this (prevents impractically large batches)
_VRAM_SAFETY_MB     = 1000 # headroom buffer for OS / driver overhead


def _make_future_timestamps(cutoff_ts: pd.Timestamp, bar_td: timedelta, pred_len: int) -> pd.DatetimeIndex:
    """Generate pred_len future timestamps, skipping Sat/Sun for intraday bars."""
    if bar_td >= timedelta(days=1):
        return pd.date_range(start=cutoff_ts + bar_td, periods=pred_len, freq=bar_td)
    ts = cutoff_ts
    result = []
    while len(result) < pred_len:
        ts = ts + bar_td
        if ts.weekday() < 5:
            result.append(ts)
    return pd.DatetimeIndex(result)


class KronosForecaster:
    """Wrapper around KronosPredictor analogous to ChronosForecaster.

    Weights auto-download from HuggingFace on first load() call.
    Model cached in Kronos/model/ (gitignored).

    Parallel sampling: n_samples paths run in GPU-parallel batches of max_parallel.
    max_parallel is auto-detected from VRAM on load(); override via constructor.
    """

    def __init__(
        self,
        model_name:     str = "NeoQuasar/Kronos-base",
        tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base",
        device:         str = "auto",
        max_context:    int = 512,
        cache_dir:      str | None = None,
        max_parallel:   int = 0,   # 0 = auto-detect from VRAM after load()
    ) -> None:
        if max_context > MAX_CONTEXT:
            print(f"[KronosForecaster] max_context clamped {max_context} → {MAX_CONTEXT}")
            max_context = MAX_CONTEXT
        self.model_name            = model_name
        self.tokenizer_name        = tokenizer_name
        self.device                = device
        self.max_context           = max_context
        self.cache_dir             = Path(cache_dir) if cache_dir else MODEL_DIR
        self._max_parallel_override = max_parallel
        self._max_parallel          = 1
        self._predictor             = None

    # ── load ──────────────────────────────────────────────────────────────────

    def load(self) -> "KronosForecaster":
        """Load model + tokenizer from HuggingFace, cache locally. Auto-benchmarks VRAM."""
        repo = str(KRONOS_REPO)
        if repo not in sys.path:
            sys.path.insert(0, repo)

        from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: PLC0415

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache = str(self.cache_dir)

        # KronosPredictor does not accept "auto" — resolve to concrete device string
        device = self.device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        print(f"[KronosForecaster] Loading tokenizer {self.tokenizer_name} ...")
        tokenizer = KronosTokenizer.from_pretrained(self.tokenizer_name, cache_dir=cache)

        print(f"[KronosForecaster] Loading model {self.model_name} ...")
        model = Kronos.from_pretrained(self.model_name, cache_dir=cache)

        self._predictor = KronosPredictor(
            model, tokenizer, device=device, max_context=self.max_context
        )

        if self._max_parallel_override > 0:
            self._max_parallel = self._max_parallel_override
        else:
            self._max_parallel = self._benchmark_max_parallel()

        dev = self._predictor.device
        print(f"[KronosForecaster] Ready on {dev}  context={self.max_context}  max_parallel={self._max_parallel}")
        return self

    def _ensure_loaded(self) -> None:
        if self._predictor is None:
            self.load()

    # ── VRAM benchmark ────────────────────────────────────────────────────────

    def _benchmark_max_parallel(self) -> int:
        """Measure per-sample VRAM cost and compute max safe parallel batch size.

        Uses a short dummy inference (pred_len=5, ctx=max_context) to measure
        peak memory at batch=1 and batch=2.  Result is capped at _VRAM_PRACTICAL_CAP.
        Falls back to 1 on CPU/MPS.
        """
        p = self._predictor
        if "cuda" not in str(p.device):
            return 1

        dev_id = int(str(p.device).split(":")[-1]) if ":" in str(p.device) else 0

        from model.kronos import auto_regressive_inference  # noqa: PLC0415

        ctx      = self.max_context
        pred_len = 5  # short dummy — VRAM dominated by context buffers, not pred_len

        def _peak_mb(batch: int) -> float:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev_id)
            torch.cuda.synchronize(dev_id)
            dx  = torch.zeros(batch, ctx, 6, device=p.device)
            dxs = torch.zeros(batch, ctx, 5, device=p.device)
            dys = torch.zeros(batch, pred_len, 5, device=p.device)
            try:
                auto_regressive_inference(
                    p.tokenizer, p.model, dx, dxs, dys,
                    ctx, pred_len, p.clip, 1.0, 0, 0.9, 1, False,
                )
            except RuntimeError:
                return float("inf")
            torch.cuda.synchronize(dev_id)
            return torch.cuda.max_memory_allocated(dev_id) / 1e6

        alloc_mb   = torch.cuda.memory_allocated(dev_id) / 1e6
        total_mb   = torch.cuda.get_device_properties(dev_id).total_memory / 1e6
        peak1      = _peak_mb(1)
        peak2      = _peak_mb(2)

        if peak1 == float("inf") or peak2 == float("inf"):
            return 1

        per_sample_mb         = max(1.0, peak2 - peak1)
        fixed_inference_mb    = max(0.0, peak1 - alloc_mb)
        available_mb          = total_mb - alloc_mb - fixed_inference_mb - _VRAM_SAFETY_MB
        max_par               = max(1, int(available_mb / per_sample_mb))
        max_par               = min(max_par, _VRAM_PRACTICAL_CAP)

        print(
            f"[KronosForecaster] VRAM: total={total_mb:.0f} MB  "
            f"model={alloc_mb:.0f} MB  per_sample={per_sample_mb:.0f} MB  "
            f"→ max_parallel={max_par}"
        )
        return max_par

    # ── Core parallel sampling ────────────────────────────────────────────────

    def _run_parallel_batch(
        self,
        df_input:    pd.DataFrame,
        x_timestamp: pd.Series,
        y_timestamp: pd.Series,
        pred_len:    int,
        batch_size:  int,
        temperature: float,
        top_p:       float,
        verbose:     bool,
    ) -> np.ndarray:
        """Run exactly `batch_size` stochastic paths in one parallel GPU pass.

        Returns (batch_size, pred_len, 6) array in original price scale.
        """
        from model.kronos import auto_regressive_inference, calc_time_stamps  # noqa: PLC0415

        p = self._predictor

        df = df_input[p.price_cols].copy()
        df[p.vol_col] = 0.0
        df[p.amt_vol] = 0.0

        x_time_df = calc_time_stamps(x_timestamp)
        y_time_df = calc_time_stamps(y_timestamp)

        x       = df[p.price_cols + [p.vol_col, p.amt_vol]].values.astype(np.float32)
        x_stamp = x_time_df.values.astype(np.float32)
        y_stamp = y_time_df.values.astype(np.float32)

        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x_norm        = np.clip((x - x_mean) / (x_std + 1e-5), -p.clip, p.clip)

        # expand batch dim to batch_size (all paths share the same input)
        x_t  = torch.from_numpy(x_norm[np.newaxis]).to(p.device).repeat(batch_size, 1, 1)
        xs_t = torch.from_numpy(x_stamp[np.newaxis]).to(p.device).repeat(batch_size, 1, 1)
        ys_t = torch.from_numpy(y_stamp[np.newaxis]).to(p.device).repeat(batch_size, 1, 1)

        with torch.no_grad():
            raw = auto_regressive_inference(
                p.tokenizer, p.model,
                x_t, xs_t, ys_t,
                p.max_context, pred_len,
                p.clip, temperature, 0, top_p,
                1,       # sample_count=1; we already expanded to batch_size
                verbose,
            )
            # raw: (batch_size, pred_len, 6)  on CPU as numpy

        raw = raw[:, -pred_len:, :]
        raw = raw * (x_std + 1e-5) + x_mean    # denormalise
        return raw.astype(np.float32)

    def _generate_samples(
        self,
        df_input:    pd.DataFrame,
        x_timestamp: pd.Series,
        y_timestamp: pd.Series,
        pred_len:    int,
        n_samples:   int,
        temperature: float,
        top_p:       float,
        verbose:     bool,
    ) -> np.ndarray:
        """Generate n_samples paths, chunked into batches of self._max_parallel.

        Returns (n_samples, pred_len, 6).
        """
        chunks = []
        remaining = n_samples
        batch_num = 0
        total_batches = math.ceil(n_samples / self._max_parallel)

        while remaining > 0:
            bs = min(remaining, self._max_parallel)
            if total_batches > 1:
                print(f"  [samples] batch {batch_num + 1}/{total_batches}  ({bs} paths)", flush=True)
            chunk = self._run_parallel_batch(
                df_input, x_timestamp, y_timestamp, pred_len, bs, temperature, top_p, verbose,
            )
            chunks.append(chunk)
            remaining -= bs
            batch_num += 1

        return np.concatenate(chunks, axis=0)   # (n_samples, pred_len, 6)

    # ── Public forecast API ───────────────────────────────────────────────────

    def forecast(
        self,
        ohlcv_df:    pd.DataFrame,
        pred_len:    int,
        x_timestamp: pd.Series,
        y_timestamp: pd.Series,
        temperature: float = 1.0,
        top_p:       float = 0.9,
        n_samples:   int   = 1,
        percentiles: Sequence[float] | None = None,
        verbose:     bool  = False,
    ) -> dict:
        """Run Kronos inference.

        n_samples=1  → single OHLCV forecast (point estimate)
        n_samples>1  → stochastic ensemble; paths are batched in chunks of
                       max_parallel (auto-detected from VRAM at load time).

        Args:
            ohlcv_df:    DataFrame with OHLC columns.
            pred_len:    Bars to forecast.
            x_timestamp: Historical timestamps (pd.Series).
            y_timestamp: Future timestamps (pd.Series, length=pred_len).
            temperature: Sampling temperature. Must be > 0 for meaningful variance.
            top_p:       Nucleus sampling threshold.
            n_samples:   Number of stochastic paths to draw.
            percentiles: Quantile levels for the returned 'quantiles' dict
                         (e.g. [0.05, 0.25, 0.50, 0.75, 0.95]).
                         Only used when n_samples > 1.

        Returns dict:
            n_samples=1:
                mean_forecast      pd.DataFrame (pred_len, 6)
                samples            [mean_forecast]  (list of length 1)
                close_samples      np.ndarray (1, pred_len)
                all_samples_raw    np.ndarray (1, pred_len, 6)
            n_samples>1, additionally:
                quantiles          dict {float: np.ndarray(pred_len)} — close percentiles
            always:
                forecast_timestamps  pd.DatetimeIndex
                context_ohlcv        pd.DataFrame
                pred_len             int
                n_samples            int
        """
        self._ensure_loaded()

        df_input = ohlcv_df[OHLC_COLS].copy()
        all_raw  = self._generate_samples(
            df_input, x_timestamp, y_timestamp,
            pred_len, n_samples, temperature, top_p, verbose,
        )
        # all_raw: (n_samples, pred_len, 6)

        col_names     = ["open", "high", "low", "close", "volume", "amount"]
        y_idx         = pd.DatetimeIndex(y_timestamp)
        samples       = [pd.DataFrame(all_raw[i], columns=col_names, index=y_idx) for i in range(n_samples)]
        mean_forecast = pd.DataFrame(all_raw.mean(axis=0), columns=col_names, index=y_idx)
        close_matrix  = all_raw[:, :, 3]   # close is feature index 3

        result = {
            "samples":             samples,
            "mean_forecast":       mean_forecast,
            "close_samples":       close_matrix,
            "all_samples_raw":     all_raw,
            "forecast_timestamps": y_idx,
            "context_ohlcv":       ohlcv_df,
            "pred_len":            pred_len,
            "n_samples":           n_samples,
        }

        if n_samples > 1 and percentiles:
            result["quantiles"] = {
                float(p): np.percentile(close_matrix, p * 100, axis=0)
                for p in percentiles
            }

        return result

    # ── Higher-level entry points ──────────────────────────────────────────────

    def forecast_from_df(
        self,
        price_df:    pd.DataFrame,
        context_end: str | pd.Timestamp,
        pred_len:    int   = 20,
        temperature: float = 1.0,
        top_p:       float = 0.9,
        n_samples:   int   = 1,
        percentiles: Sequence[float] | None = None,
        verbose:     bool  = False,
    ) -> dict:
        """Forecast from a raw OHLCV DataFrame (analogous to ChronosForecaster.forecast_from_df)."""
        price_df = price_df.copy()
        price_df.index = pd.to_datetime(price_df.index)

        loc = price_df.index.get_indexer([pd.Timestamp(context_end)], method="pad")[0]
        if loc < 0:
            loc = 0
        cutoff_ts = price_df.index[loc]

        context_df  = price_df.iloc[max(0, loc + 1 - self.max_context): loc + 1].copy()
        x_timestamp = pd.Series(context_df.index)

        bar_td      = (context_df.index[-1] - context_df.index[-2]) if len(context_df) >= 2 else timedelta(hours=1)
        y_idx       = _make_future_timestamps(cutoff_ts, bar_td, pred_len)
        y_timestamp = pd.Series(y_idx)

        fc = self.forecast(
            context_df[OHLC_COLS], pred_len, x_timestamp, y_timestamp,
            temperature, top_p, n_samples, percentiles, verbose,
        )

        gt_end          = min(loc + 1 + pred_len, len(price_df))
        ground_truth_df = price_df.iloc[loc + 1: gt_end] if gt_end > loc + 1 else None

        fc.update({
            "context_df":        context_df,
            "ground_truth_df":   ground_truth_df,
            "context_end":       cutoff_ts,
            "forecast_start":    y_idx[0],
            "prediction_length": pred_len,
            "pair":              price_df.attrs.get("pair", ""),
            "timeframe":         price_df.attrs.get("timeframe", ""),
        })
        return fc

    def forecast_from_pipeline(
        self,
        results:     dict,
        pred_len:    int   = 20,
        context_end: str   = "train_end",
        temperature: float = 1.0,
        top_p:       float = 0.9,
        n_samples:   int   = 1,
        percentiles: Sequence[float] | None = None,
    ) -> dict:
        """Convenience wrapper for ForexPipeline output dict."""
        raw_m1  = results["raw_m1"]
        ce_map  = {
            "train_end": results["train"].index[-1],
            "val_end":   results["val"].index[-1],
            "test_end":  results["test"].index[-1],
        }
        cutoff = ce_map.get(context_end, context_end)
        return self.forecast_from_df(raw_m1, cutoff, pred_len, temperature, top_p, n_samples, percentiles)
