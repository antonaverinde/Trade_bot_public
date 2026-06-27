"""
Forex Data Pipeline
====================
Ingests raw M1 OHLCV data → resamples to multiple timeframes →
computes all features → outputs walk-forward-safe train/val/test splits.

Pairs: EUR/USD, GBP/USD (easily extended)
Timeframes: M1, M5, M15, H1, H4, D1

Normalization modes (applied before feature engineering):
  "log_returns" — log(col/col.shift(1)) per OHLC bar (default)
  "fracdiff"    — fractional differentiation (López de Prado), d in (0,1)
  "raw"         — no transformation; Chronos uses this

Target modes:
  "lag"            — direction_{h} and future_ret_{h} (original)
  "triple_barrier" — tb_label (1=TP, -1=SL, 0=time expired), tb_ret
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings

from torch import threshold
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

class ForexDataLoader:
    """
    Loads M1 OHLCV data from HistData CSVs.
    Expected format: semicolon-delimited, no header,
    columns: YYYYMMDD HHMMSS;open;high;low;close;volume
    Volume from HistData is always 0 — replaced with 1.0 placeholder.
    """

    REQUIRED_COLS = {"open", "high", "low", "close"}

    def load_csv(self, path: str, pair: str = "EURUSD") -> pd.DataFrame:
        df = pd.read_csv(
            path,
            sep=";",
            header=None,
            names=["datetime", "open", "high", "low", "close", "volume"],
            parse_dates=["datetime"],
            date_format="%Y%m%d %H%M%S",
            index_col="datetime",
        )
        df.columns = df.columns.str.lower()
        missing = self.REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        if "volume" not in df.columns:
            df["volume"] = 1.0
        # HistData volume is always 0 — replace with placeholder
        elif (df["volume"] == 0).all():
            df["volume"] = 1.0
        df = df[["open", "high", "low", "close", "volume"]].sort_index()
        df.attrs["pair"] = pair
        print(f"[Loader] {pair}: {len(df):,} M1 bars  |  {df.index[0]} → {df.index[-1]}")
        return df

    def load_and_merge(self, histdata_dir: str, pair: str,
                       years: list[int] = None,
                       weekends: str = "nogap") -> pd.DataFrame:
        """Load and merge multiple annual HistData CSVs for one pair."""
        histdata_dir = Path(histdata_dir)
        pattern = f"DAT_ASCII_{pair}_M1_*.csv"
        files = sorted(histdata_dir.glob(pattern))
        if years:
            files = [f for f in files if int(f.stem[-4:]) in years]
        if not files:
            raise FileNotFoundError(f"No files matching {pattern} in {histdata_dir}")
        dfs = [self.load_csv(f, pair=pair) for f in files]
        df = pd.concat(dfs).sort_index()
        df = df[~df.index.duplicated(keep="first")]

        # Fill intra-week gaps (thin liquidity, data dropouts) with previous bar.
        df = df.resample("1min").ffill()
        if weekends == "nogap":
            df = df[df.index.dayofweek < 5]
        elif weekends == "filled":
            df = _extend_weekend_grid(df, "1min", fill=True)
        else:  # "gaps"
            df = _extend_weekend_grid(df, "1min", fill=False)

        df.attrs["pair"] = pair
        print(f"[Loader] Merged {pair}: {len(df):,} M1 bars  |  {df.index[0]} → {df.index[-1]}")
        return df

    def generate_synthetic(self, pair: str = "EURUSD", n_bars: int = 100_000,
                           start: str = "2020-01-01") -> pd.DataFrame:
        """
        Generates realistic synthetic M1 Forex data for testing.
        Uses GBM with volatility clustering (simplified GARCH-like).
        """
        np.random.seed(42)
        idx = pd.date_range(start, periods=n_bars, freq="1min")
        # Remove weekends
        idx = idx[idx.dayofweek < 5]
        n = len(idx)

        # GBM with vol clustering
        vol = np.zeros(n)
        vol[0] = 0.0001
        returns = np.zeros(n)
        for i in range(1, n):
            vol[i] = np.sqrt(0.00001 + 0.1 * returns[i-1]**2 + 0.85 * vol[i-1]**2)
            returns[i] = vol[i] * np.random.randn()

        price_base = 1.1000 if pair == "EURUSD" else 1.2500
        closes = price_base * np.exp(np.cumsum(returns))

        highs = closes * (1 + np.abs(np.random.normal(0, 0.0001, n)))
        lows  = closes * (1 - np.abs(np.random.normal(0, 0.0001, n)))
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        volumes = np.random.lognormal(5, 1, n)

        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows,
            "close": closes, "volume": volumes
        }, index=idx)
        df.attrs["pair"] = pair
        print(f"[Loader] Synthetic {pair}: {len(df):,} M1 bars  |  {df.index[0]} → {df.index[-1]}")
        return df


# ─────────────────────────────────────────────
# 2. RESAMPLER — M1 → multiple timeframes
# ─────────────────────────────────────────────

TIMEFRAMES = {
    "M1":  "1min",
    "M5":  "5min",
    "M15": "15min",
    "H1":  "1h",
    "H4":  "4h",
    "D1":  "1D",
}

def resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    agg = {
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }
    resampled = df.resample(freq).agg(agg).dropna()
    return resampled

def _extend_weekend_grid(df: pd.DataFrame, freq: str, fill: bool) -> pd.DataFrame:
    """Reindex df to a full 7-day calendar at freq. fill=True ffills, False leaves NaN."""
    full_idx = pd.date_range(df.index[0], df.index[-1], freq=freq)
    df = df.reindex(full_idx)
    if fill:
        df = df.ffill()
    return df

def build_all_timeframes(df_m1: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tfs = {}
    for name, freq in TIMEFRAMES.items():
        tfs[name] = resample_ohlcv(df_m1, freq)
        print(f"  [{name}] {len(tfs[name]):,} bars")
    return tfs


# ─────────────────────────────────────────────
# 3. PRICE NORMALIZATION
# ─────────────────────────────────────────────

def _fracdiff_weights(d: float, threshold: float = 6*1e-4) -> np.ndarray:#200 - 92.9% 3.0e-4,1000  97.4% 4.3e-5   500   95.7% 1.0e-4       
    """Compute fractional differentiation weights (López de Prado)."""
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1])  # oldest weight first


def normalize_prices(df: pd.DataFrame, method: str = "log_returns",
                     d: float = 0.4,threshold: float = 6*1e-4) -> pd.DataFrame:
    """
    Normalize OHLCV before feature engineering.

    method="log_returns"  — log(col_t / col_{t-1}) per OHLC column (default)
    method="fracdiff"     — fractional differentiation with order d
    method="raw"          — no transformation (pass-through for Chronos)

    Volume is always log1p-transformed (non-negative, heavy-tailed).
    """
    df = df.copy()
    price_cols = ["open", "high", "low", "close"]

    if method == "raw":
        return df

    if method == "log_returns":
        for col in price_cols:
            df[col] = np.log(df[col] / df[col].shift(1))
        df["volume"] = np.log1p(df["volume"])
        return df

    if method == "fracdiff":
        weights = _fracdiff_weights(d=d,threshold=threshold)
        w_len = len(weights)
        for col in price_cols:
            vals = df[col].values.astype(float)
            result = np.full(len(vals), np.nan)
            for i in range(w_len - 1, len(vals)):
                result[i] = np.dot(weights, vals[i - w_len + 1: i + 1])
            df[col] = result
        df["volume"] = np.log1p(df["volume"])
        return df

    raise ValueError(f"Unknown normalization method: {method!r}. "
                     f"Choose 'log_returns', 'fracdiff', or 'raw'.")


def _resolve_barrier_norm_method(
    barrier_norm_method: str | None,
    barrier_on_raw: bool | None,
) -> str:
    """
    Resolve the triple-barrier calculation basis.

    barrier_on_raw is kept as a compatibility alias:
      True  -> "raw"
      False -> "features"
    """
    if barrier_norm_method is None:
        method = "raw" if barrier_on_raw is not False else "features"
    else:
        method = barrier_norm_method

    valid = {"raw", "log_returns", "fracdiff", "features"}
    if method not in valid:
        raise ValueError(
            f"Unknown barrier_norm_method: {method!r}. "
            f"Choose one of {sorted(valid)}."
        )
    return method


# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────

class FeatureEngineer:
    """
    Computes a curated set of features from normalized OHLCV data.
    All features are computed with .shift() — zero look-ahead bias.

    Active features (computed on normalized price):
      - RSI 14 & 21: speed, acceleration, level crosses (50/70)
      - ADX 14: trend strength, +DI/-DI diff, delta (acceleration)
      - Distance to EMA 200
      - Relative ATR (ATR / close)
      - %B Bollinger Band position
      - Time: hour/DOW sin+cos, is_monday, is_friday
      - Candle structure: body ratio, shadow ratio (upper), body gap
      - Distribution: rolling skew, kurtosis, EWMA volatility
      - Lag features: rsi_14, rsi_14_speed, rsi_21, rsi_21_speed, bb_pct_b

    Commented-out methods from v1 are preserved below for reference.
    """

    def __init__(self, lags: list[int] = [1, 2, 5, 10]):
        self.lags = lags

    # ── RSI helper ────────────────────────────
    def _rsi(self, series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    # ── RSI features (14 and 21) ──────────────
    def _rsi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for period in [14, 21]:
            rsi = self._rsi(df["close"], period)
            rsi_prev = rsi.shift(1)
            speed = rsi - rsi_prev
            if period == 21:
                # Disabled as highly correlated pipeline features.
                # Uncomment these three lines to restore rsi_21, rsi_21_speed,
                # and rsi_21_accel in the default feature set.
                # df[f"rsi_{period}"]          = rsi
                # df[f"rsi_{period}_speed"]    = speed
                # df[f"rsi_{period}_accel"]    = speed - speed.shift(1)
                pass
            else:
                df[f"rsi_{period}"]          = rsi
                df[f"rsi_{period}_speed"]    = speed
                df[f"rsi_{period}_accel"]    = speed - speed.shift(1)
            df[f"rsi_{period}_cross_50"] = ((rsi >= 50) & (rsi_prev < 50)).astype(int)
            df[f"rsi_{period}_cross_70"] = ((rsi >= 70) & (rsi_prev < 70)).astype(int)
        return df

    # ── ATR helper ────────────────────────────
    def _atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        h, l, c = df["high"], df["low"], df["close"].shift(1)
        tr = pd.concat([
            h - l,
            (h - c).abs(),
            (l - c).abs()
        ], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    # ── ADX features ─────────────────────────
    def _adx_features(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        high, low, close = df["high"], df["low"], df["close"]
        prev_high = high.shift(1)
        prev_low  = low.shift(1)
        prev_close= close.shift(1)

        # True Range
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs()
        ], axis=1).max(axis=1)

        # Directional movement
        dm_plus  = (high - prev_high).clip(lower=0)
        dm_minus = (prev_low - low).clip(lower=0)
        # Zero out when the other DM is larger
        mask = dm_plus < dm_minus
        dm_plus[mask] = 0.0
        mask = dm_minus < dm_plus
        dm_minus[mask] = 0.0

        # Wilder smoothing
        atr_w  = tr.ewm(span=period, adjust=False).mean()
        di_plus  = 100 * dm_plus.ewm(span=period, adjust=False).mean()  / (atr_w + 1e-9)
        di_minus = 100 * dm_minus.ewm(span=period, adjust=False).mean() / (atr_w + 1e-9)

        dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-9)
        adx = dx.ewm(span=period, adjust=False).mean()

        df["adx_14"]   = adx
        df["di_diff"]  = di_plus - di_minus          # +: bullish, -: bearish
        df["adx_delta"]= adx - adx.shift(3)          # trend building (+) or fading (-)
        return df

    # ── EMA 200 distance ─────────────────────
    def _ma_distance(self, df: pd.DataFrame) -> pd.DataFrame:
        ema200 = df["close"].ewm(span=200, adjust=False).mean()
        df["dist_ema200"] = (df["close"] - ema200) / (df["close"].abs() + 1e-9)
        return df

    # ── Relative ATR ─────────────────────────
    def _relative_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        df["atr_rel"] = self._atr(df, period) / (df["close"].abs() + 1e-9)
        return df

    # ── Bollinger %B ─────────────────────────
    def _bollinger_pct_b(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        mid   = df["close"].rolling(period).mean()
        std   = df["close"].rolling(period).std()
        upper = mid + 2 * std
        lower = mid - 2 * std
        pct_b = (df["close"] - lower) / (upper - lower + 1e-9)
        df["bb_pct_b"] = pct_b.clip(-0.5, 1.5)
        return df

    # ── Time features ─────────────────────────
    def _time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        idx = df.index
        df["hour_sin"]   = np.sin(2 * np.pi * idx.hour / 24)
        df["hour_cos"]   = np.cos(2 * np.pi * idx.hour / 24)
        df["dow_sin"]    = np.sin(2 * np.pi * idx.dayofweek / 5)
        df["dow_cos"]    = np.cos(2 * np.pi * idx.dayofweek / 5)
        df["is_monday"]  = (idx.dayofweek == 0).astype(int)
        df["is_friday"]  = (idx.dayofweek == 4).astype(int)
        return df

    # ── Candle structure ──────────────────────
    def _candle_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        body        = (df["close"] - df["open"]).abs()
        candle_range= (df["high"] - df["low"] + 1e-10)
        upper_shadow= df["high"] - df[["open", "close"]].max(axis=1)

        # Body ratio: what fraction of the candle is body (0 → doji, 1 → marubozu)
        df["body_ratio"]  = body / candle_range

        # Disabled as a highly correlated pipeline feature.
        # Uncomment this line to restore shadow_ratio in the default feature set.
        # df["shadow_ratio"]= upper_shadow / candle_range

        # Normalized gap: open vs previous close as a fraction of price
        df["body_gap"]    = (df["open"] - df["close"].shift(1)) / (df["close"].shift(1).abs() + 1e-9)
        return df

    # ── Distribution features ─────────────────
    def _distribution_features(self, df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
        close = df["close"]
        df["ret_skew"]  = close.rolling(window).skew()
        df["ret_kurt"]  = close.rolling(window).kurt()
        # EWMA volatility — same metric used in triple barrier labeling.
        # Gives the model a sense of current barrier width.
        df["vol_ewma"]  = close.pct_change().ewm(span=100).std()
        return df

    # ── Lag features ──────────────────────────
    def _lags(self, df: pd.DataFrame) -> pd.DataFrame:
        # Lags of close itself (normalized) — pure price memory signal.
        for lag in self.lags:
            df[f"close_lag{lag}"] = df["close"].shift(lag)
        return df

    # ── Targets ───────────────────────────────
    def _triple_barrier_targets(self, df: pd.DataFrame,
                                k_up: float, k_down: float,
                                horizon_bars: int,
                                barrier_price: str = "close",
                                barrier_on_raw: bool = True,
                                barrier_norm_method: str | None = None) -> pd.DataFrame:
        """
        Triple barrier labeling (López de Prado).

        For each bar t, scan the chosen barrier basis until one of three exits wins:
          upper barrier -> label 1
          lower barrier -> label -1
          no hit before horizon_bars -> label 0

        Raw-price barriers use multiplicative levels. Transformed bases
        (log_returns, fracdiff, or normalized features) use additive levels.
        """
        close   = df["close"].values
        n       = len(close)
        labels  = np.zeros(n, dtype=np.int8)
        tb_ret  = np.zeros(n, dtype=np.float32)

        # Prefer explicit barrier helper columns prepared by ForexPipeline.run().
        # Fall back to the pre-existing raw/feature behavior for direct FeatureEngineer use.
        if "_barrier_close" in df.columns:
            barrier_close = df["_barrier_close"].values
            barrier_high  = df["_barrier_high"].values
            barrier_low   = df["_barrier_low"].values
            sigma         = df["_barrier_sigma"].values
            multiplicative = bool(df["_barrier_multiplicative"].iloc[0])
        elif barrier_on_raw and "_raw_close" in df.columns:
            barrier_close = df["_raw_close"].values
            barrier_high  = df["_raw_high"].values
            barrier_low   = df["_raw_low"].values
            sigma         = df["_raw_sigma"].values
            multiplicative = True
        else:
            barrier_close = close
            barrier_high  = df["high"].values
            barrier_low   = df["low"].values
            sigma         = df["vol_ewma"].values
            multiplicative = barrier_norm_method == "raw"

        if barrier_price not in {"close", "hl"}:
            raise ValueError("barrier_price must be 'close' or 'hl'")

        if barrier_price == "hl":
            check_up   = barrier_high
            check_down = barrier_low
        else:
            check_up   = barrier_close
            check_down = barrier_close

        for t in range(n - horizon_bars):
            if multiplicative:
                upper = barrier_close[t] * (1 + sigma[t] * k_up)
                lower = barrier_close[t] * (1 - sigma[t] * k_down)
            else:
                upper = barrier_close[t] + sigma[t] * k_up
                lower = barrier_close[t] - sigma[t] * k_down

            window_up   = check_up[t + 1: t + horizon_bars + 1]
            window_down = check_down[t + 1: t + horizon_bars + 1]

            hit_up   = np.argmax(window_up >= upper)   if (window_up >= upper).any()   else -1
            hit_down = np.argmax(window_down <= lower)  if (window_down <= lower).any() else -1

            if hit_up == -1 and hit_down == -1:
                exit_idx = t + horizon_bars
                labels[t] = 0
            elif hit_up == -1:
                exit_idx = t + 1 + hit_down
                labels[t] = -1
            elif hit_down == -1:
                exit_idx = t + 1 + hit_up
                labels[t] = 1
            else:
                if hit_up <= hit_down:
                    exit_idx = t + 1 + hit_up
                    labels[t] = 1
                else:
                    exit_idx = t + 1 + hit_down
                    labels[t] = -1

            if multiplicative:
                tb_ret[t] = np.log(barrier_close[exit_idx] / barrier_close[t] + 1e-9)
            else:
                tb_ret[t] = barrier_close[exit_idx] - barrier_close[t]

        # Last horizon_bars rows have no valid label
        labels[-horizon_bars:] = 0
        tb_ret[-horizon_bars:] = np.nan

        df["tb_label"] = labels
        df["tb_ret"]   = tb_ret
        return df

    def _targets(self, df: pd.DataFrame,
                 horizons: list[int],
                 target_type: str,
                 k_up: float,
                 k_down: float,
                 horizon_bars: int,
                 barrier_price: str = "close",
                 barrier_on_raw: bool = True,
                 barrier_norm_method: str | None = None) -> pd.DataFrame:
        if target_type == "triple_barrier":
            df = self._triple_barrier_targets(
                df, k_up, k_down, horizon_bars, barrier_price,
                barrier_on_raw, barrier_norm_method
            )
        else:
            # Original lag-based targets
            for h in horizons:
                future_ret = df["close"].shift(-h)
                df[f"future_ret_{h}"] = future_ret
                df[f"direction_{h}"]  = (future_ret > 0).astype(int)
        return df

    # ── Main entry point ──────────────────────
    def transform(self, df: pd.DataFrame,
                  target_horizons: list[int] = [1, 5, 15],
                  target_type: str = "lag",
                  k_up: float = 2.0,
                  k_down: float = 1.0,
                  horizon_bars: int = 10,
                  barrier_price: str = "close",
                  barrier_on_raw: bool = True,
                  barrier_norm_method: str | None = None) -> pd.DataFrame:
        df = df.copy()
        df = self._rsi_features(df)
        df = self._adx_features(df)
        df = self._ma_distance(df)
        df = self._relative_atr(df)
        df = self._bollinger_pct_b(df)
        df = self._time_features(df)
        df = self._candle_structure(df)
        df = self._distribution_features(df)
        df = self._lags(df)
        df = self._targets(df, horizons=target_horizons, target_type=target_type,
                           k_up=k_up, k_down=k_down, horizon_bars=horizon_bars,
                           barrier_price=barrier_price, barrier_on_raw=barrier_on_raw,
                           barrier_norm_method=barrier_norm_method)
        df = df.dropna()
        return df

    # ── Commented-out v1 methods (kept for reference) ─────────────────────────
    # def _log_returns(self, df): ...   # now handled by normalize_prices()
    # def _volatility(self, df): ...    # atr_14, vol_roll_1h, vol_roll_1d
    # def _technicals(self, df): ...    # rsi_6, dist_ema20/50, MACD, old BB
    # def _microstructure(self, df): .. # log_volume, vol_zscore, candle_dir
    # Old _time_features had session flags (sess_asia, sess_london, sess_ny, sess_overlap)
    # Old _lags used: ret_lag{n}, vol_lag{n}, rsi_lag{n}
    # Old _targets: direction_{h} from log_return.shift(-h)


# ─────────────────────────────────────────────
# 5. WALK-FORWARD SPLITTER
# ─────────────────────────────────────────────

class WalkForwardSplitter:
    """
    Produces (train, val, test) index splits with a gap between each
    to prevent leakage from autocorrelated labels.

    Strategy:
      ├── train  : 60% of data
      ├── gap    : gap_bars bars (dropped) — prevents label leakage
      ├── val    : 20% of data
      ├── gap    : gap_bars bars (dropped)
      └── test   : 20% of data (never touched until final eval)
    """

    def __init__(self, train_ratio: float = 0.6, val_ratio: float = 0.2,
                 gap_bars: int = 50):
        self.train_ratio = train_ratio
        self.val_ratio   = val_ratio
        self.gap_bars    = gap_bars

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        n = len(df)
        train_end  = int(n * self.train_ratio)
        val_start  = train_end + self.gap_bars
        val_end    = val_start + int(n * self.val_ratio)
        test_start = val_end + self.gap_bars

        train = df.iloc[:train_end]
        val   = df.iloc[val_start:val_end]
        test  = df.iloc[test_start:]

        print(f"\n[Splitter] Walk-forward split (gap={self.gap_bars} bars):")
        print(f"  Train : {len(train):>7,} bars  {train.index[0].date()} → {train.index[-1].date()}")
        print(f"  Val   : {len(val):>7,} bars  {val.index[0].date()} → {val.index[-1].date()}")
        print(f"  Test  : {len(test):>7,} bars  {test.index[0].date()} → {test.index[-1].date()}")
        return train, val, test


# ─────────────────────────────────────────────
# 6. SCALER — fit on train only, apply to all
# ─────────────────────────────────────────────

class ForexScaler:
    """
    Robust scaler (median + IQR) — more appropriate than MinMax for
    financial data with fat tails. Fitted on train set ONLY.
    """

    def __init__(self):
        self.stats: dict = {}

    @property
    def feature_cols(self):
        return list(self.stats.keys())

    def fit(self, train: pd.DataFrame, feature_cols: list[str]) -> "ForexScaler":
        for col in feature_cols:
            q25 = train[col].quantile(0.25)
            q75 = train[col].quantile(0.75)
            iqr = q75 - q25 + 1e-9
            self.stats[col] = {"median": train[col].median(), "iqr": iqr}
        print(f"[Scaler] Fitted on {len(train):,} train bars | {len(feature_cols)} features")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, s in self.stats.items():
            if col in df.columns:
                df[col] = (df[col] - s["median"]) / s["iqr"]
        return df

    def fit_transform(self, train: pd.DataFrame,
                      feature_cols: list[str]) -> pd.DataFrame:
        self.fit(train, feature_cols)
        return self.transform(train)


class RollingScaler:
    """
    Window-based robust scaler. For each bar, normalizes using median and IQR
    computed over the preceding `window_size` bars.
    Regime-aware: adapts to volatility and level shifts over time.
    No fit() step — stats are computed inline, safe to call independently
    on each split.
    """

    def __init__(self, window_size: int = 500):
        self.window_size = window_size
        self._feature_cols: list[str] = []

    @property
    def feature_cols(self) -> list[str]:
        return self._feature_cols

    def transform(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        df = df.copy()
        self._feature_cols = feature_cols
        for col in feature_cols:
            if col not in df.columns:
                continue
            s = df[col]
            roll = s.rolling(self.window_size, min_periods=1)
            med = roll.median()
            iqr = roll.quantile(0.75) - roll.quantile(0.25) + 1e-9
            df[col] = (s - med) / iqr
        print(f"[RollingScaler] window={self.window_size} | {len(feature_cols)} features")
        return df

    def fit_transform(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        return self.transform(df, feature_cols)


# ─────────────────────────────────────────────
# 7. FULL PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────

def get_feature_cols(df: pd.DataFrame, include_raw: bool = False) -> list[str]:
    """Return all feature columns — excludes raw OHLCV and target columns."""
    exclude = set() if include_raw else {"open", "high", "low", "close", "volume"}
    exclude |= {c for c in df.columns
                if c.startswith("future_") or c.startswith("direction_")
                or c.startswith("_raw_") or c.startswith("_barrier_")
                or c in {"tb_label", "tb_ret"}}
    return [c for c in df.columns if c not in exclude]


def build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a flat split DataFrame into 3D sliding-window arrays for
    sequence models (Transformer, LSTM).

    Returns
    -------
    X : shape (n_samples, seq_len, n_features)
    y : shape (n_samples,)
    """
    X_flat = df[feature_cols].values
    y_flat = df[target_col].values
    X = np.lib.stride_tricks.sliding_window_view(X_flat, (seq_len, X_flat.shape[1]))
    X = X[:, 0, :, :]
    y = y_flat[seq_len - 1:]
    return X, y


class ForexPipeline:
    """
    End-to-end pipeline. Call .run() with a raw M1 DataFrame.
    Returns a dict with everything needed for model training.

    Parameters
    ----------
    norm_method : "log_returns" | "fracdiff" | "raw"
        Normalization applied to OHLCV before feature engineering.
        Chronos always receives raw price levels via result["raw_m1"].
    fracdiff_d : float
        Fractional differentiation order (only used when norm_method="fracdiff").
    target_type : "lag" | "triple_barrier"
        "lag"            → direction_{h}, future_ret_{h} for each horizon
        "triple_barrier" → tb_label (1/-1/0), tb_ret
    k_up, k_down : float
        Barrier multipliers for triple barrier (k_up * vol_ewma above/below price).
    horizon_bars : int
        Max bars to hold before time barrier triggers (triple barrier only).
    barrier_price : "close" | "hl"
        "close" — use close prices to check both upper and lower barriers.
        "hl"    — use high to check the upper barrier, low for the lower barrier.
    barrier_norm_method : "raw" | "log_returns" | "fracdiff" | "features"
        Data basis used for triple-barrier label calculation. "features" means
        reuse the normalized OHLC produced by norm_method. If omitted,
        barrier_on_raw remains the compatibility alias.
    weekends : "nogap" | "gaps" | "filled"
        "nogap"  — default; 5-day week, no weekend rows.
        "gaps"   — 7-day calendar; weekend rows present, all values NaN.
        "filled" — 7-day calendar; weekend rows forward-filled from Friday's close.
    """

    def __init__(self,
                 lags: list[int] = [1, 2, 5, 10],
                 target_horizons: list[int] = [1, 5, 15],
                 gap_bars: int = 50,
                 scaling: str = "global",
                 window_size: int = 500,
                 include_raw: bool = False,
                 seq_len: int = 0,
                 norm_method: str = "log_returns",
                 fracdiff_d: float = 0.4,
                 target_type: str = "lag",
                 k_up: float = 2.0,
                 k_down: float = 1.0,
                 horizon_bars: int = 10,
                 barrier_price: str = "close",
                 barrier_on_raw: bool = True,
                 barrier_norm_method: str | None = None,
                 threshold: float = 6*1e-4,
                 weekends: str = "nogap"):
        if include_raw and scaling != "rolling":
            warnings.warn(
                "[Pipeline] include_raw=True with global scaling is dangerous: "
                "raw OHLC prices are non-stationary. Use scaling='rolling'.",
                UserWarning, stacklevel=2,
            )
        self.norm_method    = norm_method
        self.fracdiff_d     = fracdiff_d
        self.target_type    = target_type
        self.k_up           = k_up
        self.k_down         = k_down
        self.horizon_bars   = horizon_bars
        self.barrier_price  = barrier_price
        self.barrier_on_raw = barrier_on_raw
        self.barrier_norm_method = _resolve_barrier_norm_method(
            barrier_norm_method, barrier_on_raw
        )
        self.include_raw    = include_raw
        self.seq_len        = seq_len
        self.weekends       = weekends
        self.target_horizons= target_horizons
        self.threshold = threshold 
        self.engineer  = FeatureEngineer(lags=lags)
        self.splitter  = WalkForwardSplitter(gap_bars=gap_bars)
        #self.scaler    = RollingScaler(window_size=window_size) if scaling == "rolling" else ForexScaler()
        self.scaler = (
        RollingScaler(window_size=window_size) if scaling == "rolling"
        else ForexScaler()                      if scaling == "global"
        else None
        )

    def run(self, df_m1: pd.DataFrame, timeframe: str = "M1") -> dict:
        pair = df_m1.attrs.get("pair", "UNKNOWN")
        print(f"\n{'='*55}")
        print(f"  Pipeline: {pair} | {timeframe} | norm={self.norm_method} | target={self.target_type} | weekends={self.weekends}")
        print(f"{'='*55}")

        # 1. Resample if needed
        if timeframe != "M1":
            freq = TIMEFRAMES[timeframe]
            df = resample_ohlcv(df_m1, freq)
            print(f"[Resample] M1 → {timeframe}: {len(df):,} bars")
        else:
            freq = "1min"
            df = df_m1.copy()

        if self.weekends == "filled":
            df = _extend_weekend_grid(df, freq, fill=True)
            print(f"[Weekends] Forward-filled to 7-day grid: {len(df):,} bars")
        elif self.weekends == "gaps":
            df = _extend_weekend_grid(df, freq, fill=False)
            print(f"[Weekends] Extended to 7-day grid (NaN weekends): {len(df):,} bars")

        # 2. Preserve raw OHLCV for Chronos (needs price levels, not log returns)
        df_raw = df.copy()

        # 3. Normalize prices before feature engineering
        df_norm = normalize_prices(df, method=self.norm_method, d=self.fracdiff_d,threshold=self.threshold)

        # Prepare a separate OHLC basis for triple-barrier labels. Feature output
        # still follows norm_method; this only controls label calculation.
        if self.target_type == "triple_barrier":
            if self.barrier_norm_method == "raw":
                df_barrier = df_raw
                barrier_multiplicative = True
                sigma = df_barrier["close"].pct_change().ewm(span=100).std()
            elif self.barrier_norm_method == "features":
                df_barrier = df_norm
                barrier_multiplicative = self.norm_method == "raw"
                sigma = (
                    df_barrier["close"].pct_change().ewm(span=100).std()
                    if barrier_multiplicative else
                    df_barrier["close"].ewm(span=100).std()
                )
            else:
                df_barrier = normalize_prices(
                    df, method=self.barrier_norm_method,
                    d=self.fracdiff_d, threshold=self.threshold
                )
                barrier_multiplicative = self.barrier_norm_method == "raw"
                sigma = df_barrier["close"].ewm(span=100).std()

            df_norm["_barrier_close"] = df_barrier["close"]
            df_norm["_barrier_high"]  = df_barrier["high"]
            df_norm["_barrier_low"]   = df_barrier["low"]
            df_norm["_barrier_sigma"] = sigma
            df_norm["_barrier_multiplicative"] = barrier_multiplicative

        # 4. Feature engineering
        print("[Features] Computing...")
        df_feat = self.engineer.transform(
            df_norm,
            target_horizons=self.target_horizons,
            target_type=self.target_type,
            k_up=self.k_up,
            k_down=self.k_down,
            horizon_bars=self.horizon_bars,
            barrier_price=self.barrier_price,
            barrier_on_raw=self.barrier_on_raw,
            barrier_norm_method=self.barrier_norm_method,
        )

        # Drop helper cols — they are excluded from feature_cols but clean up the df.
        helper_cols = [
            c for c in df_feat.columns
            if c.startswith("_raw_") or c.startswith("_barrier_")
        ]
        if helper_cols:
            df_feat = df_feat.drop(columns=helper_cols)

        feat_cols = get_feature_cols(df_feat, include_raw=self.include_raw)
        print(f"[Features] {len(feat_cols)} features | {len(df_feat):,} clean bars")

        # 5. Walk-forward split
        train, val, test = self.splitter.split(df_feat)

        # 6. Scale
        if self.scaler is None:
            train_scaled = train.copy()
            val_scaled   = val.copy()
            test_scaled  = test.copy()
        elif isinstance(self.scaler, RollingScaler):
            train_scaled = self.scaler.transform(train, feat_cols)
            val_scaled   = self.scaler.transform(val,   feat_cols)
            test_scaled  = self.scaler.transform(test,  feat_cols)
        else:
            train_scaled = self.scaler.fit_transform(train, feat_cols)
            val_scaled   = self.scaler.transform(val)
            test_scaled  = self.scaler.transform(test)

        # Determine target column names for convenience
        if self.target_type == "triple_barrier":
            target_cols   = ["tb_ret"]
            direction_cols= ["tb_label"]
        else:
            target_cols   = [f"future_ret_{h}" for h in self.target_horizons]
            direction_cols= [f"direction_{h}"  for h in self.target_horizons]

        result = {
            "pair":          pair,
            "timeframe":     timeframe,
            "feature_cols":  feat_cols,
            "target_cols":   target_cols,
            "direction_cols":direction_cols,
            # Raw unscaled splits (for target extraction)
            "train_raw":     train,
            "val_raw":       val,
            "test_raw":      test,
            # Scaled feature splits (ready for models)
            "train":         train_scaled,
            "val":           val_scaled,
            "test":          test_scaled,
            # Raw OHLCV for Chronos
            "raw_m1":        df_raw,
            "scaler":        self.scaler,
            "weekends":      self.weekends,
            "norm_method":   self.norm_method,
            "barrier_norm_method": self.barrier_norm_method,
            "barrier_price": self.barrier_price,
        }

        # 7. Sequence arrays for Transformer / LSTM (opt-in via seq_len > 0)
        if self.seq_len > 0:
            default_target = direction_cols[0]
            for name, split in [("train", train_scaled), ("val", val_scaled), ("test", test_scaled)]:
                X_seq, y_seq = build_sequences(split, feat_cols, default_target, self.seq_len)
                result[f"{name}_seq_X"] = X_seq
                result[f"{name}_seq_y"] = y_seq
                print(f"[Sequences] {name}: X={X_seq.shape}  y={y_seq.shape}")
            result["seq_len"] = self.seq_len

        return result

    def get_xy(self, split: pd.DataFrame,
               target: str,
               feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Helper: extract X (features) and y (target) arrays."""
        X = split[feature_cols].values
        y = split[target].values
        return X, y


# ─────────────────────────────────────────────
# 8. QUICK SMOKE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    loader   = ForexDataLoader()
    pipeline = ForexPipeline(
        lags=[1, 2, 5, 10],
        target_horizons=[1, 5, 15],
        gap_bars=50,
        norm_method="log_returns",
        target_type="lag",
    )

    df_m1 = loader.generate_synthetic("EURUSD", n_bars=200_000)

    results = {}
    for tf in ["M1", "M5", "M15", "H1"]:
        results[tf] = pipeline.run(df_m1, timeframe=tf)

    r = results["M15"]
    print(f"\n[M15 Features] ({len(r['feature_cols'])} total):")
    for i, f in enumerate(r["feature_cols"], 1):
        print(f"  {i:2d}. {f}")

    X_train, y_train = pipeline.get_xy(r["train"], "direction_1", r["feature_cols"])
    X_val,   y_val   = pipeline.get_xy(r["val"],   "direction_1", r["feature_cols"])
    print(f"\n[Ready for XGBoost]")
    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_val  : {X_val.shape}    y_val  : {y_val.shape}")
    print(f"  Class balance (train): {y_train.mean():.3f} positive")
