"""
FVG + Fractals event pipeline.

Builds an event-level dataset from raw M1 forex OHLCV data:
  raw M1 -> base/higher timeframe bars -> confirmed fractals + FVG events
  -> event features -> first-fractal-break targets -> train/val/test splits.

Fractals, FVGs, and targets are always calculated on raw OHLC prices. The
optional engineered feature block follows the existing pipeline normalization
setting so experiments can compare log returns, fracdiff, and raw features.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from Pipeline.pipeline import (
    FeatureEngineer,
    ForexDataLoader,
    ForexScaler,
    RollingScaler,
    TIMEFRAMES,
    normalize_prices,
    resample_ohlcv,
)


TARGET_COLUMNS = {
    "target_first_break_dir",
    "target_first_break_level_kind",
    "target_first_break_level_price",
    "target_first_break_bars",
    "target_is_ambiguous",
}


@dataclass(frozen=True)
class FractalSet:
    highs: pd.DataFrame
    lows: pd.DataFrame


def _timeframe_minutes(timeframe: str) -> int:
    if timeframe.startswith("M"):
        return int(timeframe[1:])
    if timeframe.startswith("H"):
        return int(timeframe[1:]) * 60
    if timeframe.startswith("D"):
        return int(timeframe[1:]) * 60 * 24
    raise ValueError(f"Unsupported timeframe: {timeframe!r}")


def _pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("JPY") else 0.0001


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _validate_fractal_window(fractal_window: int) -> None:
    if fractal_window < 3 or fractal_window % 2 == 0:
        raise ValueError("fractal_window must be an odd integer >= 3, e.g. 3 or 5.")


def detect_confirmed_fractals(df: pd.DataFrame, fractal_window: int) -> FractalSet:
    """
    Detect strict centered fractals and index them by confirmation time.

    A 5-bar high fractal centered at i is known only at i+2. Therefore the
    returned rows use confirmation_time as the index and keep center_time as
    metadata.
    """
    _validate_fractal_window(fractal_window)
    radius = fractal_window // 2
    highs: list[dict] = []
    lows: list[dict] = []
    high_vals = df["high"].to_numpy()
    low_vals = df["low"].to_numpy()
    idx = df.index

    for center in range(radius, len(df) - radius):
        left = center - radius
        right = center + radius + 1
        center_high = high_vals[center]
        center_low = low_vals[center]
        surrounding_highs = np.r_[high_vals[left:center], high_vals[center + 1:right]]
        surrounding_lows = np.r_[low_vals[left:center], low_vals[center + 1:right]]
        confirmation = center + radius

        if np.isfinite(center_high) and np.all(center_high > surrounding_highs):
            highs.append(
                {
                    "confirmation_time": idx[confirmation],
                    "center_time": idx[center],
                    "center_pos": center,
                    "confirmation_pos": confirmation,
                    "level": float(center_high),
                }
            )

        if np.isfinite(center_low) and np.all(center_low < surrounding_lows):
            lows.append(
                {
                    "confirmation_time": idx[confirmation],
                    "center_time": idx[center],
                    "center_pos": center,
                    "confirmation_pos": confirmation,
                    "level": float(center_low),
                }
            )

    high_df = pd.DataFrame(highs)
    low_df = pd.DataFrame(lows)
    if not high_df.empty:
        high_df = high_df.set_index("confirmation_time").sort_index()
    else:
        high_df = pd.DataFrame(columns=["center_time", "center_pos", "confirmation_pos", "level"])
        high_df.index.name = "confirmation_time"

    if not low_df.empty:
        low_df = low_df.set_index("confirmation_time").sort_index()
    else:
        low_df = pd.DataFrame(columns=["center_time", "center_pos", "confirmation_pos", "level"])
        low_df.index.name = "confirmation_time"

    return FractalSet(highs=high_df, lows=low_df)


def detect_fvgs(
    df: pd.DataFrame,
    atr: pd.Series,
    min_fvg_atr: float,
    pip_size: float,
) -> pd.DataFrame:
    """
    Detect standard 3-candle wick FVGs.

    bullish: low[t] > high[t-2]
    bearish: high[t] < low[t-2]
    """
    rows: list[dict] = []
    idx = df.index
    for pos in range(2, len(df)):
        prev2 = pos - 2
        atr_value = float(atr.iloc[pos])
        if not np.isfinite(atr_value) or atr_value <= 0:
            continue

        low_now = float(df["low"].iloc[pos])
        high_now = float(df["high"].iloc[pos])
        high_prev2 = float(df["high"].iloc[prev2])
        low_prev2 = float(df["low"].iloc[prev2])

        if low_now > high_prev2:
            gap_low = high_prev2
            gap_high = low_now
            direction = 1
        elif high_now < low_prev2:
            gap_low = high_now
            gap_high = low_prev2
            direction = -1
        else:
            continue

        size = gap_high - gap_low
        size_atr = size / atr_value
        if size_atr < min_fvg_atr:
            continue

        rows.append(
            {
                "event_time": idx[pos],
                "event_pos": pos,
                "fvg_direction": direction,
                "fvg_gap_low": gap_low,
                "fvg_gap_high": gap_high,
                "fvg_mid": (gap_low + gap_high) / 2,
                "fvg_size": size,
                "fvg_size_pips": size / pip_size,
                "fvg_size_atr": size_atr,
                "event_open": float(df["open"].iloc[pos]),
                "event_high": high_now,
                "event_low": low_now,
                "event_close": float(df["close"].iloc[pos]),
                "atr": atr_value,
            }
        )

    if not rows:
        result = pd.DataFrame(columns=["event_time"]).set_index("event_time")
        return result

    return pd.DataFrame(rows).set_index("event_time").sort_index()


def _add_decision_context(
    raw_base: pd.DataFrame,
    events: pd.DataFrame,
    decision_delay_bars: int,
) -> pd.DataFrame:
    """
    Add the first tradable bar and latest known close for each FVG event.

    Resampled bars are indexed by open time. An FVG detected on event_pos is
    only known after that candle closes, which is the next base bar open. Thus:
      decision_delay_bars=1 -> decide at event_pos + 1 using close[event_pos]
      decision_delay_bars=2 -> wait one extra closed bar and decide at event_pos + 2
    """
    if decision_delay_bars < 1:
        raise ValueError("decision_delay_bars must be >= 1 for live-safe FVG timing.")

    events = events.copy()
    n = len(raw_base)
    decision_pos = events["event_pos"].astype(int) + decision_delay_bars
    decision_close_pos = decision_pos - 1
    valid = decision_pos < n
    events = events.loc[valid].copy()
    decision_pos = decision_pos.loc[valid].astype(int)
    decision_close_pos = decision_close_pos.loc[valid].astype(int)

    events["decision_pos"] = decision_pos.to_numpy()
    events["decision_time"] = raw_base.index[decision_pos.to_numpy()]
    events["decision_close"] = raw_base["close"].iloc[decision_close_pos.to_numpy()].to_numpy()
    return events


def _nth_prior_available(
    fractals: pd.DataFrame,
    decision_time: pd.Timestamp,
    nth: int,
    timeframe_minutes: int,
) -> pd.Series | None:
    if fractals.empty:
        return None
    availability_time = fractals.index + pd.to_timedelta(timeframe_minutes, unit="min")
    prior = fractals.loc[availability_time <= decision_time]
    if len(prior) < nth:
        return None
    return prior.iloc[-nth]


def _add_fractal_context(
    events: pd.DataFrame,
    fractals: FractalSet,
    prefix: str,
    timeframe_minutes: int,
) -> pd.DataFrame:
    events = events.copy()
    for side_name, side_df in [("high", fractals.highs), ("low", fractals.lows)]:
        for nth, nth_name in [(1, "last"), (2, "second")]:
            levels = []
            ages = []
            centers = []
            for decision_time in events["decision_time"]:
                row = _nth_prior_available(side_df, decision_time, nth, timeframe_minutes)
                if row is None:
                    levels.append(np.nan)
                    ages.append(np.nan)
                    centers.append(pd.NaT)
                    continue
                levels.append(float(row["level"]))
                centers.append(row["center_time"])
                age_minutes = (decision_time - row["center_time"]).total_seconds() / 60
                ages.append(age_minutes / timeframe_minutes)

            base = f"{prefix}_{nth_name}_{side_name}"
            events[f"{base}_level"] = levels
            events[f"{base}_age_bars"] = ages
            events[f"{base}_center_time"] = centers
    return events


def _add_distance_features(events: pd.DataFrame, pip_size: float) -> pd.DataFrame:
    events = events.copy()
    level_cols = [c for c in events.columns if c.endswith("_level")]
    for col in level_cols:
        name = col[:-6]
        signed = events[col] - events["decision_close"]
        events[f"{name}_dist_signed"] = signed
        events[f"{name}_dist_signed_pips"] = signed / pip_size
        events[f"{name}_dist_signed_atr"] = signed / (events["atr"] + 1e-12)
        events[f"{name}_dist_abs_atr"] = signed.abs() / (events["atr"] + 1e-12)

    for prefix, nth_name in [("base", "last"), ("base", "second"), ("higher", "last")]:
        high_col = f"{prefix}_{nth_name}_high_level"
        low_col = f"{prefix}_{nth_name}_low_level"
        if high_col in events.columns and low_col in events.columns:
            events[f"{prefix}_{nth_name}_range_atr"] = (
                (events[high_col] - events[low_col]).abs()
                / (events["atr"] + 1e-12)
            )

    events["fvg_mid_dist_signed_atr"] = (
        (events["fvg_mid"] - events["decision_close"]) / (events["atr"] + 1e-12)
    )
    events["fvg_gap_low_dist_atr"] = (
        (events["fvg_gap_low"] - events["decision_close"]) / (events["atr"] + 1e-12)
    )
    events["fvg_gap_high_dist_atr"] = (
        (events["fvg_gap_high"] - events["decision_close"]) / (events["atr"] + 1e-12)
    )
    return events


def _add_time_features(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    idx = pd.DatetimeIndex(events["decision_time"] if "decision_time" in events else events.index)
    events["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    events["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    events["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 5)
    events["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 5)
    events["is_monday"] = (idx.dayofweek == 0).astype(int)
    events["is_friday"] = (idx.dayofweek == 4).astype(int)
    return events


def _build_engineered_features(
    raw_base: pd.DataFrame,
    norm_method: str,
    fracdiff_d: float,
    threshold: float,
    lags: list[int],
) -> pd.DataFrame:
    df_norm = normalize_prices(raw_base, method=norm_method, d=fracdiff_d, threshold=threshold)
    engineer = FeatureEngineer(lags=lags)
    # Use a harmless lag target only to reuse the existing feature block.
    engineered = engineer.transform(df_norm, target_horizons=[1], target_type="lag")
    drop_cols = [
        c for c in engineered.columns
        if c.startswith("future_") or c.startswith("direction_")
    ]
    return engineered.drop(columns=drop_cols, errors="ignore")


def _target_scan(
    raw_base: pd.DataFrame,
    events: pd.DataFrame,
    lookahead_bars: int,
    require_unbroken_levels: bool = True,
) -> pd.DataFrame:
    events = events.copy()
    high = raw_base["high"].to_numpy()
    low = raw_base["low"].to_numpy()
    n = len(raw_base)

    watched = {}
    for prefix in ["base", "higher"]:
        for nth_name in ["last", "second"]:
            for side, direction in [("high", 1), ("low", -1)]:
                name = f"{prefix}_{nth_name}_{side}"
                if f"{name}_level" in events.columns:
                    watched[name] = (side, direction)

    target_dirs = []
    target_kinds = []
    target_prices = []
    target_bars = []
    ambiguous = []
    hit_columns = {f"hit_{name}": [] for name in watched}

    for _, row in events.iterrows():
        start = int(row["decision_pos"])
        end = min(start + lookahead_bars - 1, n - 1)
        first_hits: list[tuple[int, str, int, float]] = []
        hit_flags = {name: 0 for name in watched}

        for name, (side, direction) in watched.items():
            level = row.get(f"{name}_level", np.nan)
            if not np.isfinite(level) or start > end:
                continue
            if require_unbroken_levels:
                decision_close = float(row["decision_close"])
                if side == "high" and level <= decision_close:
                    continue
                if side == "low" and level >= decision_close:
                    continue

            if side == "high":
                hits = np.flatnonzero(high[start:end + 1] >= level)
            else:
                hits = np.flatnonzero(low[start:end + 1] <= level)

            if len(hits):
                bars = int(hits[0] + 1)
                hit_flags[name] = 1
                first_hits.append((bars, name, direction, float(level)))

        for name in watched:
            hit_columns[f"hit_{name}"].append(hit_flags[name])

        if not first_hits:
            target_dirs.append(0)
            target_kinds.append("none")
            target_prices.append(np.nan)
            target_bars.append(np.nan)
            ambiguous.append(0)
            continue

        first_hits.sort(key=lambda item: item[0])
        first_bar = first_hits[0][0]
        same_bar = [hit for hit in first_hits if hit[0] == first_bar]
        directions = {hit[2] for hit in same_bar}

        if len(directions) > 1:
            target_dirs.append(0)
            target_kinds.append("ambiguous_same_bar")
            target_prices.append(np.nan)
            target_bars.append(first_bar)
            ambiguous.append(1)
        else:
            first = same_bar[0]
            target_dirs.append(first[2])
            target_kinds.append(first[1])
            target_prices.append(first[3])
            target_bars.append(first_bar)
            ambiguous.append(0)

    events["target_first_break_dir"] = target_dirs
    events["target_first_break_level_kind"] = target_kinds
    events["target_first_break_level_price"] = target_prices
    events["target_first_break_bars"] = target_bars
    events["target_is_ambiguous"] = ambiguous
    for col, values in hit_columns.items():
        events[col] = values

    return events


def get_fvg_feature_cols(
    df: pd.DataFrame,
    use_engineered_features: bool = True,
) -> list[str]:
    exclude = set(TARGET_COLUMNS)
    exclude |= {c for c in df.columns if c.startswith("target_")}
    exclude |= {c for c in df.columns if c.startswith("hit_")}
    exclude |= {c for c in df.columns if c.endswith("_center_time")}
    exclude |= {
        "event_pos",
        "event_open",
        "event_high",
        "event_low",
        "event_close",
        "decision_pos",
        "decision_close",
        "atr",
        "fvg_gap_low",
        "fvg_gap_high",
        "fvg_mid",
    }
    exclude |= {c for c in df.columns if c.endswith("_level")}

    if not use_engineered_features:
        exclude |= {c for c in df.columns if c.startswith("eng_")}

    return [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]


class FVGFractalPipeline:
    """
    Event-level pipeline for FVG/fractal prediction.

    The primary target is target_first_break_dir:
      1  -> an upper watched fractal level breaks first
      -1 -> a lower watched fractal level breaks first
      0  -> no watched level breaks first, or same-bar high/low ambiguity
    """

    def __init__(
        self,
        base_timeframe: str = "M15",
        higher_timeframe: str = "H1",
        fractal_window: int = 5,
        lookahead_bars: int | None = None,
        min_fvg_atr: float = 0.10,
        lags: list[int] | None = None,
        gap_events: int = 50,
        scaling: str = "none",
        window_size: int = 500,
        norm_method: str = "log_returns",
        fracdiff_d: float = 0.3,
        threshold: float = 6e-4,
        use_engineered_features: bool = True,
        decision_delay_bars: int = 2,
        single_timeframe: bool = False,
        require_unbroken_levels: bool = True,
    ):
        if base_timeframe not in TIMEFRAMES:
            raise ValueError(f"base_timeframe must be one of {sorted(TIMEFRAMES)}")
        if higher_timeframe not in TIMEFRAMES:
            raise ValueError(f"higher_timeframe must be one of {sorted(TIMEFRAMES)}")
        _validate_fractal_window(fractal_window)
        if min_fvg_atr < 0:
            raise ValueError("min_fvg_atr must be >= 0.")
        if decision_delay_bars < 1:
            raise ValueError("decision_delay_bars must be >= 1.")

        self.base_timeframe = base_timeframe
        self.higher_timeframe = higher_timeframe
        self.fractal_window = fractal_window
        self.lookahead_bars = (
            lookahead_bars
            if lookahead_bars is not None
            else int(24 * 60 / _timeframe_minutes(base_timeframe))
        )
        self.min_fvg_atr = min_fvg_atr
        self.lags = lags or [1, 2, 5, 10]
        self.gap_events = gap_events
        self.scaling = scaling
        self.window_size = window_size
        self.norm_method = norm_method
        self.fracdiff_d = fracdiff_d
        self.threshold = threshold
        self.use_engineered_features = use_engineered_features
        self.decision_delay_bars = decision_delay_bars
        self.single_timeframe = single_timeframe
        self.require_unbroken_levels = require_unbroken_levels
        self.scaler = self._make_scaler()

    def _make_scaler(self):
        if self.scaling == "rolling":
            return RollingScaler(window_size=self.window_size)
        if self.scaling == "global":
            return ForexScaler()
        if self.scaling in {"none", None}:
            return None
        raise ValueError("scaling must be 'none', 'global', or 'rolling'.")

    def _resample(self, df_m1: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        if timeframe == "M1":
            return df_m1.copy()
        return resample_ohlcv(df_m1, TIMEFRAMES[timeframe])

    def _split_events(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        n = len(df)
        train_end = int(n * 0.6)
        val_start = min(train_end + self.gap_events, n)
        val_end = min(val_start + int(n * 0.2), n)
        test_start = min(val_end + self.gap_events, n)
        train = df.iloc[:train_end].copy()
        val = df.iloc[val_start:val_end].copy()
        test = df.iloc[test_start:].copy()
        print(f"\n[Splitter] Event split (gap={self.gap_events} events):")
        for name, split in [("Train", train), ("Val", val), ("Test", test)]:
            if split.empty:
                print(f"  {name:<5}:       0 events")
            else:
                print(
                    f"  {name:<5}: {len(split):>7,} events  "
                    f"{split.index[0].date()} -> {split.index[-1].date()}"
                )
        return train, val, test

    def run(self, df_m1: pd.DataFrame) -> dict:
        pair = df_m1.attrs.get("pair", "UNKNOWN")
        pip = _pip_size(pair)
        print(f"\n{'=' * 70}")
        print(
            "  FVG/Fractal Pipeline: "
            f"{pair} | base={self.base_timeframe} "
            f"| higher={'off' if self.single_timeframe else self.higher_timeframe} "
            f"| fractal={self.fractal_window} | decision_delay={self.decision_delay_bars} "
            f"| lookahead={self.lookahead_bars}"
        )
        print(f"{'=' * 70}")

        raw_base = self._resample(df_m1, self.base_timeframe).dropna()
        raw_higher = (
            pd.DataFrame()
            if self.single_timeframe
            else self._resample(df_m1, self.higher_timeframe).dropna()
        )
        base_minutes = _timeframe_minutes(self.base_timeframe)
        higher_minutes = _timeframe_minutes(self.higher_timeframe)
        higher_msg = "off" if self.single_timeframe else f"{len(raw_higher):,} bars"
        print(f"[Resample] base={len(raw_base):,} bars | higher={higher_msg}")

        base_atr = _atr(raw_base)
        events = detect_fvgs(raw_base, base_atr, self.min_fvg_atr, pip)
        if events.empty:
            raise ValueError("No FVG events found. Lower min_fvg_atr or check input data.")
        print(f"[FVG] {len(events):,} events after min_fvg_atr={self.min_fvg_atr}")
        events = _add_decision_context(raw_base, events, self.decision_delay_bars)
        if events.empty:
            raise ValueError("No FVG events remain after applying decision_delay_bars.")

        base_fractals = detect_confirmed_fractals(raw_base, self.fractal_window)
        if self.single_timeframe:
            higher_fractals = FractalSet(highs=pd.DataFrame(), lows=pd.DataFrame())
            print(
                "[Fractals] "
                f"base high/low={len(base_fractals.highs):,}/{len(base_fractals.lows):,} | "
                "higher=off"
            )
        else:
            higher_fractals = detect_confirmed_fractals(raw_higher, self.fractal_window)
            print(
                "[Fractals] "
                f"base high/low={len(base_fractals.highs):,}/{len(base_fractals.lows):,} | "
                f"higher high/low={len(higher_fractals.highs):,}/{len(higher_fractals.lows):,}"
            )

        events = _add_fractal_context(events, base_fractals, "base", base_minutes)
        if not self.single_timeframe:
            events = _add_fractal_context(events, higher_fractals, "higher", higher_minutes)
        events = _add_distance_features(events, pip)
        events = _add_time_features(events)
        events = _target_scan(
            raw_base,
            events,
            self.lookahead_bars,
            require_unbroken_levels=self.require_unbroken_levels,
        )

        if self.use_engineered_features:
            engineered = _build_engineered_features(
                raw_base=raw_base,
                norm_method=self.norm_method,
                fracdiff_d=self.fracdiff_d,
                threshold=self.threshold,
                lags=self.lags,
            ).add_prefix("eng_")
            events = events.join(engineered, how="left")

        events = events.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["base_last_high_level", "base_last_low_level"]
        )
        feature_cols = get_fvg_feature_cols(events, self.use_engineered_features)
        events = events.dropna(subset=feature_cols)
        structure_feature_cols = [c for c in feature_cols if not c.startswith("eng_")]

        print(
            f"[Features] {len(feature_cols)} features "
            f"({len(structure_feature_cols)} structure, "
            f"{len(feature_cols) - len(structure_feature_cols)} engineered) | "
            f"{len(events):,} clean events"
        )
        print("[Targets] target_first_break_dir counts:")
        print(events["target_first_break_dir"].value_counts(dropna=False).sort_index().to_string())

        train_raw, val_raw, test_raw = self._split_events(events)

        if self.scaler is None:
            train = train_raw.copy()
            val = val_raw.copy()
            test = test_raw.copy()
        elif isinstance(self.scaler, RollingScaler):
            train = self.scaler.transform(train_raw, feature_cols)
            val = self.scaler.transform(val_raw, feature_cols)
            test = self.scaler.transform(test_raw, feature_cols)
        else:
            train = self.scaler.fit_transform(train_raw, feature_cols)
            val = self.scaler.transform(val_raw)
            test = self.scaler.transform(test_raw)

        return {
            "pair": pair,
            "base_timeframe": self.base_timeframe,
            "higher_timeframe": self.higher_timeframe,
            "events": events,
            "feature_cols": feature_cols,
            "structure_feature_cols": structure_feature_cols,
            "target_cols": [
                "target_first_break_dir",
                "target_first_break_level_kind",
                "target_first_break_level_price",
                "target_first_break_bars",
            ],
            "aux_target_cols": [c for c in events.columns if c.startswith("hit_")],
            "train_raw": train_raw,
            "val_raw": val_raw,
            "test_raw": test_raw,
            "train": train,
            "val": val,
            "test": test,
            "raw_base": raw_base,
            "raw_higher": raw_higher,
            "base_fractals": base_fractals,
            "higher_fractals": higher_fractals,
            "scaler": self.scaler,
            "norm_method": self.norm_method,
            "use_engineered_features": self.use_engineered_features,
            "min_fvg_atr": self.min_fvg_atr,
            "fractal_window": self.fractal_window,
            "lookahead_bars": self.lookahead_bars,
            "decision_delay_bars": self.decision_delay_bars,
            "single_timeframe": self.single_timeframe,
            "require_unbroken_levels": self.require_unbroken_levels,
        }

    def get_xy(
        self,
        split: pd.DataFrame,
        target: str,
        feature_cols: list[str],
        drop_timeout: bool = False,
        binary_direction: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        data = split
        if drop_timeout:
            data = data[data[target] != 0]
        X = data[feature_cols].to_numpy()
        y = data[target].to_numpy()
        if binary_direction:
            if target != "target_first_break_dir":
                warnings.warn("binary_direction=True is intended for target_first_break_dir.")
            y = (y == 1).astype(int)
        return X, y


if __name__ == "__main__":
    loader = ForexDataLoader()
    df = loader.generate_synthetic("EURUSD", n_bars=50_000)
    pipeline = FVGFractalPipeline(
        base_timeframe="M15",
        higher_timeframe="H1",
        fractal_window=5,
        min_fvg_atr=0.0,
        use_engineered_features=False,
    )
    result = pipeline.run(df)
    X_train, y_train = pipeline.get_xy(
        result["train"],
        "target_first_break_dir",
        result["feature_cols"],
        drop_timeout=True,
        binary_direction=True,
    )
    print(f"Smoke X={X_train.shape} y={y_train.shape}")
