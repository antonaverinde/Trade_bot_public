from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CostConfig


@dataclass(frozen=True)
class LevelChoice:
    price: float
    source: str


@dataclass(frozen=True)
class TradeOutcome:
    net_pips: float
    gross_pips: float
    reason: str
    bars_held: int


def _finite(value: object) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) and value > 0 else None


def _nearest_level(
    row: pd.Series,
    candidates: list[tuple[str, str]],
    ref: float,
    above: bool,
) -> LevelChoice | None:
    best: LevelChoice | None = None
    for col, source in candidates:
        value = _finite(row.get(col))
        if value is None:
            continue
        if above and value <= ref:
            continue
        if not above and value >= ref:
            continue
        if best is None:
            best = LevelChoice(value, source)
        elif above and value < best.price:
            best = LevelChoice(value, source)
        elif not above and value > best.price:
            best = LevelChoice(value, source)
    return best


def fvg_level_candidates(
    row: pd.Series,
    ref: float | None = None,
    allowed_sources: set[str] | None = None,
    use_gap_fallback: bool = True,
) -> dict[str, LevelChoice | None]:
    ref = float(row["decision_close"] if ref is None else ref)
    high_candidates_all = [
        ("base_last_high_level", "base_last"),
        ("base_second_high_level", "base_second"),
        ("higher_last_high_level", "higher_last"),
        ("higher_second_high_level", "higher_second"),
    ]
    low_candidates_all = [
        ("base_last_low_level", "base_last"),
        ("base_second_low_level", "base_second"),
        ("higher_last_low_level", "higher_last"),
        ("higher_second_low_level", "higher_second"),
    ]
    if allowed_sources is None:
        high_candidates = high_candidates_all
        low_candidates = low_candidates_all
    else:
        high_candidates = [item for item in high_candidates_all if item[1] in allowed_sources]
        low_candidates = [item for item in low_candidates_all if item[1] in allowed_sources]
    high = _nearest_level(row, high_candidates, ref, above=True)
    low = _nearest_level(row, low_candidates, ref, above=False)

    gap_low = _finite(row.get("fvg_gap_low"))
    if use_gap_fallback and low is None and gap_low is not None and gap_low < ref:
        low = LevelChoice(gap_low, "fvg_gap_fallback")

    gap_high = _finite(row.get("fvg_gap_high"))
    if use_gap_fallback and high is None and gap_high is not None and gap_high > ref:
        high = LevelChoice(gap_high, "fvg_gap_fallback")

    return {"high": high, "low": low}


def simulate_fixed_levels(
    raw_base: pd.DataFrame,
    entry_pos: int,
    direction: int,
    stop_price: float,
    take_price: float,
    max_hold_bars: int,
    costs: CostConfig,
    pip_size: float,
) -> TradeOutcome | None:
    if direction not in {-1, 1}:
        raise ValueError("direction must be -1 or 1")
    if entry_pos < 0 or entry_pos >= len(raw_base):
        return None
    if max_hold_bars < 1:
        raise ValueError("max_hold_bars must be >= 1")

    entry_price = float(raw_base["open"].iloc[entry_pos])
    end_pos = min(entry_pos + max_hold_bars - 1, len(raw_base) - 1)
    exit_price = float(raw_base["close"].iloc[end_pos])
    reason = "max_hold"
    bars_held = end_pos - entry_pos + 1

    for pos in range(entry_pos, end_pos + 1):
        high = float(raw_base["high"].iloc[pos])
        low = float(raw_base["low"].iloc[pos])
        if direction == 1:
            hit_stop = low <= stop_price
            hit_take = high >= take_price
        else:
            hit_stop = high >= stop_price
            hit_take = low <= take_price

        if hit_stop and hit_take:
            exit_price = stop_price
            reason = "same_bar_stop_before_take"
        elif hit_stop:
            exit_price = stop_price
            reason = "stop_loss"
        elif hit_take:
            exit_price = take_price
            reason = "take_profit"
        else:
            continue
        bars_held = pos - entry_pos + 1
        break

    gross_pips = direction * (exit_price - entry_price) / pip_size
    net_pips = gross_pips - costs.round_trip_pips
    return TradeOutcome(
        net_pips=float(net_pips),
        gross_pips=float(gross_pips),
        reason=reason,
        bars_held=int(bars_held),
    )


def add_fvg_profit_labels(
    events: pd.DataFrame,
    raw_base: pd.DataFrame,
    costs: CostConfig,
    pip_size: float,
    max_hold_bars: int,
    profit_buffer_pips: float,
    allowed_sources: set[str] | None = None,
    use_gap_fallback: bool = True,
    tie_buffer_pips: float = 0.5,
) -> pd.DataFrame:
    events = events.copy()
    long_net = []
    short_net = []
    long_reason = []
    short_reason = []
    long_take_source = []
    long_stop_source = []
    short_take_source = []
    short_stop_source = []
    labels = []

    for _, row in events.iterrows():
        levels = fvg_level_candidates(
            row,
            allowed_sources=allowed_sources,
            use_gap_fallback=use_gap_fallback,
        )
        high = levels["high"]
        low = levels["low"]
        entry_pos = int(row["decision_pos"])

        long_out = None
        short_out = None
        if high is not None and low is not None:
            long_out = simulate_fixed_levels(
                raw_base,
                entry_pos,
                1,
                low.price,
                high.price,
                max_hold_bars,
                costs,
                pip_size,
            )
            short_out = simulate_fixed_levels(
                raw_base,
                entry_pos,
                -1,
                high.price,
                low.price,
                max_hold_bars,
                costs,
                pip_size,
            )

        l_net = long_out.net_pips if long_out is not None else np.nan
        s_net = short_out.net_pips if short_out is not None else np.nan
        long_net.append(l_net)
        short_net.append(s_net)
        long_reason.append(long_out.reason if long_out is not None else "no_levels")
        short_reason.append(short_out.reason if short_out is not None else "no_levels")
        long_take_source.append(high.source if high is not None else "")
        long_stop_source.append(low.source if low is not None else "")
        short_take_source.append(low.source if low is not None else "")
        short_stop_source.append(high.source if high is not None else "")

        long_ok = np.isfinite(l_net) and l_net >= profit_buffer_pips
        short_ok = np.isfinite(s_net) and s_net >= profit_buffer_pips
        if long_ok and not short_ok:
            labels.append(1)
        elif short_ok and not long_ok:
            labels.append(2)
        elif long_ok and short_ok:
            if l_net >= s_net + tie_buffer_pips:
                labels.append(1)
            elif s_net >= l_net + tie_buffer_pips:
                labels.append(2)
            else:
                labels.append(0)
        else:
            labels.append(0)

    events["target_profit_label"] = labels
    events["target_long_net_pips"] = long_net
    events["target_short_net_pips"] = short_net
    events["target_long_reason"] = long_reason
    events["target_short_reason"] = short_reason
    events["target_long_take_source"] = long_take_source
    events["target_long_stop_source"] = long_stop_source
    events["target_short_take_source"] = short_take_source
    events["target_short_stop_source"] = short_stop_source
    events["target_profit_buffer_pips"] = float(profit_buffer_pips)
    events["target_profit_max_hold_bars"] = int(max_hold_bars)
    return events
