from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import CostConfig, DecisionParams, RiskConfig


@dataclass
class BacktestResult:
    metrics: dict
    trades: pd.DataFrame
    equity_curve: pd.DataFrame


@dataclass
class Position:
    direction: int
    entry_time: pd.Timestamp
    source_time: pd.Timestamp | None
    entry_i: int
    entry_price: float
    units: float
    stop_price: float
    take_price: float
    stop_pips: float
    take_pips: float
    level_basis: str
    take_source: str
    stop_source: str
    raw_take_pips: float
    raw_stop_pips: float
    fvg_size_pips: float
    fvg_size_atr: float
    signal_atr: float
    p_entry_long: float
    p_entry_short: float
    predicted_long_net_pips: float
    predicted_short_net_pips: float
    predicted_long_net_pips_std: float


def infer_periods_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 3:
        return 252.0
    deltas = pd.Series(index).diff().dropna().dt.total_seconds() / 60
    minutes = float(deltas[deltas > 0].median())
    if not np.isfinite(minutes) or minutes <= 0:
        return 252.0
    return 252.0 * 24.0 * 60.0 / minutes


def _expected_edge_pips(
    direction: int,
    p_long: float,
    p_short: float,
    stop_pips: float,
    take_pips: float,
    costs: CostConfig,
) -> float:
    denom = max(p_long + p_short, 1e-12)
    p_win = p_long / denom if direction == 1 else p_short / denom
    return p_win * take_pips - (1.0 - p_win) * stop_pips - costs.round_trip_pips


def _entry_direction(
    signal: pd.Series,
    levels: dict[int, dict[str, float]],
    params: DecisionParams,
    costs: CostConfig,
) -> int:
    p_long = float(signal["p_long"])
    p_short = float(signal["p_short"])
    p_hold = float(signal.get("p_hold", 0.0))
    predicted_long_net_pips = signal.get("predicted_long_net_pips", np.nan)
    predicted_short_net_pips = signal.get("predicted_short_net_pips", np.nan)
    predicted_long_net_pips_std = signal.get("predicted_long_net_pips_std", np.nan)

    candidates: list[tuple[int, float, float]] = []
    if params.trade_side in {"both", "long"} and (
        p_long >= params.entry_threshold
        and p_long - max(p_short, p_hold) >= params.min_conf_gap
        and 1 in levels
    ):
        if params.min_predicted_net_pips > -998:
            try:
                pred_value = float(predicted_long_net_pips)
                pred_std = float(predicted_long_net_pips_std)
            except (TypeError, ValueError):
                pred_value = float("nan")
                pred_std = float("nan")
            if (
                not np.isfinite(pred_value)
                or pred_value < params.min_predicted_net_pips
                or (
                    params.max_prediction_std_pips < 998
                    and (not np.isfinite(pred_std) or pred_std > params.max_prediction_std_pips)
                )
            ):
                pass
            else:
                stop_pips = levels[1]["stop_pips"]
                take_pips = levels[1]["take_pips"]
                edge = _expected_edge_pips(1, p_long, p_short, stop_pips, take_pips, costs)
                if edge >= params.min_edge_pips:
                    candidates.append((1, edge, p_long))
        else:
            stop_pips = levels[1]["stop_pips"]
            take_pips = levels[1]["take_pips"]
            edge = _expected_edge_pips(1, p_long, p_short, stop_pips, take_pips, costs)
            if edge >= params.min_edge_pips:
                candidates.append((1, edge, p_long))

    if params.trade_side in {"both", "short"} and (
        p_short >= params.entry_threshold
        and p_short - max(p_long, p_hold) >= params.min_conf_gap
        and -1 in levels
    ):
        if params.min_predicted_net_pips > -998:
            try:
                pred_value = float(predicted_short_net_pips)
            except (TypeError, ValueError):
                pred_value = float("nan")
            if not np.isfinite(pred_value) or pred_value < params.min_predicted_net_pips:
                pass
            else:
                stop_pips = levels[-1]["stop_pips"]
                take_pips = levels[-1]["take_pips"]
                edge = _expected_edge_pips(-1, p_long, p_short, stop_pips, take_pips, costs)
                if edge >= params.min_edge_pips:
                    candidates.append((-1, edge, p_short))
        else:
            stop_pips = levels[-1]["stop_pips"]
            take_pips = levels[-1]["take_pips"]
            edge = _expected_edge_pips(-1, p_long, p_short, stop_pips, take_pips, costs)
            if edge >= params.min_edge_pips:
                candidates.append((-1, edge, p_short))

    if not candidates:
        return 0
    candidates.sort(key=lambda item: (item[1], item[2]), reverse=True)
    if len(candidates) > 1 and np.isclose(candidates[0][1], candidates[1][1]):
        return 0
    return candidates[0][0]


def _should_exit_on_signal(position: Position, signal: pd.Series, params: DecisionParams) -> bool:
    p_long = float(signal["p_long"])
    p_short = float(signal["p_short"])
    if position.direction == 1:
        return p_short >= params.exit_threshold or p_long <= params.exit_floor
    return p_long >= params.exit_threshold or p_short <= params.exit_floor


def _session_allows_entry(ts: pd.Timestamp, params: DecisionParams) -> bool:
    start = int(params.session_start_hour)
    end = int(params.session_end_hour)
    excluded = str(params.excluded_hours).strip()
    if excluded:
        try:
            excluded_hours = {int(item.strip()) for item in excluded.split(",") if item.strip()}
        except ValueError as exc:
            raise ValueError("excluded_hours must be empty or comma-separated hour numbers") from exc
        if any(hour < 0 or hour > 23 for hour in excluded_hours):
            raise ValueError("excluded_hours values must be in 0..23")
        if int(ts.hour) in excluded_hours:
            return False
    if start < 0 or end < 0 or start == end:
        return True
    if not (0 <= start <= 23 and 0 <= end <= 23):
        raise ValueError("session_start_hour and session_end_hour must be -1 or in 0..23")
    hour = int(ts.hour)
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


def _month_allows_entry(ts: pd.Timestamp, params: DecisionParams) -> bool:
    value = str(params.allowed_months).strip()
    if not value:
        return True
    try:
        months = {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as exc:
        raise ValueError("allowed_months must be empty or comma-separated month numbers") from exc
    if not months:
        return True
    if any(month < 1 or month > 12 for month in months):
        raise ValueError("allowed_months values must be in 1..12")
    return int(ts.month) in months


def _weekday_allows_entry(ts: pd.Timestamp, params: DecisionParams) -> bool:
    value = str(params.excluded_weekdays).strip()
    if not value:
        return True
    try:
        weekdays = {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as exc:
        raise ValueError("excluded_weekdays must be empty or comma-separated weekday numbers") from exc
    if any(day < 0 or day > 6 for day in weekdays):
        raise ValueError("excluded_weekdays values must be in 0..6")
    return int(ts.weekday()) not in weekdays


def _trend_allows_entry(bar: pd.Series, signal: pd.Series | None, params: DecisionParams) -> bool:
    source = signal if signal is not None else bar
    ema200_dist = float(
        source.get(
            "source_ema200_dist_pips",
            source.get("ema200_dist_pips", bar.get("ema200_dist_pips", 0.0)),
        )
    )
    ema50_slope = float(
        source.get(
            "source_ema50_slope_pips",
            source.get("ema50_slope_pips", bar.get("ema50_slope_pips", 0.0)),
        )
    )
    if not np.isfinite(ema200_dist) or not np.isfinite(ema50_slope):
        return False
    return (
        params.min_ema200_dist_pips <= ema200_dist <= params.max_ema200_dist_pips
        and params.min_ema50_slope_pips <= ema50_slope <= params.max_ema50_slope_pips
    )


def _rolling_pnl_allows_entry(closed_pnls: list[float], params: DecisionParams) -> bool:
    window = int(params.rolling_pnl_window)
    if window <= 0:
        return True
    if len(closed_pnls) < window:
        return True
    recent = float(np.sum(closed_pnls[-window:]))
    return recent >= float(params.min_rolling_pnl)


def _position_units(
    equity: float,
    price: float,
    stop_distance: float,
    params: DecisionParams,
    risk: RiskConfig,
) -> float:
    if equity <= 0 or price <= 0 or stop_distance <= 0:
        return 0.0
    risk_units = equity * params.risk_per_trade / stop_distance
    leverage_units = equity * risk.max_leverage / price
    return max(0.0, min(risk_units, leverage_units))


def _finite_price(value: object) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) and value > 0 else None


def _scaled_level(
    entry_price: float,
    level_price: float | None,
    direction: int,
    scale: float,
    is_take: bool,
) -> float | None:
    if level_price is None:
        return None
    distance = abs(level_price - entry_price) * scale
    if distance <= 0:
        return None
    price = entry_price + direction * distance if is_take else entry_price - direction * distance
    if direction == 1 and is_take and price <= entry_price:
        return None
    if direction == 1 and not is_take and price >= entry_price:
        return None
    if direction == -1 and is_take and price >= entry_price:
        return None
    if direction == -1 and not is_take and price <= entry_price:
        return None
    return price


def _candidate_levels(
    signal: pd.Series,
    open_price: float,
    atr: float,
    pip_size: float,
    params: DecisionParams,
) -> dict[int, dict[str, float]]:
    levels: dict[int, dict[str, float]] = {}
    fvg_size_atr = float(signal.get("fvg_size_atr", np.nan))
    if params.min_fvg_size_atr > 0 and (not np.isfinite(fvg_size_atr) or fvg_size_atr < params.min_fvg_size_atr):
        return levels
    signal_atr = float(signal.get("signal_atr", atr))
    signal_atr_pips = signal_atr / pip_size if np.isfinite(signal_atr) and pip_size > 0 else np.nan
    if (
        not np.isfinite(signal_atr_pips)
        or signal_atr_pips < params.min_signal_atr_pips
        or signal_atr_pips > params.max_signal_atr_pips
    ):
        return levels
    mode = params.level_mode
    for direction, prefix in [(1, "long"), (-1, "short")]:
        basis = "atr_multiple"
        take_price = None
        stop_price = None
        if mode == "model":
            raw_take = _finite_price(signal.get(f"{prefix}_take_price"))
            raw_stop = _finite_price(signal.get(f"{prefix}_stop_price"))
            take_price = _scaled_level(open_price, raw_take, direction, params.take_atr, True)
            stop_price = _scaled_level(open_price, raw_stop, direction, params.stop_atr, False)
            if take_price is not None and stop_price is not None:
                basis = str(signal.get("level_basis", "model_levels"))

        if take_price is None or stop_price is None:
            if mode == "model":
                continue
            level_atr = signal_atr if np.isfinite(signal_atr) and signal_atr > 0 else atr
            if not np.isfinite(level_atr) or level_atr <= 0:
                continue
            stop_distance = level_atr * params.stop_atr
            take_distance = level_atr * params.take_atr
            stop_price = open_price - direction * stop_distance
            take_price = open_price + direction * take_distance
            basis = "atr_multiple_fallback" if mode == "model" else "atr_multiple"

        stop_distance = abs(open_price - stop_price)
        take_distance = abs(take_price - open_price)
        if stop_distance <= 0 or take_distance <= 0:
            continue
        levels[direction] = {
            "stop_price": float(stop_price),
            "take_price": float(take_price),
            "stop_distance": float(stop_distance),
            "take_distance": float(take_distance),
            "stop_pips": float(stop_distance / pip_size),
            "take_pips": float(take_distance / pip_size),
            "raw_stop_pips": (
                float(abs(open_price - raw_stop) / pip_size)
                if mode == "model" and raw_stop is not None else float(stop_distance / pip_size)
            ),
            "raw_take_pips": (
                float(abs(raw_take - open_price) / pip_size)
                if mode == "model" and raw_take is not None else float(take_distance / pip_size)
            ),
            "level_basis": basis,
            "take_source": str(signal.get(f"{prefix}_take_source", basis)),
            "stop_source": str(signal.get(f"{prefix}_stop_source", basis)),
            "fvg_size_pips": float(signal.get("fvg_size_pips", np.nan)),
            "fvg_size_atr": float(signal.get("fvg_size_atr", np.nan)),
            "signal_atr": float(signal.get("signal_atr", atr)),
        }
    return levels


def _close_position(
    position: Position,
    exit_time: pd.Timestamp,
    exit_price: float,
    reason: str,
    costs: CostConfig,
    pip_size: float,
    equity_before: float,
) -> dict:
    gross_pips = position.direction * (exit_price - position.entry_price) / pip_size
    net_pips = gross_pips - costs.round_trip_pips
    pnl = net_pips * pip_size * position.units
    return {
        "entry_time": position.entry_time,
        "source_time": position.source_time,
        "exit_time": exit_time,
        "direction": "long" if position.direction == 1 else "short",
        "entry_price": position.entry_price,
        "exit_price": exit_price,
        "units": position.units,
        "gross_pips": gross_pips,
        "cost_pips": costs.round_trip_pips,
        "net_pips": net_pips,
        "pnl": pnl,
        "return_on_equity": pnl / equity_before if equity_before else 0.0,
        "reason": reason,
        "stop_pips": position.stop_pips,
        "take_pips": position.take_pips,
        "level_basis": position.level_basis,
        "take_source": position.take_source,
        "stop_source": position.stop_source,
        "raw_stop_pips": position.raw_stop_pips,
        "raw_take_pips": position.raw_take_pips,
        "fvg_size_pips": position.fvg_size_pips,
        "fvg_size_atr": position.fvg_size_atr,
        "signal_atr": position.signal_atr,
        "p_entry_long": position.p_entry_long,
        "p_entry_short": position.p_entry_short,
        "predicted_long_net_pips": position.predicted_long_net_pips,
        "predicted_short_net_pips": position.predicted_short_net_pips,
        "predicted_long_net_pips_std": position.predicted_long_net_pips_std,
    }


def simulate(
    market: pd.DataFrame,
    signals: pd.DataFrame,
    params: DecisionParams,
    costs: CostConfig | None = None,
    risk: RiskConfig | None = None,
) -> BacktestResult:
    costs = costs or CostConfig()
    risk = risk or RiskConfig()
    market = market.sort_index()
    signals = signals.sort_index()

    required = {"open", "high", "low", "close", "atr", "pip_size"}
    missing = required - set(market.columns)
    if missing:
        raise ValueError(f"market is missing columns: {sorted(missing)}")

    equity = risk.initial_equity
    peak_equity = equity
    daily_start_equity = equity
    current_day = None
    trades_opened_today = 0
    cooldown_until_i = -1
    stop_trading = False
    position: Position | None = None
    trades: list[dict] = []
    closed_pnls: list[float] = []
    curve: list[dict] = []

    signal_by_time = {idx: row for idx, row in signals.iterrows()}
    rows = list(market.iterrows())

    for i, (ts, bar) in enumerate(rows):
        day = ts.date()
        if day != current_day:
            current_day = day
            daily_start_equity = equity
            trades_opened_today = 0

        signal = signal_by_time.get(ts)
        pip_size = float(bar["pip_size"])
        open_price = float(bar["open"])
        high_price = float(bar["high"])
        low_price = float(bar["low"])
        close_price = float(bar["close"])
        atr = float(bar["atr"])

        if position is not None and signal is not None and _should_exit_on_signal(position, signal, params):
            trade = _close_position(
                position, ts, open_price, "opposite_or_weak_signal",
                costs, pip_size, equity,
            )
            equity += trade["pnl"]
            trades.append(trade)
            closed_pnls.append(float(trade["pnl"]))
            cooldown_until_i = i + params.cooldown_bars
            position = None

        daily_loss = (daily_start_equity - equity) / daily_start_equity if daily_start_equity else 0.0
        drawdown = (peak_equity - equity) / peak_equity if peak_equity else 0.0
        block_new_entries = (
            stop_trading
            or daily_loss >= risk.daily_loss_stop
            or drawdown >= risk.max_drawdown_stop
            or i < cooldown_until_i
            or (params.max_trades_per_day > 0 and trades_opened_today >= params.max_trades_per_day)
            or not _session_allows_entry(ts, params)
            or not _month_allows_entry(ts, params)
            or not _weekday_allows_entry(ts, params)
            or not _trend_allows_entry(bar, signal, params)
            or not _rolling_pnl_allows_entry(closed_pnls, params)
        )

        if position is None and signal is not None and not block_new_entries and np.isfinite(atr) and atr > 0:
            levels = _candidate_levels(signal, open_price, atr, pip_size, params)
            direction = _entry_direction(signal, levels, params, costs)
            selected = levels.get(direction)
            units = (
                _position_units(equity, open_price, selected["stop_distance"], params, risk)
                if selected else 0.0
            )
            if direction and units > 0:
                trades_opened_today += 1
                position = Position(
                    direction=direction,
                    entry_time=ts,
                    source_time=pd.Timestamp(signal["source_time"]) if "source_time" in signal else None,
                    entry_i=i,
                    entry_price=open_price,
                    units=units,
                    stop_price=selected["stop_price"],
                    take_price=selected["take_price"],
                    stop_pips=selected["stop_pips"],
                    take_pips=selected["take_pips"],
                    level_basis=selected["level_basis"],
                    take_source=selected["take_source"],
                    stop_source=selected["stop_source"],
                    raw_take_pips=selected["raw_take_pips"],
                    raw_stop_pips=selected["raw_stop_pips"],
                    fvg_size_pips=selected["fvg_size_pips"],
                    fvg_size_atr=selected["fvg_size_atr"],
                    signal_atr=selected["signal_atr"],
                    p_entry_long=float(signal["p_long"]),
                    p_entry_short=float(signal["p_short"]),
                    predicted_long_net_pips=float(signal.get("predicted_long_net_pips", np.nan)),
                    predicted_short_net_pips=float(signal.get("predicted_short_net_pips", np.nan)),
                    predicted_long_net_pips_std=float(signal.get("predicted_long_net_pips_std", np.nan)),
                )

        if position is not None:
            exit_price = None
            reason = ""
            if position.direction == 1:
                hit_stop = low_price <= position.stop_price
                hit_take = high_price >= position.take_price
            else:
                hit_stop = high_price >= position.stop_price
                hit_take = low_price <= position.take_price

            if hit_stop and hit_take:
                exit_price = position.stop_price
                reason = "same_bar_stop_before_take"
            elif hit_stop:
                exit_price = position.stop_price
                reason = "stop_loss"
            elif hit_take:
                exit_price = position.take_price
                reason = "take_profit"
            elif i - position.entry_i + 1 >= params.max_hold_bars:
                exit_price = close_price
                reason = "max_hold"

            if exit_price is not None:
                trade = _close_position(
                    position, ts, float(exit_price), reason, costs, pip_size, equity
                )
                equity += trade["pnl"]
                trades.append(trade)
                closed_pnls.append(float(trade["pnl"]))
                cooldown_until_i = i + params.cooldown_bars
                position = None

        mark_equity = equity
        if position is not None:
            unrealized = (
                position.direction
                * (close_price - position.entry_price)
                * position.units
            )
            estimated_exit_cost = costs.round_trip_pips * pip_size * position.units
            mark_equity = equity + unrealized - estimated_exit_cost

        peak_equity = max(peak_equity, mark_equity)
        mark_drawdown = (peak_equity - mark_equity) / peak_equity if peak_equity else 0.0
        if mark_drawdown >= risk.max_drawdown_stop:
            stop_trading = True

        curve.append(
            {
                "time": ts,
                "equity": mark_equity,
                "realized_equity": equity,
                "drawdown": mark_drawdown,
                "open_position": 0 if position is None else position.direction,
            }
        )

    if position is not None and rows:
        ts, bar = rows[-1]
        trade = _close_position(
            position,
            ts,
            float(bar["close"]),
            "end_of_data",
            costs,
            float(bar["pip_size"]),
            equity,
        )
        equity += trade["pnl"]
        trades.append(trade)
        closed_pnls.append(float(trade["pnl"]))
        if curve and curve[-1]["time"] == ts:
            peak_equity = max(peak_equity, equity)
            curve[-1].update(
                {
                    "equity": equity,
                    "realized_equity": equity,
                    "drawdown": (peak_equity - equity) / peak_equity if peak_equity else 0.0,
                    "open_position": 0,
                }
            )

    trades_df = pd.DataFrame(trades)
    curve_df = pd.DataFrame(curve).drop_duplicates("time", keep="last")
    if not curve_df.empty:
        curve_df = curve_df.set_index("time")

    metrics = calculate_metrics(curve_df, trades_df, risk.initial_equity, market.index)
    metrics.update(
        {
            "round_trip_cost_pips": costs.round_trip_pips,
            "params": asdict(params),
            "risk": asdict(risk),
            "costs": asdict(costs),
        }
    )
    return BacktestResult(metrics=metrics, trades=trades_df, equity_curve=curve_df)


def calculate_metrics(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    initial_equity: float,
    market_index: pd.DatetimeIndex,
) -> dict:
    if equity_curve.empty:
        return {
            "net_profit": 0.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "trade_count": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_trade_pnl": 0.0,
            "total_cost_pips": 0.0,
        }

    final_equity = float(equity_curve["realized_equity"].iloc[-1])
    net_profit = final_equity - initial_equity
    total_return = net_profit / initial_equity if initial_equity else 0.0
    max_drawdown = float(equity_curve["drawdown"].max())

    returns = equity_curve["equity"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    periods = infer_periods_per_year(market_index)
    if returns.std(ddof=0) > 0:
        sharpe = float(returns.mean() / returns.std(ddof=0) * np.sqrt(periods))
    else:
        sharpe = 0.0
    downside = returns[returns < 0]
    if len(downside) and downside.std(ddof=0) > 0:
        sortino = float(returns.mean() / downside.std(ddof=0) * np.sqrt(periods))
    else:
        sortino = 0.0

    trade_count = int(len(trades))
    if trade_count:
        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] < 0]
        win_rate = float(len(wins) / trade_count)
        gross_profit = float(wins["pnl"].sum())
        gross_loss = float(-losses["pnl"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        avg_trade = float(trades["pnl"].mean())
        total_cost_pips = float(trades["cost_pips"].sum())
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade = 0.0
        total_cost_pips = 0.0

    calmar = total_return / max_drawdown if max_drawdown > 0 else 0.0
    return {
        "net_profit": float(net_profit),
        "total_return": float(total_return),
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": float(calmar),
        "trade_count": trade_count,
        "win_rate": win_rate,
        "profit_factor": float(profit_factor),
        "avg_trade_pnl": avg_trade,
        "total_cost_pips": total_cost_pips,
    }
