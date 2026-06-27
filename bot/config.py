from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CostConfig:
    """Retail-FX transaction cost assumptions, expressed in pips."""

    spread_pips: float = 1.0
    slippage_pips_per_side: float = 0.2
    commission_pips_per_side: float = 0.0

    @property
    def round_trip_pips(self) -> float:
        return (
            self.spread_pips
            + 2 * self.slippage_pips_per_side
            + 2 * self.commission_pips_per_side
        )


@dataclass(frozen=True)
class RiskConfig:
    initial_equity: float = 10_000.0
    max_leverage: float = 10.0
    max_drawdown_stop: float = 0.20
    daily_loss_stop: float = 0.03


@dataclass(frozen=True)
class DecisionParams:
    level_mode: str = "model"
    trade_side: str = "both"
    entry_threshold: float = 0.58
    exit_threshold: float = 0.54
    exit_floor: float = 0.40
    min_conf_gap: float = 0.08
    min_edge_pips: float = 0.3
    stop_atr: float = 1.0
    take_atr: float = 1.0
    max_hold_bars: int = 12
    cooldown_bars: int = 2
    risk_per_trade: float = 0.01
    min_fvg_size_atr: float = 0.0
    min_signal_atr_pips: float = 0.0
    max_signal_atr_pips: float = 999.0
    max_trades_per_day: int = 0
    min_predicted_net_pips: float = -999.0
    max_prediction_std_pips: float = 999.0
    session_start_hour: int = -1
    session_end_hour: int = -1
    excluded_hours: str = ""
    allowed_months: str = ""
    excluded_weekdays: str = ""
    min_ema200_dist_pips: float = -999.0
    max_ema200_dist_pips: float = 999.0
    min_ema50_slope_pips: float = -999.0
    max_ema50_slope_pips: float = 999.0
    rolling_pnl_window: int = 0
    min_rolling_pnl: float = -999999.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ModelRunConfig:
    experiment_name: str
    run_id: str | None = None


@dataclass(frozen=True)
class OptimizerConfig:
    trials: int = 100
    seed: int = 42
    min_trades: int = 20
    max_acceptable_drawdown: float = 0.20


H1_DATA_CFG = {
    "pair": "EURUSD",
    "years": [2022, 2023],
    "timeframe": "H1",
    "weekends": "nogap",
    "norm_method": "fracdiff",
    "target_type": "triple_barrier",
    "target_col": "tb_label",
    "lags": [1, 2, 5, 10],
    "target_horizons": [1, 5, 15],
    "gap_bars": 50,
    "scaling": "none",
    "window_size": 500,
    "fracdiff_d": 0.3,
    "threshold": 6e-4,
    "k_up": 2.0,
    "k_down": 1.0,
    "horizon_bars": 10,
    "barrier_price": "hl",
    "barrier_norm_method": "log_returns",
}


FVG_DATA_CFG = {
    "pair": "EURUSD",
    "years": [2023],
    "weekends": "nogap",
    "base_timeframe": "M15",
    "higher_timeframe": "H1",
    "fractal_window": 5,
    "lookahead_bars": 96,
    "min_fvg_atr": 0.10,
    "norm_method": "log_returns",
    "fracdiff_d": 0.3,
    "threshold": 6e-4,
    "lags": [1, 2, 5, 10],
    "gap_events": 50,
    "scaling": "none",
    "window_size": 500,
    "use_engineered_features": False,
    "decision_delay_bars": 2,
    "single_timeframe": False,
    "require_unbroken_levels": True,
}
