"""Lightweight volatility gate to filter orders based on reachable price within a short horizon."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np


@dataclass
class VolatilityDecision:
    probability: float
    sigma_daily: float
    z_score: float
    samples_used: int
    horizon_days: float


class VolatilityGate:
    """Estimates near-term reach probability using log-return volatility."""

    def __init__(self, cfg: object) -> None:
        self.horizon_days = max(float(getattr(cfg, "vola_horizon_days", 2.0) or 2.0), 0.1)
        self.threshold = float(getattr(cfg, "vola_probability_threshold", 0.0) or 0.0)
        self.min_samples = max(int(getattr(cfg, "vola_min_samples", 64) or 64), 10)

    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _estimate_daily_sigma(self, closes: Sequence[float], timeframe_minutes: Optional[float]) -> Optional[tuple[float, int]]:
        if closes is None:
            return None
        arr = np.asarray(list(closes), dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < self.min_samples:
            return None
        log_returns = np.diff(np.log(arr))
        if log_returns.size <= 1:
            return None
        sigma_per_bar = float(np.std(log_returns, ddof=1))
        if sigma_per_bar <= 0:
            return None
        periods_per_day = None
        if timeframe_minutes and timeframe_minutes > 0:
            periods_per_day = max(1.0, 1440.0 / float(timeframe_minutes))
        sigma_daily = sigma_per_bar if not periods_per_day else sigma_per_bar * math.sqrt(periods_per_day)
        return sigma_daily, int(log_returns.size)

    def probability_to_reach(
        self,
        *,
        current_price: float,
        target_price: float,
        closes: Sequence[float],
        timeframe_minutes: Optional[float],
    ) -> Optional[VolatilityDecision]:
        if current_price <= 0 or target_price <= 0:
            return None
        estimate = self._estimate_daily_sigma(closes, timeframe_minutes)
        if estimate is None:
            return None
        sigma_daily, samples_used = estimate
        horizon_sigma = sigma_daily * math.sqrt(self.horizon_days)
        if horizon_sigma <= 0:
            return None
        log_move = math.log(target_price / current_price)
        z = log_move / horizon_sigma
        if target_price >= current_price:
            probability = 1.0 - self._norm_cdf(z)
        else:
            probability = self._norm_cdf(z)
        return VolatilityDecision(
            probability=max(0.0, min(1.0, float(probability))),
            sigma_daily=sigma_daily,
            z_score=z,
            samples_used=samples_used,
            horizon_days=self.horizon_days,
        )

    def allows(self, decision: Optional[VolatilityDecision]) -> bool:
        if decision is None:
            return True
        return decision.probability >= self.threshold