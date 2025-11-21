"""Cycle orchestrator to run a full sweep over configured assets."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable, Sequence

import pandas as pd

from .config import LiveConfig
from .execution import OrderManager
from .mt5_adapter import MetaTrader5Adapter
from .signals import SignalEngine


@dataclass
class CycleSummary:
    index: int
    start_time: datetime
    duration_seconds: float
    symbols_processed: int
    total_signals: int
    dry_run: bool


def build_dataframe(rates: Sequence[dict]) -> pd.DataFrame:
    if not rates:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    if "time" in df.columns:
        df["date"] = pd.to_datetime(df["time"], unit="s")
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
    return df


class CycleRunner:
    def __init__(
        self,
        cfg: LiveConfig,
        adapter: MetaTrader5Adapter,
        engine: SignalEngine,
        manager: OrderManager,
        logger: Callable[[str, str], None],
    ) -> None:
        self.cfg = cfg
        self.adapter = adapter
        self.engine = engine
        self.manager = manager
        self.logger = logger
        self.cycle_index = 0

    def run_cycle(self, symbols: Iterable[str], dry_run: bool) -> CycleSummary:
        self.cycle_index += 1
        start_time = datetime.now(timezone.utc)
        total_signals = 0
        symbols_list = list(symbols)
        for symbol in symbols_list:
            self.cfg.symbol = symbol
            rates = self.adapter.get_rates(symbol, self.cfg.timeframe, self.cfg.lookback_bars)
            df = build_dataframe(rates)
            df = self.engine.preprocess(df)
            signals = self.engine.build_signals(df, symbol)
            total_signals += len(signals)
            last_signal = signals[-1] if signals else "none"
            self.logger(symbol, f"Signals={len(signals)} LastEntry={last_signal}")
            if not dry_run:
                self.manager.evaluate_signals(symbol, signals)
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        return CycleSummary(
            index=self.cycle_index,
            start_time=start_time,
            duration_seconds=duration,
            symbols_processed=len(symbols_list),
            total_signals=total_signals,
            dry_run=dry_run,
        )