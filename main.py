"""Entry-Point für das Live-System (MT5 + Vantage)."""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd

from live_core.config import LiveConfig
from live_core.cycle import CycleRunner
from live_core.execution import OrderManager
from live_core.ml_model import MLProbabilityProvider
from live_core.mt5_adapter import MetaTrader5Adapter
from live_core.signals import SignalEngine

logger = logging.getLogger("ew_live")


def configure_logging(log_path: Path) -> None:
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live-Automation für EW-Strategie ohne EMA/ML")
    parser.add_argument("--config", "-c", help="Pfad zur JSON-Konfigurationsdatei")
    parser.add_argument("--dry-run", action="store_true", help="Nur Signale berechnen, nichts senden")
    parser.add_argument("--once", action="store_true", help="Nur ein Zyklus statt Dauerschleife")
    parser.add_argument("--symbols-file", help="Pfad zur Datei mit einem Symbol pro Zeile (Standard: Symbols.txt)")
    parser.add_argument("--log-file", default="live_execution.log", help="Pfad zur Logdatei (Standard: live_execution.log)")
    return parser.parse_args()


def load_symbols(path: Optional[str]) -> List[str]:
    if not path:
        return []
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / path
    if not candidate.exists():
        return []
    return [line.strip() for line in candidate.read_text(encoding="utf-8").splitlines() if line.strip() and not line.strip().startswith("#")]


def verify_mt5_connection(adapter: MetaTrader5Adapter) -> None:
    adapter.connect()
    info = adapter.get_account_info()
    if not info:
        raise ConnectionError("MT5 meldet keine Kontoinformationen")
    logger.info(
        "MT5 verbunden: Konto %(login)s %(currency)s Balance=%(balance).2f Leverage=%(leverage)s",
        {
            "login": info.get("login"),
            "currency": info.get("currency"),
            "balance": info.get("balance"),
            "leverage": info.get("leverage"),
        },
    )


def main() -> None:
    args = parse_args()
    base_config = LiveConfig.load_from_file(args.config) if args.config else LiveConfig()
    cfg = base_config.with_overrides(LiveConfig.env_overrides())
    symbols_file = args.symbols_file or cfg.symbols_file
    cfg.symbols_file = symbols_file
    log_path = Path(args.log_file)
    if not log_path.is_absolute():
        log_path = Path.cwd() / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    configure_logging(log_path)
    symbols = load_symbols(symbols_file)
    if not symbols:
        logger.warning(f"Symbols-Datei '{symbols_file}' leer oder fehlt -> verwende '{cfg.symbol}'")
        symbols = [cfg.symbol]
    adapter = MetaTrader5Adapter()
    try:
        verify_mt5_connection(adapter)
    except Exception as exc:
        logger.error(f"MT5-Verbindung fehlgeschlagen: {exc}")
        sys.exit(1)
    ml_provider: Optional[MLProbabilityProvider] = None
    if cfg.ml_probability_path:
        try:
            ml_provider = MLProbabilityProvider(cfg.ml_probability_path)
        except FileNotFoundError as exc:
            logger.warning(f"ML-Wahrscheinlichkeiten nicht geladen: {exc}")
    engine = SignalEngine(cfg, ml_provider)
    manager = OrderManager(adapter, cfg)

    def log_live(symbol: str, message: str) -> None:
        logger.info(f"[{symbol}] {message}")

    runner = CycleRunner(cfg, adapter, engine, manager, log_live)

    try:
        while True:
            summary = runner.run_cycle(symbols, args.dry_run)
            log_live(
                "cycle",
                f"Cycle #{summary.index} abgeschlossen (Symbole={summary.symbols_processed}, "
                f"Signale={summary.total_signals}, Dauer={summary.duration_seconds:.2f}s)",
            )
            if args.once or args.dry_run:
                break
            time.sleep(10)
    finally:
        for handler in logger.handlers:
            handler.flush()
        adapter.disconnect()


if __name__ == "__main__":
    main()