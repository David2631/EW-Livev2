"""Hilfsmodul, um ML-Wahrscheinlichkeiten fÃ¼r Live-Signale einzuspeisen."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class MLProbabilityEntry:
    symbol: str
    entry_time: datetime
    setup: Optional[str]
    direction: Optional[str]
    probability: float


class MLProbabilityProvider:
    """LÃ¤dt Wahrscheinlichkeiten aus CSV/JSON-Dateien und liefert sie fÃ¼r Signale."""

    _TIME_TOLERANCE = timedelta(minutes=3)

    def __init__(self, path: Optional[str]):
        self._path = Path(path).expanduser() if path else None
        self._entries: Dict[str, List[MLProbabilityEntry]] = {}
        if self._path:
            self.load()

    def load(self) -> None:
        """Lade den angegebenen Pfad (CSV oder JSON)."""
        self._entries.clear()
        if not self._path:
            return
        if not self._path.exists():
            raise FileNotFoundError(f"ML-Wahrscheinlichkeiten nicht gefunden: {self._path}")
        suffix = self._path.suffix.lower()
        if suffix == ".csv":
            self._load_csv()
        else:
            self._load_json()

    def _load_csv(self) -> None:
        import csv

        with self._path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                entry = self._parse_row(row)
                if entry is None:
                    continue
                self._entries.setdefault(entry.symbol, []).append(entry)

    def _load_json(self) -> None:
        import json

        with self._path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, dict):
            payload = [payload]
        for row in payload:
            entry = self._parse_row(row)
            if entry is None:
                continue
            self._entries.setdefault(entry.symbol, []).append(entry)

    def _parse_row(self, row: Dict[str, object]) -> Optional[MLProbabilityEntry]:
        symbol = str(row.get("symbol", "")).upper().strip()
        if not symbol:
            return None
        probability = self._parse_probability(row.get("probability") or row.get("prob"))
        if probability is None:
            return None
        setup = row.get("setup")
        direction = row.get("direction")
        entry_time = self._parse_timestamp(str(row.get("entry_time", "")))
        if entry_time is None:
            return None
        return MLProbabilityEntry(symbol=symbol, entry_time=entry_time, setup=self._normalize_text(setup), direction=self._normalize_text(direction), probability=probability)

    @staticmethod
    def _parse_probability(value: object) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_timestamp(value: str) -> Optional[datetime]:
        if not value:
            return None
        value = value.strip()
        try:
            ts = pd.to_datetime(value, utc=True)
            if pd.isna(ts):
                return None
            return ts.tz_convert(None).to_pydatetime()
        except Exception:  # pragma: no cover - defensive
            try:
                return datetime.fromisoformat(value)
            except Exception:
                return None

    @staticmethod
    def _normalize_text(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return value.strip().upper()

    def get_probability(
        self,
        symbol: str,
        entry_time: datetime,
        setup: str,
        direction: str,
    ) -> Optional[float]:
        """Versuche, eine WT-Entry-Wahrscheinlichkeit aus dem Cache zu liefern."""
        entries = self._entries.get(symbol.upper())
        if not entries:
            return None
        target_time = self._normalize_time(entry_time)
        best: Optional[MLProbabilityEntry] = None
        best_diff = self._TIME_TOLERANCE
        norm_setup = self._normalize_text(setup)
        norm_direction = direction.upper() if direction else None
        for entry in entries:
            if entry.probability is None:
                continue
            if norm_setup and entry.setup and entry.setup != norm_setup:
                continue
            if norm_direction and entry.direction and entry.direction != norm_direction:
                continue
            diff = abs(entry.entry_time - target_time)
            if diff <= self._TIME_TOLERANCE and diff <= best_diff:
                best = entry
                best_diff = diff
        return best.probability if best else None

    @staticmethod
    def _normalize_time(value: datetime) -> datetime:
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value.replace(second=0, microsecond=0)

    @property
    def has_entries(self) -> bool:
        return bool(self._entries)