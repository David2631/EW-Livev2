"""Persistence layer for placed orders and active tickets (SQLite-backed)."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional


class OrderStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.mode = "sqlite" if self.path.suffix.lower() in {".db", ".sqlite", ".sqlite3"} else "json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.mode == "sqlite":
            self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT,
                    symbol TEXT,
                    direction TEXT,
                    timestamp TEXT,
                    ticket TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS active_positions (
                    ticket TEXT PRIMARY KEY,
                    key TEXT,
                    symbol TEXT,
                    last_seen TEXT,
                    opened TEXT
                )
                """
            )
            conn.commit()

    def load_executions(self) -> List[dict]:
        if self.mode == "json":
            if not self.path.exists():
                return []
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                return []
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                "SELECT key, symbol, direction, timestamp, ticket FROM executions ORDER BY id"
            ).fetchall()
        return [
            {"key": key, "symbol": symbol, "direction": direction, "timestamp": ts, "ticket": ticket}
            for key, symbol, direction, ts, ticket in rows
        ]

    def replace_executions(self, records: Iterable[dict]) -> None:
        if self.mode == "json":
            data = list(records)
            self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            return
        with sqlite3.connect(self.path) as conn:
            conn.execute("DELETE FROM executions")
            conn.executemany(
                "INSERT INTO executions (key, symbol, direction, timestamp, ticket) VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        r.get("key"),
                        r.get("symbol"),
                        str(r.get("direction")),
                        r.get("timestamp"),
                        r.get("ticket"),
                    )
                    for r in records
                ],
            )
            conn.commit()

    def append_execution(self, record: dict) -> None:
        if self.mode == "json":
            data = self.load_executions()
            data.append(record)
            self.replace_executions(data)
            return
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT INTO executions (key, symbol, direction, timestamp, ticket) VALUES (?, ?, ?, ?, ?)",
                (
                    record.get("key"),
                    record.get("symbol"),
                    str(record.get("direction")),
                    record.get("timestamp"),
                    record.get("ticket"),
                ),
            )
            conn.commit()

    def load_active(self) -> Dict[str, dict]:
        if self.mode == "json":
            fallback = self.path.parent / "active_positions.json"
            if not fallback.exists():
                return {}
            try:
                data = json.loads(fallback.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                "SELECT ticket, key, symbol, last_seen, opened FROM active_positions"
            ).fetchall()
        return {
            str(ticket): {"key": key, "symbol": symbol, "last_seen": last_seen, "opened": opened}
            for ticket, key, symbol, last_seen, opened in rows
        }

    def replace_active(self, active: Dict[str, dict]) -> None:
        if self.mode == "json":
            fallback = self.path.parent / "active_positions.json"
            fallback.write_text(json.dumps(active, ensure_ascii=False, indent=2), encoding="utf-8")
            return
        with sqlite3.connect(self.path) as conn:
            conn.execute("DELETE FROM active_positions")
            conn.executemany(
                "INSERT OR REPLACE INTO active_positions (ticket, key, symbol, last_seen, opened) VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        ticket,
                        entry.get("key"),
                        entry.get("symbol"),
                        entry.get("last_seen"),
                        entry.get("opened"),
                    )
                    for ticket, entry in active.items()
                ],
            )
            conn.commit()

    def upsert_active(self, ticket: str, entry: dict) -> None:
        if not ticket:
            return
        if self.mode == "json":
            active = self.load_active()
            active[str(ticket)] = entry
            self.replace_active(active)
            return
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO active_positions (ticket, key, symbol, last_seen, opened) VALUES (?, ?, ?, ?, ?)",
                (ticket, entry.get("key"), entry.get("symbol"), entry.get("last_seen"), entry.get("opened")),
            )
            conn.commit()

    def delete_tickets(self, tickets: Iterable[str]) -> None:
        to_delete = [t for t in tickets if t]
        if not to_delete:
            return
        if self.mode == "json":
            active = self.load_active()
            for ticket in to_delete:
                active.pop(str(ticket), None)
            self.replace_active(active)
            return
        with sqlite3.connect(self.path) as conn:
            conn.executemany(
                "DELETE FROM active_positions WHERE ticket = ?",
                [(t,) for t in to_delete],
            )
            conn.commit()