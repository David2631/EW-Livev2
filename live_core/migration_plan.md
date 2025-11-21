# Migration vom Backtest zum Live-System

1. **Konfiguration & Tooling**
   - Zentrale `LiveConfig` mit JSON/Env-Overrides, damit Startparameter von CLI oder Deployment beeinflusst werden können.
   - `requirements.txt` dokumentiert `MetaTrader5`, `pandas` und `numpy` für Live-Execution.

2. **Datenquelle**
   - `MetaTrader5Adapter` zieht aktuell die letzten `lookback_bars` aus MT5 und bietet einen Mock-Modus für lokale Tests (DryRun).
   - Sollte Vantage auch eine offene API anbieten, kann ein zweiter Adapter implementiert werden und über eine Factory getauscht werden.

3. **Signal-Engine**
   - `SignalEngine` nutzt ein vereinfachtes Zig-Zag-Pattern ohne EMA-/ML-Filter und erzeugt `EntrySignal` mit SL/TP.
   - Weitergehende Live-Validierung (Spread, Handelszeiten, News-Timeouts) kann als Filter-Pipeline ergänzt werden.

4. **Order-Manager**
   - `OrderManager` übernimmt Signale, berechnet Volumina und übergibt Aufträge an den Adapter.
   - Risikosteuerung (max offene Trades, ATR-basiertes SL) kann hier hinzugefügt werden.

5. **Main-Loop**
   - `main.py` verbindet alles zu einem zyklischen Ablauf: Daten holen → Signale bauen → Orders abschicken.
   - Für Livebetrieb reicht es, den Loop zu behalten und in Produktionsumgebung z. B. als Dienst zu starten.

6. **Nächste Schritte**
   - Logging/Monitoring (Equity, offene Trades, Notifications).
   - Integration eines Order-Trackings und Recovery nach MT5-Reconnects.
   - Vantage-spezifische Order-Checks (z. B. REST-Failover oder zusätzliche Pre-Check-Hooks).