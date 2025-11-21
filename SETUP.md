# Setup

This project requires a few Python packages before running the live executor. Install everything with:

```powershell
pip install -r requirements.txt
```

`MetaTrader5` only runs on Windows and depends on the MT5 terminal running (e.g. your VPS copy). Its importer is wrapped in `live_core.mt5_adapter`, so the rest of the code keeps working even when the package is missing during testing.

On the VPS where live trading happens, install MetaTrader5 using the Windows wheel:

```powershell
pip install MetaTrader5>=5.0.40
```

After that, verify the account credentials are exported as environment variables or passed via CLI flags (`--mt5-login`, `--mt5-password`, `--mt5-server`). Restart the terminal if the MT5 terminal updates the trading account or credentials.
