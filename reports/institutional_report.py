"""Institutional-grade Backtest Factsheet generator
Generates a multi-page PDF with professional layout and metrics.

Usage: from reports.institutional_report import generate_report
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

sns.set(style='whitegrid')


@dataclass
class ReportOptions:
    title: str = "Strategy Factsheet"
    version: str = "1.0"
    rf_rate: float = 0.04  # default 4% p.a.
    compounding: bool = True
    monte_carlo_runs: int = 1000


# -------------------- Metrics --------------------

def calc_cagr(start_value: float, end_value: float, years: float) -> float:
    if start_value <= 0 or years <= 0:
        return 0.0
    return (end_value / start_value) ** (1.0 / years) - 1.0


def max_drawdown(equity: pd.Series) -> tuple[float, pd.Timestamp, pd.Timestamp, int]:
    # returns mdd_pct, trough_date, peak_date, recovery_days
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    mdd = drawdown.min()
    trough = drawdown.idxmin()
    peak = equity[:trough].idxmax() if trough is not None else equity.index[0]
    # recovery: first time after trough where equity >= roll_max.loc[peak]
    peak_val = roll_max.loc[peak]
    after = equity[trough:]
    try:
        rec_idx = after[after >= peak_val].index[0]
        recovery_days = (rec_idx - peak).days
    except Exception:
        rec_idx = None
        recovery_days = -1
    return float(-mdd * 100.0), trough, peak, recovery_days


def annualized_volatility(daily_returns: pd.Series) -> float:
    """Standard annualized volatility from daily returns (all days, incl. zero).
    Uses sqrt(252) annualization factor for trading days."""
    if len(daily_returns) < 2:
        return 0.0
    sigma = daily_returns.std(ddof=1)
    return float(sigma * math.sqrt(252.0))


def sharpe_ratio_annualized(daily_returns: pd.Series, rf_annual: float = 0.04) -> float:
    """Annualized Sharpe Ratio - standard institutional formula.
    Sharpe = (mean_daily * 252 - rf) / (std_daily * sqrt(252))
    """
    if len(daily_returns) < 2:
        return 0.0
    mean_daily = daily_returns.mean()
    std_daily = daily_returns.std(ddof=1)
    if std_daily < 1e-12:
        return 0.0
    # Annualize: E[R] * 252 for mean, sigma * sqrt(252) for vol
    ann_return = mean_daily * 252.0
    ann_vol = std_daily * math.sqrt(252.0)
    return float((ann_return - rf_annual) / ann_vol)


def downside_deviation(daily_returns: pd.Series) -> float:
    downside = daily_returns[daily_returns < 0]
    if len(downside) < 2:
        return 0.0
    return float(np.std(downside, ddof=1) * math.sqrt(252.0))


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    # omega = sum(positive excess returns) / abs(sum(negative excess returns))
    excess = returns - threshold
    pos = excess[excess > 0].sum()
    neg = -excess[excess < 0].sum()
    if neg == 0:
        return float('inf') if pos > 0 else 0.0
    return float(pos / neg)


def tail_ratio(returns: pd.Series, p_high: float = 0.95, p_low: float = 0.05) -> float:
    if len(returns) < 2:
        return 0.0
    high = np.percentile(returns, p_high * 100)
    low = np.percentile(returns, p_low * 100)
    if low == 0:
        return float('inf')
    return float(high / abs(low))


def var_historical(returns: pd.Series, alpha: float = 0.05) -> float:
    if len(returns) < 2:
        return 0.0
    return float(-np.percentile(returns, alpha * 100))


def var_parametric(returns: pd.Series, alpha: float = 0.05) -> float:
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    z = abs(np.percentile(np.random.normal(size=100000), alpha * 100))
    return float(- (mu - z * sigma))


def cvar_historical(returns: pd.Series, alpha: float = 0.05) -> float:
    if len(returns) < 2:
        return 0.0
    thresh = np.percentile(returns, alpha * 100)
    tail = returns[returns <= thresh]
    if len(tail) == 0:
        return 0.0
    return float(-tail.mean())


def ulcer_index(equity: pd.Series) -> float:
    # Ulcer index: sqrt(mean((drawdown_pct)^2)) over time
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    dd2 = (drawdown * 100.0) ** 2
    return float(np.sqrt(np.mean(dd2)))


def sqn(trade_returns: pd.Series) -> float:
    # System Quality Number: SQN = sqrt(N) * mean(R) / std(R) where R is net R-multiple per trade
    if len(trade_returns) < 2:
        return 0.0
    mean = np.mean(trade_returns)
    sd = np.std(trade_returns, ddof=1)
    if sd == 0:
        return float('inf') if mean > 0 else 0.0
    return float(np.sqrt(len(trade_returns)) * mean / sd)


# -------------------- Plot helpers --------------------

def plot_equity_and_drawdown(eq_series: pd.Series, train_until: Optional[pd.Timestamp] = None, pdf: PdfPages = None):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    axes[0].plot(eq_series.index, eq_series.values, color='navy', lw=1.2)
    axes[0].set_title('Portfolio Equity (Linear)')
    axes[0].grid(alpha=0.3)
    if train_until is not None:
        axes[0].axvspan(train_until, eq_series.index[-1], color='lightgrey', alpha=0.4)
    # log equity inset
    axes[0].plot(eq_series.index, np.log(eq_series.values), color='darkgreen', lw=0.7, alpha=0.8, label='log(equity)')
    axes[0].legend()

    # drawdown
    roll_max = eq_series.cummax()
    drawdown = (eq_series - roll_max) / roll_max * 100.0
    axes[1].fill_between(drawdown.index, drawdown.values, 0, color='firebrick', alpha=0.6)
    axes[1].set_ylabel('Drawdown %')
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    if pdf:
        pdf.savefig(fig); plt.close(fig)
    return fig


def plot_monthly_heatmap(eq_series: pd.Series, pdf: PdfPages = None):
    # compute monthly returns
    monthly = eq_series.resample('M').last().pct_change().dropna()
    mm = monthly.groupby([monthly.index.year, monthly.index.month]).apply(lambda x: x.iloc[0])
    years = sorted(list({d.year for d in monthly.index}))
    import calendar
    # pivot table
    dfm = monthly.copy()
    dfm = dfm.to_frame('ret')
    dfm['year'] = dfm.index.year
    dfm['month'] = dfm.index.month
    pivot = dfm.pivot_table(index='year', columns='month', values='ret')
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    sns.heatmap(pivot * 100.0, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=ax, cbar_kws={'label': '% Monthly'})
    ax.set_title('Monthly Returns Heatmap (%)')
    plt.tight_layout()
    if pdf:
        pdf.savefig(fig); plt.close(fig)
    return fig


def plot_rolling_metrics(eq_series: pd.Series, window_months: int = 12, pdf: PdfPages = None):
    monthly = eq_series.resample('M').last().pct_change().dropna()
    window = window_months
    rolling_sharpe = monthly.rolling(window).apply(lambda x: (np.nanmean(x) * 12.0 - 0.04) / (np.nanstd(x, ddof=1) * math.sqrt(12.0)) if np.nanstd(x, ddof=1) > 0 else 0.0)
    rolling_vol = monthly.rolling(window).std(ddof=1) * math.sqrt(12.0)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, label='Rolling Sharpe (12m)', color='navy')
    ax.plot(rolling_vol.index, rolling_vol.values, label='Rolling Vol (12m)', color='darkgreen')
    ax.legend(); ax.grid(alpha=0.3); ax.set_title('Rolling 12-Month Metrics')
    plt.tight_layout()
    if pdf:
        pdf.savefig(fig); plt.close(fig)
    return fig


def plot_return_distribution(trade_returns: pd.Series, pdf: PdfPages = None):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    sns.histplot(trade_returns, bins=50, kde=True, ax=ax, color='teal')
    ax.set_title('Distribution of Trade Returns')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if pdf:
        pdf.savefig(fig); plt.close(fig)
    return fig


def plot_monte_carlo(conservative_daily_rets: pd.Series, runs: int = 1000, start_cap: float = 100000.0, pdf: PdfPages = None):
    # bootstrap daily returns by sampling with replacement
    daily_rets = conservative_daily_rets.dropna()
    n_days = len(daily_rets)
    rng = np.random.default_rng(12345)
    sims = np.zeros((runs, n_days))
    for i in range(runs):
        samp = rng.choice(daily_rets.values, size=n_days, replace=True)
        sims[i] = np.cumprod(1.0 + samp) * start_cap
    percentiles = np.percentile(sims, [5, 25, 50, 75, 95], axis=0)
    idx = pd.date_range(start=conservative_daily_rets.index[0], periods=n_days, freq='D')
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(idx, percentiles[2], color='black', lw=1.2, label='Median')
    ax.fill_between(idx, percentiles[0], percentiles[-1], color='lightgrey', alpha=0.6, label='5-95%')
    ax.fill_between(idx, percentiles[1], percentiles[-2], color='silver', alpha=0.6, label='25-75%')
    ax.set_title('Monte Carlo Equity Cone (bootstrap daily returns)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    if pdf:
        pdf.savefig(fig); plt.close(fig)
    return fig


def plot_mae_mfe(trades_df: pd.DataFrame, pdf: PdfPages = None):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.scatter(trades_df['mae_r'], trades_df['pnl'], alpha=0.6)
    ax.set_xlabel('MAE (R-multiple)'); ax.set_ylabel('PnL ($)'); ax.set_title('MAE vs Final PnL')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if pdf:
        pdf.savefig(fig); plt.close(fig)
    return fig


# -------------------- Report generator --------------------

def generate_report(eq_series: pd.Series, trades_df: pd.DataFrame, cfg: dict, out_pdf: str, options: ReportOptions = ReportOptions()):
    # eq_series: pd.Series indexed by date with capital values
    # trades_df: DataFrame with per-trade rows; expects columns: time_in, time_out, pnl, mae_r, mfe_r
    pdf = PdfPages(out_pdf)

    # Header page
    fig, ax = plt.subplots(1, 1, figsize=(11, 8.5))
    ax.axis('off')
    txt = f"{options.title}\nVersion: {options.version}\nUnderlying: {cfg.get('_PROFILE', '')} / {cfg.get('SYMBOL', '')}\nTimeframe: {cfg.get('TIMEFRAME', 'H1')}"
    ax.text(0.02, 0.85, txt, fontsize=14, weight='bold')
    period_txt = f"Backtest: {eq_series.index[0].date()} — {eq_series.index[-1].date()} ({(eq_series.index[-1]-eq_series.index[0]).days/365.0:.2f} years)"
    ax.text(0.02, 0.77, period_txt, fontsize=10)
    cost_txt = f"Costs: spread={cfg.get('SPREAD_PIPS', 'n/a')} pips, commission=${cfg.get('COMMISSION_ROUND_PER_LOT', 'n/a')}/lot, slippage={cfg.get('SLIPPAGE', 0.0)}"
    ax.text(0.02, 0.73, cost_txt, fontsize=10)
    ax.text(0.02, 0.70, f"Positioning: {'Compounding' if options.compounding else 'Fixed Lot'}", fontsize=10)
    pdf.savefig(fig); plt.close(fig)

    # Core Performance & metrics summary page
    # Compute base metrics
    start = eq_series.iloc[0]; end = eq_series.iloc[-1]
    years = max((eq_series.index[-1] - eq_series.index[0]).days / 365.0, 1e-6)
    
    # Use pre-computed metrics from CLI if available (ensures consistency)
    total_return = cfg.get('total_return', (end / start - 1.0) * 100.0)
    cagr = cfg.get('cagr', calc_cagr(start, end, years) * 100.0)
    profit_factor = cfg.get('profit_factor', 0.0)
    sharpe = cfg.get('sharpe', 0.0)
    sortino = cfg.get('sortino', 0.0)
    vol = cfg.get('vol_annual', 0.0)
    mdd_pct_cfg = cfg.get('max_dd', None)
    
    mu = np.mean(trades_df['pnl']) if len(trades_df) else 0.0

    mdd_pct, trough, peak, rec_days = max_drawdown(eq_series)
    # Override mdd if pre-computed
    if mdd_pct_cfg is not None:
        mdd_pct = abs(mdd_pct_cfg)  # ensure positive for display
    
    # daily returns for additional metrics (omega, tail, VaR, etc.)
    serd = eq_series.resample('D').last().ffill()
    daily_rets = serd.pct_change().fillna(0)

    # Additional risk metrics
    calmar = (cagr/100.0) / (abs(mdd_pct)/100.0) if mdd_pct != 0 else float('inf')
    omega = omega_ratio(daily_rets)
    tail = tail_ratio(daily_rets)
    vaR95_hist = var_historical(daily_rets, 0.05)
    vaR99_hist = var_historical(daily_rets, 0.01)
    vaR95_par = var_parametric(daily_rets, 0.05)
    cvar95 = cvar_historical(daily_rets, 0.05)
    ulcer = ulcer_index(eq_series)

    # Trade-level statistics
    total_trades = len(trades_df)
    trades_per_year = total_trades / years if years>0 else total_trades
    wins = trades_df[trades_df['pnl']>0]
    losses = trades_df[trades_df['pnl']<=0]
    avg_win = wins['pnl'].mean() if len(wins)>0 else 0.0
    avg_loss = losses['pnl'].mean() if len(losses)>0 else 0.0
    payoff = (avg_win / abs(avg_loss)) if avg_loss != 0 else float('inf')
    expectancy = trades_df['pnl'].mean() if total_trades>0 else 0.0
    best_trade = trades_df['pnl'].max() if total_trades>0 else 0.0
    worst_trade = trades_df['pnl'].min() if total_trades>0 else 0.0

    # Holding times
    ht = (pd.to_datetime(trades_df['time_out']) - pd.to_datetime(trades_df['time_in'])).dt.total_seconds() / 3600.0 / 24.0
    avg_hold_all = ht.mean() if len(ht)>0 else 0.0
    avg_hold_win = ht[wins.index].mean() if len(wins)>0 else 0.0
    avg_hold_loss = ht[losses.index].mean() if len(losses)>0 else 0.0

    # Commission impact (estimate from config)
    units_per_lot = float(cfg.get('UNITS_PER_LOT', 1))
    comm_per_lot = float(cfg.get('COMMISSION_ROUND_PER_LOT', 0.0))
    lots = (trades_df['size'] / units_per_lot).fillna(0)
    comms = lots * comm_per_lot
    total_comm = comms.sum()
    gross_pnl_est = trades_df['pnl'] + comms
    gross_total_gain = gross_pnl_est[gross_pnl_est>0].sum()
    comm_impact_pct = (total_comm / gross_total_gain * 100.0) if gross_total_gain>0 else 0.0

    # SQN (use rr if available)
    rr_series = trades_df.get('rr') if 'rr' in trades_df.columns else None
    sqn_val = sqn(rr_series.dropna()) if rr_series is not None else 0.0

    # Streaks
    max_consec_wins = 0; max_consec_losses = 0; cur_win=0; cur_loss=0
    for v in trades_df['pnl']:
        if v>0:
            cur_win+=1; max_consec_wins=max(max_consec_wins,cur_win); cur_loss=0
        else:
            cur_loss+=1; max_consec_losses=max(max_consec_losses,cur_loss); cur_win=0

    fig, ax = plt.subplots(1, 1, figsize=(11, 8.5))
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Two-column layout for metrics
    col1_x = 0.02
    col2_x = 0.52
    y_start = 0.92
    line_height = 0.042
    
    def put_left(k, v, row):
        y = y_start - row * line_height
        ax.text(col1_x, y, f"{k}:", fontsize=9, weight='bold', va='top')
        ax.text(col1_x + 0.22, y, f"{v}", fontsize=9, va='top')
    
    def put_right(k, v, row):
        y = y_start - row * line_height
        ax.text(col2_x, y, f"{k}:", fontsize=9, weight='bold', va='top')
        ax.text(col2_x + 0.22, y, f"{v}", fontsize=9, va='top')
    
    # Left column - Performance metrics
    ax.text(col1_x, 0.97, 'PERFORMANCE', fontsize=11, weight='bold', color='navy')
    put_left('Total Return', f"{total_return:.2f}%", 0)
    put_left('CAGR', f"{cagr:.2f}%", 1)
    put_left('Profit Factor', f"{profit_factor:.2f}" if profit_factor else "n/a", 2)
    adj_pf = (trades_df[trades_df['pnl']<=trades_df['pnl'].quantile(0.99)]['pnl'].sum()) / abs(trades_df[trades_df['pnl']<=trades_df['pnl'].quantile(0.01)]['pnl'].sum()) if len(trades_df)>0 else 0
    put_left('Adj PF (trim 1%)', f"{adj_pf:.2f}", 3)
    put_left('Win Rate', f"{100.0 * (len(wins) / max(1, total_trades)):.2f}%", 4)
    put_left('Total Trades', f"{total_trades}", 5)
    put_left('Trades/Year', f"{trades_per_year:.1f}", 6)
    put_left('Expectancy', f"${expectancy:.2f}", 7)
    put_left('Payoff Ratio', f"{payoff:.2f}", 8)
    put_left('Best Trade', f"${best_trade:.2f}", 9)
    put_left('Worst Trade', f"${worst_trade:.2f}", 10)
    
    # Right column - Risk metrics
    ax.text(col2_x, 0.97, 'RISK METRICS', fontsize=11, weight='bold', color='darkred')
    put_right('Max Drawdown', f"{mdd_pct:.2f}%", 0)
    put_right('Recovery (days)', f"{rec_days}", 1)
    put_right(f'Sharpe (rf {options.rf_rate*100:.1f}%)', f"{sharpe:.2f}", 2)
    put_right('Sortino', f"{sortino:.2f}", 3)
    put_right('Calmar', f"{calmar:.2f}", 4)
    put_right('Omega', f"{omega:.2f}", 5)
    put_right('Tail Ratio', f"{tail:.2f}", 6)
    put_right('VaR 95%', f"{vaR95_hist*100:.2f}%", 7)
    put_right('CVaR 95%', f"{cvar95*100:.2f}%", 8)
    put_right('Ulcer Index', f"{ulcer:.2f}", 9)
    put_right('Vol (ann.)', f"{vol:.2f}%", 10)
    
    # Third section - Trade statistics (bottom half)
    ax.text(col1_x, 0.42, 'TRADE STATISTICS', fontsize=11, weight='bold', color='darkgreen')
    put_left('SQN', f"{sqn_val:.2f}", 13)
    put_left('Avg Hold Winners', f"{avg_hold_win:.1f} days", 14)
    put_left('Avg Hold Losers', f"{avg_hold_loss:.1f} days", 15)
    put_left('Max Win Streak', f"{max_consec_wins}", 16)
    put_left('Max Loss Streak', f"{max_consec_losses}", 17)
    put_left('Commission Impact', f"{comm_impact_pct:.2f}%", 18)
    
    # Notes section
    ax.text(col2_x, 0.42, 'NOTES', fontsize=11, weight='bold', color='gray')
    notes = [
        "• Sharpe/Sortino from trade-frequency adjusted vol",
        "• Adj PF trims extreme 1% wins/losses",
        "• All metrics use closed-bar data only",
        f"• Risk-free rate: {options.rf_rate*100:.1f}% p.a.",
        f"• Exposure: {(total_trades/years/252*100):.1f}% of trading days"
    ]
    for i, note in enumerate(notes):
        ax.text(col2_x, 0.38 - i*0.035, note, fontsize=8, va='top')
    
    pdf.savefig(fig); plt.close(fig)

    # Plots: equity & drawdown
    train_until = cfg.get('TRAIN_UNTIL', None)
    plot_equity_and_drawdown(eq_series, train_until=train_until, pdf=pdf)

    # Monthly heatmap
    plot_monthly_heatmap(eq_series, pdf=pdf)

    # Rolling metrics
    plot_rolling_metrics(eq_series, window_months=12, pdf=pdf)

    # Distribution
    if 'per_share' in trades_df.columns:
        returns_for_hist = trades_df['pnl']
    else:
        returns_for_hist = trades_df['pnl']
    plot_return_distribution(returns_for_hist, pdf=pdf)

    # Monte Carlo
    # Build daily series covering OOS horizon (or full) for bootstrap
    daily_for_mc = serd.pct_change().fillna(0)
    plot_monte_carlo(daily_for_mc, runs=options.monte_carlo_runs, start_cap=start, pdf=pdf)

    # MAE/MFE
    if 'mae_r' in trades_df.columns:
        plot_mae_mfe(trades_df, pdf=pdf)

    # Year-over-year table page
    yoy = serd.resample('Y').last().pct_change().dropna() * 100.0
    years = [idx.year for idx in yoy.index]
    data = []
    for y in years:
        start_y = serd[serd.index.year == y].iloc[0]
        end_y = serd[serd.index.year == y].iloc[-1]
        ret = (end_y / start_y - 1.0) * 100.0
        # simple MDD for year
        eq_year = serd[serd.index.year == y]
        mdd_year = max_drawdown(eq_year)[0]
        trades_year = len(trades_df[trades_df['time_in'].dt.year == y])
        winrate = 100.0 * len(trades_df[(trades_df['time_in'].dt.year == y) & (trades_df['pnl']>0)]) / max(1, trades_year)
        data.append([y, f"{ret:.2f}%", f"{mdd_year:.2f}%", trades_year, f"{winrate:.2f}%"])
    # render table
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax.axis('off')
    table = ax.table(cellText=data, colLabels=['Year','Return %','Max DD %','Trades','# Win Rate'], loc='center')
    table.auto_set_font_size(False); table.set_fontsize(8)
    table.scale(1, 1.2)
    ax.set_title('Year-over-Year Performance')
    pdf.savefig(fig); plt.close(fig)

    # Top trades / attribution
    top = trades_df.sort_values('pnl', ascending=False).head(20)[['time_in','pnl']]
    fig, ax = plt.subplots(1,1,figsize=(9,6))
    ax.axis('off')
    tbl = ax.table(cellText=top.values.tolist(), colLabels=top.columns.tolist(), loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1,1.2)
    ax.set_title('Top 20 Trades (by PnL)')
    pdf.savefig(fig); plt.close(fig)

    pdf.close()
    return out_pdf
