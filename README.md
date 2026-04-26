# TQQQ / SQQQ Intraday VWAP Trading System

English | [日本語](./README.ja.md)

This repository is an automated intraday trading system dedicated to TQQQ and SQQQ, utilizing a 1-minute VWAP + ATR Trailing Stop strategy. It natively implements a "Cash Account Restriction" safety lock (maximum 1 buy transaction per symbol per day) to circumvent Good Faith Violation (GFV) penalties associated with T+1 cash settlement rules while remaining immensely profitable.

## Overview

- **Target Assets**: TQQQ (Long), SQQQ (Inverse for Short exposure)
- **Strategy**: 1-minute VWAP Crossover + ATR Trailing Stop
- **Optimized Parameters**: `ATR Period = 9`, `ATR Multiplier = 27.15`, `VWAP Threshold = 0.063%`
- **Trading Rules**: Fully compliant with Cash Accounts. The system will strictly execute at most 1 Long entry for TQQQ and 1 Long entry for SQQQ per day. This acts as a robust filter against sideways, highly volatile "chop" markets.
- **End of Day Flattening**: Forces all open positions to be liquidated at 15:58 (NY Time) to completely eliminate overnight risk.

## Operating Modes

Depends entirely on the `.env` settings.

- `LIVE` mode:
  - Enabled when `WEBULL_APP_KEY`, `WEBULL_APP_SECRET`, and `WEBULL_ACCOUNT_ID` are fully set.
- `DEMO` mode:
  - Automatically falls back to Demo mode if any Webull credentials are missing. Generates fully accurate signals and Discord notifications without routing real money orders.

## How It Works

1. **Data Ingestion**: Polls high-frequency 1-minute live quotes for TQQQ and SQQQ via the FMP (Financial Modeling Prep) API.
2. **Aggregator & Math Engine**: The `QuoteBarAggregator` assembles exact 1-minute boundary bars (Typical Price & Cumulative Volume). VWAP and ATR are computed instantaneously using JIT-compiled Numba code.
3. **Signal Generation**:
   - If TQQQ Close crosses above `VWAP * (1 + 0.063%)` -> Buy TQQQ (Uptrend)
   - If TQQQ Close crosses below `VWAP * (1 - 0.063%)` -> Buy SQQQ (Downtrend)
4. **Execution & Cash Bounds**: 
   - When a buy signal triggers, the system automatically issues a SELL order for the counter-asset (if holding) before placing the BUY order. 
   - If the system attempts to buy a ticker that has *already* been bought earlier in the day, the signal is ignored (remains FLAT) to avoid triggering a GFV.
5. **EOD Flat**: Liquidates the portfolio unconditionally at exactly 15:58.

## Discord Notifications

Set `DISCORD_BOT_TOKEN` and `DISCORD_CHANNEL_ID` to receive real-time webhook updates:
- Pre-market status / BP check
- Executed / Skipped Buy Orders
- EOD Market Close Summaries

## Running the Bot

```bash
python master_scheduler.py
```

## Qullamaggie Swing Breakout Validation

This repository also includes an optional research/backtest layer for the
Qullamaggie-style `industry composite + replacement` swing breakout portfolio.
It does not replace the live TQQQ/SQQQ VWAP bot by default; it is exposed as a
separate validation CLI.

The portfolio layer expects precomputed two-lane breakout event files
(`standard_breakout` and `tight_reversal`) with 5-minute trigger fields, plus a
long daily OHLCV history and an FMP-style universe file with industry and market
cap metadata.

```bash
python run_industry_replacement_backtest.py \
  --standard-events C:/path/to/standard_events_with_outcomes.csv \
  --tight-events C:/path/to/tight_reversal_events_with_outcomes.csv \
  --daily C:/path/to/daily_history.parquet \
  --universe C:/path/to/universe.parquet \
  --outdir reports/industry_replacement_backtest
```

Implemented portfolio behavior:

- 5 slots, 25% allocation per slot.
- Same-day ranking by `leader_score >= 98`, leader score, volume thrust, move
  from open, Industry RS, setup sweet spot, small/mid-cap bonus, and standard
  breakout bonus.
- A+ replacement rule: if slots/cash are unavailable, an elite candidate may
  replace a weaker existing position unless that position is already a strong
  winner.
- Super-winner runner exit with partial profit, DMA/hold-score degradation
  rules, 50DMA disaster exit, and tiered ATR trailing stop.

## Docker Deployment

```bash
docker compose up -d --build
```
