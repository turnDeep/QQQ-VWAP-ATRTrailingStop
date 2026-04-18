import polars as pl
import numpy as np
import optuna
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from core.strategy import calc_vwap_atr, shift_signal_over_day

def simulate_trading_curve(dates, opens, closes, pos, initial_capital=25000.0, commission=0.0005):
    capital = initial_capital
    curr_pos = 0 
    shares = 0.0
    n = len(opens)
    daily_capital = np.zeros(n, dtype=np.float64)
    daily_dates = np.zeros(n, dtype=np.int32)
    daily_idx = 0
    
    for i in range(n):
        target_pos = pos[i]
        is_eod = (i == n - 1 or dates[i+1] != dates[i])
        if is_eod: target_pos = 0

        if target_pos != curr_pos:
            if curr_pos != 0:
                price = closes[i] if is_eod else opens[i]
                capital += shares * price * curr_pos
                capital -= abs(shares) * price * commission
                shares = 0.0
                curr_pos = 0
            if target_pos != 0:
                price = opens[i]
                shares = capital / price if target_pos == 1 else -(capital / price)
                capital -= shares * price
                capital -= abs(shares) * price * commission
                curr_pos = target_pos
                
        if is_eod:
            val = capital + (shares * closes[i]) if curr_pos != 0 else capital
            daily_capital[daily_idx] = val
            daily_dates[daily_idx] = dates[i]
            daily_idx += 1
            
    return daily_dates[:daily_idx], daily_capital[:daily_idx]


def load_dataset(csv_path):
    print(f"Loading {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found. Ensure you have the TQQQ dataset ready.")
        return None
    df = pl.read_csv(csv_path)
    df = df.with_columns(pl.col('date').str.to_datetime('%Y-%m-%d %H:%M:%S'))
    
    df = df.filter(
        ((pl.col('date').dt.hour() == 9) & (pl.col('date').dt.minute() >= 30)) |
        ((pl.col('date').dt.hour() >= 10) & (pl.col('date').dt.hour() <= 15))
    )
    df = df.with_columns(pl.col('date').dt.date().alias('day'))

    df = df.with_columns(typical_price=(pl.col('high') + pl.col('low') + pl.col('close')) / 3.0)
    df = df.with_columns(vp=pl.col('typical_price') * pl.col('volume'))
    df = df.with_columns([
        pl.col('vp').cum_sum().over('day').alias('cum_vp'),
        pl.col('volume').cum_sum().over('day').alias('cum_vol')
    ])
    df = df.with_columns(vwap=pl.col('cum_vp') / pl.col('cum_vol'))
    
    df = df.with_columns(prev_close=pl.col('close').shift(1))
    df = df.with_columns(
        tr1=(pl.col('high') - pl.col('low')),
        tr2=(pl.col('high') - pl.col('prev_close')).abs(),
        tr3=(pl.col('low') - pl.col('prev_close')).abs()
    )
    df = df.with_columns(tr=pl.max_horizontal(['tr1', 'tr2', 'tr3']))
    return df

def objective(trial, df):
    atr_period = trial.suggest_int('atr_period', 5, 20)
    atr_mult = trial.suggest_float('atr_mult', 5.0, 30.0)
    threshold = trial.suggest_float('threshold', 0.0001, 0.002, log=True)
    
    df_atr = df.with_columns(atr=pl.col('tr').rolling_mean(window_size=atr_period))
    closes = df['close'].fill_null(strategy="forward").to_numpy()
    opens = df['open'].to_numpy()
    vwaps = df['vwap'].fill_null(0.0).to_numpy()
    atrs = df_atr['atr'].fill_null(0.0).to_numpy()
    dates_int = df['day'].cast(pl.Int32).to_numpy()
    
    signal_raw = calc_vwap_atr(closes, vwaps, atrs, dates_int, atr_mult, threshold)
    pos = shift_signal_over_day(signal_raw, dates_int)
    _, daily_capital = simulate_trading_curve(dates_int, opens, closes, pos, 25000.0)
    return daily_capital[-1]

def main():
    # If the user doesn't have the dataset, they must download it first using FMP
    dataset_path = "tqqq_1min_historical_data.csv"
    df = load_dataset(dataset_path)
    if df is None:
        return
        
    print("Starting Optuna optimization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, df), n_trials=50, n_jobs=-1)
    
    print("\nBest VWAP Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
        
    print("\nCopy these values into core/config.py to update the system!")

if __name__ == '__main__':
    main()
