import polars as pl
import numba
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

@numba.njit
def calc_vwap_atr(closes, vwaps, atrs, dates, multiplier, thresh):
    n = len(closes)
    pos_signal = np.zeros(n, dtype=np.int32)
    current_trend = 0
    current_stop = 0.0
    for i in range(1, n):
        if dates[i] != dates[i-1]:
            current_trend = 0
            pos_signal[i] = 0
            continue
        curr_close = closes[i]
        curr_vwap = vwaps[i]
        curr_atr = atrs[i]
        
        if current_trend == 0:
            if curr_close > curr_vwap * (1 + thresh):
                current_trend = 1
                current_stop = curr_close - multiplier * curr_atr
            elif curr_close < curr_vwap * (1 - thresh):
                current_trend = -1
                current_stop = curr_close + multiplier * curr_atr
        else:
            if current_trend == 1:
                new_stop = curr_close - multiplier * curr_atr
                current_stop = max(current_stop, new_stop)
                if curr_close < current_stop:
                    current_trend = -1 if curr_close < curr_vwap * (1 - thresh) else 0
                    if current_trend == -1:
                        current_stop = curr_close + multiplier * curr_atr
            elif current_trend == -1:
                new_stop = curr_close + multiplier * curr_atr
                current_stop = min(current_stop, new_stop)
                if curr_close > current_stop:
                    current_trend = 1 if curr_close > curr_vwap * (1 + thresh) else 0
                    if current_trend == 1:
                        current_stop = curr_close - multiplier * curr_atr
        pos_signal[i] = current_trend
    return pos_signal

@numba.njit
def shift_signal_over_day(signal, dates):
    n = len(signal)
    pos = np.zeros(n, dtype=np.int32)
    for i in range(1, n):
        if dates[i] == dates[i-1]:
            pos[i] = signal[i-1]
        else:
            pos[i] = 0
    return pos

@numba.njit
def simulate_restricted_dual_trading(dates, opens_tqqq, closes_tqqq, opens_sqqq, closes_sqqq, pos, initial_capital=25000.0, commission=0.0005):
    capital = initial_capital
    curr_pos = 0 # 0=Flat, 1=TQQQ, -1=SQQQ
    shares = 0.0
    
    n = len(opens_tqqq)
    
    daily_capital = np.zeros(n, dtype=np.float64)
    daily_dates = np.zeros(n, dtype=np.int32)
    daily_idx = 0
    
    tqqq_bought_today = False
    sqqq_bought_today = False
    
    for i in range(n):
        is_eod = (i == n - 1 or dates[i+1] != dates[i])
        
        if i == 0 or dates[i] != dates[i-1]:
            tqqq_bought_today = False
            sqqq_bought_today = False
            
        target_pos = pos[i]
        if is_eod:
            target_pos = 0

        if target_pos != curr_pos:
            val_before_trade = capital
            if curr_pos == 1:
                price = closes_tqqq[i] if is_eod else opens_tqqq[i]
                capital += shares * price
                capital -= shares * commission
                shares = 0.0
                curr_pos = 0
            elif curr_pos == -1:
                price = closes_sqqq[i] if is_eod else opens_sqqq[i]
                capital += shares * price
                capital -= shares * commission
                shares = 0.0
                curr_pos = 0
                
            if target_pos == 1:
                if not tqqq_bought_today:
                    price = opens_tqqq[i]
                    shares = capital / price
                    capital -= shares * price
                    capital -= shares * commission
                    curr_pos = 1
                    tqqq_bought_today = True
            elif target_pos == -1:
                if not sqqq_bought_today:
                    price = opens_sqqq[i]
                    shares = capital / price
                    capital -= shares * price
                    capital -= shares * commission
                    curr_pos = -1
                    sqqq_bought_today = True
            
        if is_eod:
            val = capital
            if curr_pos == 1:
                val += shares * closes_tqqq[i]
            elif curr_pos == -1:
                val += shares * closes_sqqq[i]
                
            daily_capital[daily_idx] = val
            daily_dates[daily_idx] = dates[i]
            daily_idx += 1
            
    return daily_dates[:daily_idx], daily_capital[:daily_idx]

def load_data():
    df_tqqq = pl.read_csv(r"c:\Users\plane\BreakOut\tqqq_1min_historical_data_adjusted.csv")
    df_sqqq = pl.read_csv(r"c:\Users\plane\BreakOut\sqqq_1min_historical_data.csv")
    
    df_tqqq = df_tqqq.with_columns(pl.col('date').str.to_datetime('%Y-%m-%d %H:%M:%S'))
    df_sqqq = df_sqqq.with_columns(pl.col('date').str.to_datetime('%Y-%m-%d %H:%M:%S'))
    
    df = df_tqqq.join(df_sqqq, on='date', how='inner', suffix='_sqqq').sort('date')
    
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

def run_backtest():
    df = load_data()
    
    atr_period = 9
    atr_mult = 27.151930794926393
    threshold = 0.0006317922692613839
    
    df_atr = df.with_columns(atr=pl.col('tr').rolling_mean(window_size=atr_period))
    
    closes_tqqq = df['close'].fill_null(strategy="forward").to_numpy()
    opens_tqqq = df['open'].to_numpy()
    closes_sqqq = df['close_sqqq'].fill_null(strategy="forward").to_numpy()
    opens_sqqq = df['open_sqqq'].to_numpy()
    
    vwaps = df['vwap'].fill_null(0.0).to_numpy()
    atrs = df_atr['atr'].fill_null(0.0).to_numpy()
    dates_int = df['day'].cast(pl.Int32).to_numpy()
    
    signal_raw = calc_vwap_atr(closes_tqqq, vwaps, atrs, dates_int, atr_mult, threshold)
    pos = shift_signal_over_day(signal_raw, dates_int)
    
    initial_cap = 25000.0
    
    print("Running restricted execution engine (MAX 1 BUY PER ASSET PER DAY)...")
    daily_dates_int, daily_capital = simulate_restricted_dual_trading(
        dates_int, opens_tqqq, closes_tqqq, opens_sqqq, closes_sqqq, pos, initial_cap
    )
    
    bnh_shares = initial_cap / opens_tqqq[0]
    
    df_daily = df.group_by('day').tail(1).sort('day')
    daily_closes_tqqq = df_daily['close'].to_numpy()
    bnh_capital = daily_closes_tqqq * bnh_shares
    
    sys_ret = (daily_capital[-1] / initial_cap - 1) * 100
    bnh_ret = (bnh_capital[-1] / initial_cap - 1) * 100
    
    print(f"\n--- Backtest Results ---")
    print(f"Algorithm Return (Cash Restricted): {sys_ret:.2f}%")
    print(f"Buy & Hold Return (TQQQ): {bnh_ret:.2f}%")
    
    plot_dates = [datetime(1970, 1, 1) + timedelta(days=int(d)) for d in daily_dates_int]
    
    sys_ret_series = (daily_capital / initial_cap - 1) * 100
    bnh_ret_series = (bnh_capital / initial_cap - 1) * 100
    
    plt.figure(figsize=(12, 6))
    plt.plot(plot_dates, sys_ret_series, label=f'Cash Restricted Swap System (Return: {sys_ret:.1f}%)', color='red', linewidth=2)
    plt.plot(plot_dates, bnh_ret_series, label=f'Buy & Hold TQQQ Baseline', color='gray', linestyle='dashed')
    
    plt.title('TQQQ/SQQQ Swap (Cash Account Rule: 1 Buy per Asset/Day)', fontsize=14)
    plt.ylabel('Cumulative Return (%)', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=11)
    
    output_png = r"C:\Users\plane\.gemini\antigravity\brain\3a4c6ac4-8611-41ed-be89-cc3cc26244ba\cash_restricted_test.png"
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    print(f"Plot saved to {output_png}")

if __name__ == "__main__":
    run_backtest()
