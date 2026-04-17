import polars as pl
import numba
import numpy as np
import matplotlib.pyplot as plt

@numba.njit
def simulate_trading_with_equity_curve(dates, opens, closes, pos, initial_capital=25000.0, commission=0.0005):
    capital = initial_capital
    shares = 0.0
    curr_pos = 0
    n = len(opens)
    equity_curve = np.zeros(n)

    for i in range(n):
        target_pos = pos[i]
        if i == n - 1 or dates[i+1] != dates[i]:
            target_pos = 0

        if target_pos != curr_pos:
            price = closes[i] if (target_pos == 0 and (i == n-1 or dates[i+1] != dates[i])) else opens[i]
            if curr_pos == 1:
                capital += shares * price
                capital -= shares * commission
                shares = 0
            elif curr_pos == -1:
                capital -= shares * price
                capital -= shares * commission
                shares = 0
            if target_pos == 1:
                shares = capital / price
                capital -= shares * price
                capital -= shares * commission
            elif target_pos == -1:
                shares = capital / price
                capital += shares * price
                capital -= shares * commission
            curr_pos = target_pos

        current_equity = capital
        if curr_pos == 1:
            current_equity += shares * closes[i]
        elif curr_pos == -1:
            current_equity -= shares * closes[i]

        equity_curve[i] = current_equity

    return equity_curve

def run_vwap_with_atr_exit(file_path, atr_period=21, atr_mult=8.0, threshold=0.003):
    df = pl.read_csv(file_path)
    df = df.with_columns(pl.col('date').str.to_datetime('%Y-%m-%d %H:%M:%S'))
    df = df.unique(subset=['date'], keep='first').sort('date')

    # RTH filter: 9:30 to 15:30 (Webull forced liquidation)
    df = df.filter(
        ((pl.col('date').dt.hour() == 9) & (pl.col('date').dt.minute() >= 30)) |
        ((pl.col('date').dt.hour() >= 10) & (pl.col('date').dt.hour() < 15)) |
        ((pl.col('date').dt.hour() == 15) & (pl.col('date').dt.minute() <= 30))
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
    df = df.with_columns(atr=pl.col('tr').rolling_mean(window_size=atr_period))

    closes = df['close'].fill_null(strategy="forward").to_numpy()
    vwaps = df['vwap'].fill_null(0.0).to_numpy()
    atrs = df['atr'].fill_null(0.0).to_numpy()
    dates_int = df['day'].cast(pl.Int32).to_numpy()

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

    signal_raw = calc_vwap_atr(closes, vwaps, atrs, dates_int, atr_mult, threshold)

    df = df.with_columns(signal=pl.Series('signal', signal_raw))
    df = df.with_columns(pos=pl.col('signal').shift(1).fill_null(0).over('day'))

    opens = df['open'].to_numpy()
    pos = df['pos'].to_numpy()

    equity_curve = simulate_trading_with_equity_curve(dates_int, opens, closes, pos)
    df = df.with_columns(equity=pl.Series('equity', equity_curve))

    df_daily = df.group_by('day').agg([
        pl.col('equity').last().alias('equity'),
        pl.col('close').last().alias('close'),
        pl.col('open').first().alias('first_open')
    ])
    df_daily = df_daily.sort('day')

    return df_daily

if __name__ == "__main__":
    df_daily = run_vwap_with_atr_exit('tqqq_1min_historical_data.csv')

    initial_cap = 25000.0
    dates = df_daily['day'].to_list()

    # Strategy Returns
    strategy_returns = [(x / initial_cap - 1) * 100 for x in df_daily['equity'].to_list()]

    # Buy and Hold Returns
    first_open_price = df_daily['first_open'][0]
    bnh_returns = [(close_price / first_open_price - 1) * 100 for close_price in df_daily['close'].to_list()]

    plt.figure(figsize=(10, 6))
    plt.plot(dates, strategy_returns, label='TQQQ VWAP+ATR (Webull 15:30)', color='blue')
    plt.plot(dates, bnh_returns, label='TQQQ Buy & Hold', color='orange', alpha=0.75, linestyle='--')

    plt.title('Total Return (%) Over Time: Strategy vs Buy & Hold')
    plt.xlabel('Date')
    plt.ylabel('Total Return (%)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('return_plot.png')
    print("Saved return_plot.png")
