import polars as pl
import numba
import numpy as np

@numba.njit
def simulate_trading(dates, opens, closes, pos, initial_capital=25000.0, commission=0.0005):
    capital = initial_capital
    shares = 0.0
    curr_pos = 0
    n = len(opens)
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
    return capital

def run_vwap_with_atr_exit(file_path, atr_period=21, atr_mult=8.0, threshold=0.003):
    df = pl.read_csv(file_path)
    df = df.with_columns(pl.col('date').str.to_datetime('%Y-%m-%d %H:%M:%S'))
    df = df.unique(subset=['date'], keep='first').sort('date')

    # RTH filter: 9:30 to 15:59 (the absolute highest return was before we added the Webull 15:30 close rule)
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

    return simulate_trading(dates_int, opens, closes, pos)

if __name__ == "__main__":
    tqqq_cap = run_vwap_with_atr_exit('tqqq_1min_historical_data.csv', atr_period=21, atr_mult=8.0, threshold=0.003)
    print(f"Original High-Return Logic TQQQ (Mult 8.0): {(tqqq_cap / 25000 - 1) * 100:.2f}%")
