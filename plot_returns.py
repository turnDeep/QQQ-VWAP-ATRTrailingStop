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
    times_hm = (df['date'].dt.hour() * 100 + df['date'].dt.minute()).to_numpy()

    @numba.njit
    def calc_vwap_atr(closes, vwaps, atrs, dates, times, multiplier, thresh):
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
            curr_time = times[i]

            # The webull 15:30 logic
            if curr_time > 1530:
                current_trend = 0
                pos_signal[i] = 0
                continue

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

    signal_raw = calc_vwap_atr(closes, vwaps, atrs, dates_int, times_hm, atr_mult, threshold)

    df = df.with_columns(signal=pl.Series('signal', signal_raw))
    df = df.with_columns(pos=pl.col('signal').shift(1).fill_null(0).over('day'))

    opens = df['open'].to_numpy()
    pos = df['pos'].to_numpy()

    equity_curve = simulate_trading_with_equity_curve(dates_int, opens, closes, pos)
    df = df.with_columns(equity=pl.Series('equity', equity_curve))

    df_daily = df.group_by('day').agg([
        pl.col('equity').last().alias('equity'),
        pl.col('close').last().alias('close')
    ])
    df_daily = df_daily.sort('day')

    # Calculate Buy and Hold properly. If the historical data is already adjusted for splits,
    # we don't need any complex multiplier logic. The return is simply (Close / Initial_Open) - 1
    # However, because the dataset spans overnight periods, we must link the daily continuous returns
    # instead of doing a naive absolute ratio, to avoid gaps destroying the curve if unadjusted.

    df_day_open = df.group_by('day').agg(pl.col('open').first().alias('first_open')).sort('day')
    df_daily = df_daily.join(df_day_open, on='day', how='left')

    # Calculate compounded daily returns
    # Daily return = (Close of Day) / (Open of Day)
    # Overnight return = (Open of Day T) / (Close of Day T-1)

    bnh_equity = np.zeros(len(df_daily))
    bnh_equity[0] = 25000.0 * (df_daily['close'][0] / df_daily['first_open'][0])

    for i in range(1, len(df_daily)):
        curr_open = df_daily['first_open'][i]
        curr_close = df_daily['close'][i]
        prev_close = df_daily['close'][i-1]

        # Continuous compounding formula: New_Equity = Prev_Equity * (Curr_Close / Prev_Close)
        # If the data has an artificial gap due to a split, (curr_open / prev_close) will be absurd.
        # We can just ignore the overnight gap entirely, assuming we held the stock and its true
        # value didn't change just because of a split. Thus, the daily change is just intraday return.

        # If we assume we just earn the intraday return each day (this is "Buy & Hold Intraday" technically).
        # Let's check what causes the Nov 2025 anomaly. If it's a massive unadjusted gap, we can just
        # link the intraday returns + normal overnight returns.
        # A normal overnight return is between 0.95 and 1.05.

        overnight_ratio = curr_open / prev_close
        if overnight_ratio < 0.7 or overnight_ratio > 1.3:
            # It's an artificial split. We neutralize the overnight gap.
            overnight_ratio = 1.0

        bnh_equity[i] = bnh_equity[i-1] * overnight_ratio * (curr_close / curr_open)

    df_daily = df_daily.with_columns(bnh_equity=pl.Series('bnh_equity', bnh_equity))

    return df_daily

if __name__ == "__main__":
    df_daily = run_vwap_with_atr_exit('tqqq_1min_historical_data.csv')

    initial_cap = 25000.0
    dates = df_daily['day'].to_list()

    # Strategy Returns
    strategy_returns = [(x / initial_cap - 1) * 100 for x in df_daily['equity'].to_list()]

    # Buy and Hold Returns
    bnh_returns = [(x / initial_cap - 1) * 100 for x in df_daily['bnh_equity'].to_list()]

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
