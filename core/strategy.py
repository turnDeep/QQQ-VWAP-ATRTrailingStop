from __future__ import annotations

import numpy as np
import pandas as pd
import numba

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

def calculate_intraday_indicators(df: pd.DataFrame, atr_period: int) -> pd.DataFrame:
    """
    Given a DataFrame with columns: open, high, low, close, volume, timestamp
    Calculates intraday VWAP and ATR and returns a new DataFrame.
    """
    if df.empty:
        return df

    work = df.copy()
    work['date'] = pd.to_datetime(work['timestamp']).dt.tz_convert('America/New_York')
    work['day'] = work['date'].dt.date
    
    # Calculate True Range for ATR
    work['prev_close'] = work['close'].shift(1)
    work.loc[work['day'] != work['day'].shift(1), 'prev_close'] = np.nan
    work['tr1'] = work['high'] - work['low']
    work['tr2'] = (work['high'] - work['prev_close']).abs()
    work['tr3'] = (work['low'] - work['prev_close']).abs()
    work['tr'] = work[['tr1', 'tr2', 'tr3']].max(axis=1)
    work['atr'] = work.groupby('day')['tr'].transform(lambda x: x.rolling(atr_period, min_periods=1).mean())
    
    # Calculate VWAP
    work['typical_price'] = (work['high'] + work['low'] + work['close']) / 3.0
    work['vp'] = work['typical_price'] * work['volume']
    work['cum_vp'] = work.groupby('day')['vp'].cumsum()
    work['cum_vol'] = work.groupby('day')['volume'].cumsum()
    work['vwap'] = np.where(work['cum_vol'] > 0, work['cum_vp'] / work['cum_vol'], work['typical_price'])
    
    return work.drop(columns=['date', 'prev_close', 'tr1', 'tr2', 'tr3', 'tr', 'vp', 'cum_vp', 'cum_vol', 'typical_price'])

