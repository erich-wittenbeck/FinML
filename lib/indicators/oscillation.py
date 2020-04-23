
import pandas as pd

from lib.indicators.moving_averages import simple_moving_average as sma, exponential_moving_average as ema

def relative_strength_index(df, column='close', lookback=14):
    """
    RSI technical indicator

    :param df: Chart data, as pandas dataframe
    :param column: Optional. Column upon which indicator is to be computed. Default 'close'
    :param lookback: Optional. Look-back period. Default 14
    :return: A pandas series
    """

    values = df[column]

    deltas = values.diff()
    gain_pos, loss_pos = deltas > 0, deltas < 0

    gains, losses = deltas*0, deltas*0
    gains[gain_pos], losses[loss_pos] = deltas[gain_pos], deltas[loss_pos]*-1

    avg_gain, avg_loss = ema(gains, lookback=lookback), ema(losses, lookback=lookback)

    return (100 - (100/(1 + avg_gain/avg_loss))).fillna(0)

def money_flow_index(df, lookback=14):
    """
    MFI technical indicator

    :param df: Chart data, as pandas dataframe
    :param lookback: Optional. Look-back period. Default 14
    :return: A pandas series
    """

    high_values = df['high']
    low_values = df['low']
    close_values = df['close']
    volumes = df['volume']

    avg_values = (high_values + low_values + close_values)/3
    avg_values_prev = avg_values.shift(1).fillna(0)

    money_flow = avg_values*volumes

    pos_flow, neg_flow = avg_values*0, avg_values*0

    pos_flow[avg_values > avg_values_prev] = money_flow[avg_values > avg_values_prev]
    pos_flow = pos_flow.rolling(window=lookback, min_periods=0).sum().fillna(0)

    neg_flow[avg_values_prev > avg_values] = money_flow[avg_values_prev > avg_values]
    neg_flow = neg_flow.rolling(window=lookback, min_periods=0).sum().fillna(0)

    money_flow_ratio = pos_flow/neg_flow

    mfi = 100 - 100/(1 + money_flow_ratio)

    return mfi


def williams_percent_range(df, lookback=14):
    """
    WPR technical indicator

    :param df: Chart data, as pandas dataframe
    :param lookback: Optional. Look-back period. Default 14
    :return: A pandas series
    """

    high_values = df['high']
    low_values = df['low']
    close_values = df['close']

    highest_highs = high_values.rolling(window=lookback, min_periods=0).max().fillna(0)
    lowest_lows = low_values.rolling(window=lookback, min_periods=0).min().fillna(0)

    wpr = (highest_highs - close_values)/(highest_highs - lowest_lows) * (-100)

    return wpr

def awesome_oscillator(df, short=5, long=34):
    """
    AO technical indicator

    :param df: Chart data, as pandas dataframe
    :param short: Optional. Short look-back period. Default 5
    :param long: Optional. Long look-back period. Default 34
    :return: A pandas series
    """

    high_values = df['high']
    low_values = df['low']

    mid_points = (high_values + low_values)/2

    ao = sma(mid_points, short) - sma(mid_points, long)

    return ao

def ultimate_oscillator(df, short=7, medium=14, long=28):
    """
    UO technical indicator

    :param df: Chart data, as pandas dataframe
    :param short: Optional. Short look-back period. Default 7
    :param medium: Optiona. Medium look-back period. Default 14
    :param long: Optional. Long look-back period. Default 28
    :return: A pandas series
    """

    high_values = df['high']
    low_values = df['low']
    close_values = df['close']

    previous_close_values = close_values.shift(1).fillna(0)
    min_low_or_prev_close = pd.concat([low_values, previous_close_values],axis=1).min(axis=1)
    max_high_or_prev_close = pd.concat([high_values, previous_close_values], axis=1).max(axis=1)

    buying_pressures = close_values - min_low_or_prev_close
    true_ranges = max_high_or_prev_close - min_low_or_prev_close

    bp_sum = lambda x : buying_pressures.rolling(window=x, min_periods=0).sum().fillna(0)
    tr_sum = lambda x : true_ranges.rolling(window=x, min_periods=0).sum().fillna(0)

    bp_sum_short, bp_sum_medium, bp_sum_long = bp_sum(short), bp_sum(medium), bp_sum(long)
    tr_sum_short, tr_sum_medium, tr_sum_long = tr_sum(short), tr_sum(medium), tr_sum(long)

    avg_short, avg_medium, avg_long = bp_sum_short/tr_sum_short, bp_sum_medium/tr_sum_medium, bp_sum_long/tr_sum_long

    factor_short, factor_medium = long // short, long // medium

    uo = 100 * (factor_short*avg_short + factor_medium*avg_medium + avg_long)/(factor_short + factor_medium + 1)

    return uo

def stochastic_oscillator(df, lookback_k=14, lookback_d=3, return_as='k'):
    """
    STOCH technical indicator

    Can be returned in two variants: K and D

    :param df: Chart data, as pandas dataframe
    :param lookback_k: Optional. Look-back period for the K-line (longterm). Default 14
    :param lookback_d: Optional. Look-back period for the D-line (shortterm). Default 3
    :param return_as: Optional. What to return. Can be 'k' or 'd'. Default 'k'
    :return: A pandas series
    """
    high_values = df['high']
    low_values = df['low']
    close_values = df['close']

    highest_highs = high_values.rolling(window=lookback_k, min_periods=0).max().fillna(0)
    lowest_lows = low_values.rolling(window=lookback_k, min_periods=0).min().fillna(0)

    percent_k = sma(100 * (close_values - lowest_lows) / (highest_highs - lowest_lows), lookback_k)
    percent_d = sma(percent_k, lookback_d)

    if return_as == 'k':
        return percent_k
    elif return_as == 'd':
        return percent_d
    else:
        raise ValueError("return_as: expected to be either 'k' or 'd', but was " + str(return_as) + " instead!")
