
import pandas as pd

from lib.indicators.moving_averages import simple_moving_average as sma, exponential_moving_average as ema

def average_true_range(df, lookback=14):
    """
    ATR technical indicator

    :param df: Chart data, as pandas dataframe
    :param lookback: Optional. Look-back period. Default 14
    :return: A pandas series
    """

    high_values = df['high']
    low_values = df['low']
    close_values = df['close']

    a = high_values - low_values
    b = (high_values - close_values.shift(1).fillna(0)).abs()
    c = (low_values - close_values.shift(1).fillna(0)).abs()

    true_ranges = pd.concat([a,b,c], axis=1).max(axis=1)

    atr = ema(true_ranges, lookback)

    return atr

def mass_index(df, short=9, long=25):
    """
    AO technical indicator

    :param df: Chart data, as pandas dataframe
    :param short: Optional. Short look-back period. Default 9
    :param long: Optional. Long look-back period. Default 25
    :return: A pandas series
    """

    high_values = df['high']
    low_values = df['low']

    high_low_delta = high_values - low_values

    ema_single = ema(high_low_delta, short)
    ema_double = ema(ema_single, short)

    ema_div = ema_single/ema_double

    midx = ema_div.rolling(window=long, min_periods=0).sum().fillna(0)

    return midx

def bollinger_bands(df, column='close', lookback=14, return_as='total_delta'):
    """
    BBANDS technical indicator.

    Can be returned as

    - Delta between upper and lower band
    - Delta between upper and middle or middle and lower band
    - Either of the three bands

    :param df: Chart data, as pandas dataframe
    :param column: Optional. Column upon which indicator is to be computed. Default 'close'
    :param lookback: Optional. Look-back period. Default 14
    :param return_as: Optional. How to return the indicator. Can be 'total_delta', 'upper_delta', 'lower_delta', 'upper_band', 'middle_band' and 'lower_band'. Default 'total_delta'
    :return: A pandas series.
    """

    values = df[column]

    mband = sma(values, lookback)
    sdevs = mband.rolling(window=lookback, min_periods=0).var()**0.5

    # The first values will be NaN, since var requires at least 1 period, regardless of min_periods

    uband = (mband + sdevs).fillna(mband.iloc[0])
    lband = (mband - sdevs).fillna(mband.iloc[0])

    if return_as == 'total_delta':
        return uband - lband
    elif return_as == 'upper_delta':
        return uband - mband
    elif return_as == 'lower_delta':
        return mband - lband
    elif return_as == 'upper_band':
        return uband
    elif return_as == 'lower_band':
        return lband
    elif return_as == 'middle_band':
        return mband
    else:
        raise ValueError("return_as: expected to be either 'total_delta', 'upper_delta' or 'lower_delta', but was " + str(return_as) + " instead!")

def keltner_channel(df, column='close', lookback=14, return_as='total_delta'):
    """
    KLTCH technical indicator.

    Can be returned as

    - Delta between upper and lower band
    - Delta between upper and middle or middle and lower band
    - Either of the three bands

    :param df: Chart data, as pandas dataframe
    :param column: Optional. Column upon which indicator is to be computed. Default 'close'
    :param lookback: Optional. Look-back period. Default 14
    :param return_as: Optional. How to return the indicator. Can be 'total_delta', 'upper_delta', 'lower_delta', 'upper_band', 'middle_band' and 'lower_band'. Default 'total_delta'
    :return: A pandas series.
    """

    values = df[column]

    mband = ema(values, lookback)
    double_atr = average_true_range(df, lookback)*2

    uband = mband + double_atr
    lband = mband - double_atr

    if return_as == 'total_delta':
        return uband - lband
    elif return_as == 'upper_delta':
        return uband - mband
    elif return_as == 'lower_delta':
        return mband - lband
    elif return_as == 'upper_band':
        return uband
    elif return_as == 'lower_band':
        return lband
    elif return_as == 'middle_band':
        return mband
    else:
        raise ValueError("return_as: expected to be either 'total_delta', 'upper_delta' or 'lower_delta', but was " + str(return_as) + " instead!")