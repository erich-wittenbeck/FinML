
import pandas as pd

from lib.indicators.moving_averages import exponential_moving_average as ema
from lib.indicators.volatility import average_true_range as atr

def moving_average_convergence_divergence(df, column='close', short=12, long=26, medium=9, return_as='delta'):
    """
    MACD technical indicator.

    Is returned either as

    - The delta between main- and signal-line
    - The main-line
    - The signal-line

    :param df: Chart data, as pandas dataframe
    :param column: Optional. Column upon which indicator is to be computed. Default 'close'
    :param short: Optional. Short look-back period. Default 12
    :param long: Optional. Long look-back period. Default 26
    :param medium: Optional. Medium look-back period. Default 9
    :param return_as: Optional. How to return the indicator. Can be 'delta', 'main' or 'signal'. Default 'delta'
    :return: A pandas series
    """

    values = df[column]

    # The basic EMAs
    short_ema = ema(values, short)
    long_ema = ema(values, long)

    # The lines calculated from the EMAs
    main_line = short_ema - long_ema
    signal_line = ema(main_line, medium)

    if return_as == 'delta':
        return signal_line - main_line
    elif return_as == 'main':
        return main_line
    else:
        return signal_line

def triple_exponential_moving_average(df, column='close', lookback_trix=14, lookback_signal=9, return_as='delta'):
    """
    TRIX technical indicator.

    Is returned either as

    - The delta between main- and signal-line
    - The main-line
    - The signal-line

    :param df: Chart data, as pandas dataframe
    :param column: Optional. Column upon which indicator is to be computed. Default 'close'
    :param lookback_trix: Optional. The lookback-period for the main-line. Default 14
    :param lookback_signal: Optional. The lookback-period for the signal-line. Default9
    :param return_as: Optional. How to return the indicator. Can be 'delta', 'main' or 'signal'. Default 'delta'
    :return: A pandas series
    """

    values = df[column]

    single_ema = ema(values, lookback_trix)
    double_ema = ema(single_ema, lookback_trix)
    triple_ema = ema(double_ema, lookback_trix)
    triple_ema_prev = triple_ema.shift(1).fillna(triple_ema.iloc[0])

    trix = 100 * (triple_ema - triple_ema_prev)/(triple_ema_prev)
    signal = ema(trix, lookback_signal)

    if return_as == 'delta':
        return signal - trix
    elif return_as == 'main':
        return trix
    else:
        return signal

def average_directional_index(df, lookback=14):
    """
    ADX technical indicator

    :param df: Chart data, as pandas dataframe
    :param lookback: Optional. Look-back period. Default 14
    :return: A pandas series
    """

    high_values = df['high']
    low_values = df['low']

    up_move = (high_values - high_values.shift(1).fillna(0))
    dn_move = (low_values.shift(1).fillna(0) - low_values)

    up_index = (up_move > dn_move) & (up_move > 0)
    dn_index = (dn_move > up_move) & (dn_move > 0)

    plus_dm, minus_dm = high_values*0, low_values*0

    plus_dm[up_index] = up_move[up_index]
    minus_dm[dn_index] = dn_index[dn_index]

    this_atr = atr(df, lookback)

    plus_di = 100*(ema((plus_dm/this_atr), lookback))
    minus_di = 100*(ema((minus_dm/this_atr), lookback))

    adx = 100 * (ema(((plus_di - minus_di)/(plus_di + minus_di)).abs(), lookback))

    return adx

def vortex_indicator(df, lookback=14):
    """
    Vortex technical indicator

    :param df: Chart data, as pandas dataframe
    :param lookback: Optional. Look-back period. Default 14
    :return: A pandas series
    """

    high_values = df['high']
    low_values = df['low']
    close_values = df['close']

    prev_high_values = high_values.shift(1).fillna(0)
    prev_low_values = low_values.shift(1).fillna(0)
    prev_close_values = close_values.shift(1).fillna(0)

    # Vortex movements

    plus_vm = (high_values - prev_low_values).abs()
    minus_vm = (low_values - prev_high_values).abs()

    plus_vm_sum = plus_vm.rolling(window=lookback, min_periods=0).sum().fillna(0)
    minus_vm_sum = minus_vm.rolling(window=lookback, min_periods=0).sum().fillna(0)

    # True ranges

    a = high_values - low_values
    b = (high_values - prev_close_values).abs()
    c = (low_values - prev_close_values).abs()

    true_ranges = pd.concat([a, b, c], axis=1).max(axis=1)
    tr_sum = true_ranges.rolling(window=lookback, min_periods=0).sum().fillna(0)

    # Vortex indicator through normalizing vortex movements

    plus_vi = plus_vm_sum/tr_sum
    minus_vi = minus_vm_sum/tr_sum

    return plus_vi - minus_vi