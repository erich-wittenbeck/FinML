
from lib.indicators.moving_averages import exponential_moving_average as ema

def on_balance_volume(df):
    """
    OBV technical indicator

    :param df: Chart data, as pandas dataframe
    :return: A pandas series.
    """

    close_values = df['close']
    volumes = df['volume']

    prev_close_values = close_values.shift(1).fillna(close_values.iloc[0])

    pos_index = close_values > prev_close_values
    neg_index = close_values < prev_close_values

    summands = volumes*0
    summands[pos_index] = volumes[pos_index]
    summands[neg_index] = volumes[neg_index]*(-1)

    obv = summands.cumsum()

    return obv

def force_index(df, lookback=14):
    """
    FIDX technical indicator

    :param df: Chart data, as pandas dataframe
    :param lookback: Optional. Look-back period. Default 14
    :return: A pandas series
    """

    close_values = df['close']
    volumes = df['volume']

    prev_close_values = close_values.shift(1).fillna(0)

    base = (close_values - prev_close_values)*volumes

    fidx = ema(base, lookback)

    return fidx

def chaikin_money_flow(df, lookback=20):
    """
    CMF technical indicator

    :param df: Chart data, as pandas dataframe
    :param lookback: Optional. Look-back period. Default 20
    :return: A pandas series
    """

    high_values = df['high']
    low_values = df['low']
    close_values = df['close']
    volumes = df['volume']

    mf_multiplier = ((close_values - low_values)-(high_values - close_values))/(high_values - low_values)
    mf_volume = mf_multiplier*volumes

    cmf = (mf_volume.rolling(window=lookback, min_periods=0).sum().fillna(0))/(volumes.rolling(window=lookback, min_periods=0).sum().fillna(0))

    return cmf

def accumulation_distribution_line(df, lookback=20):
    """
    ADL technical indicator

    :param df: Chart data, as pandas dataframe
    :param lookback: Optional. Look-back period. Default 20
    :return: A pandas series
    """

    high_values = df['high']
    low_values = df['low']
    close_values = df['close']
    volumes = df['volume']

    mf_multiplier = ((close_values - low_values) - (high_values - close_values)) / (high_values - low_values)
    mf_volume = mf_multiplier * volumes

    adl = ema(mf_volume, lookback)

    return adl