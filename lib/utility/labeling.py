
def _percent_change(old_val, new_val):
    one_percent = old_val / 100
    difference = new_val - old_val

    return (difference / one_percent) / 100

def binary_labels(df, n, column='close'):
    """
    Labeling function for outputting binary labels:

    - Buy (1)
    - Sell (-1)

    :param df: Chart data, as pandas dataframe
    :param n: Look-ahead period, integer
    :param column: Optional. Which OHLCV-column to compute the labels from. Default 'close'
    :return: A list of binary labels.
    """
    series = df[column]

    return [1 if delta >= 0 else -1
            for delta in [_percent_change(series[t], series[t + n]) # (series[t + n] - series[t]) * (100 / series[t - n])
                          if t+n < len(series) else 0
                          for t in range(len(series))]]

def ternary_labels(df, n, buy_margin, sell_margin, column='close'):
    """
    Labeling function for outputting ternary labels:

    - Buy (1)
    - Hold (0)
    - Sell (-1)

    :param df: Chart data, as pandas dataframe
    :param n: Look-ahead period, integer
    :param buy_margin: Margin above 0 for when to put out a buy-signal
    :param sell_margin: Margin below 0 for when to put out a sell-signal
    :param column: Optional. Which OHLCV-column to compute the labels from. Default 'close'
    :return: A list of terbary labels.
    """
    series = df[column]

    result = [1 if delta > buy_margin else -1 if delta < sell_margin*-1 else 0 for delta in [_percent_change(series[t], series[t + n]) if t+n < len(series) else 0 for t in range(len(series))]]

    return result

def quartary_labels(df, n, strong_buy_margin, strong_sell_margin, column='close'):
    """
    Labeling function for outputting quartary labels:

    - Strong Buy (2)
    - Weak Buy (1)
    - Weak Sell(-1)
    - Strong Sell (-2)

    :param df: Chart data, as pandas dataframe
    :param n: Look-ahead period, integer
    :param strong_buy_margin: Margin above 0 for when to put out a strong buy-signal
    :param strong_sell_margin: Margin below for when to put out a strong sell-signal
    :param column: Optional. Which OHLCV-column to compute the labels from. Default 'close'
    :return: A list of quartary labels.
    """
    series = df[column]

    return [2 if delta >= strong_buy_margin else
            1 if delta >= 0 else
            -1 if delta >= strong_sell_margin*-1 else
            -2 for delta in [_percent_change(series[t], series[t + n])
                             if t+n < len(series) else 0
                             for t in range(len(series))]]

def pentary_labels(df, n, strong_buy_margin, weak_buy_margin, weak_sell_margin, strong_sell_margin, column='close'):
    """
    Labeling function for outputting pentary labels:

    - Strong Buy (2)
    - Weak Buy (1)
    - Hold (0)
    - Weak Sell(-1)
    - Strong Sell (-2)

    :param df: Chart data, as pandas dataframe
    :param n: Look-ahead period, integer
    :param strong_buy_margin: Margin above 0 for when to put out a strong buy-signal
    :param weak_buy_margin: Margin above 0 for when to put out a weak buy-signal
    :param weak_sell_margin: Margin below for when to put out a strong weak sell-signal
    :param strong_sell_margin: Margin below for when to put out a strong strong sell-signal
    :param column: Optional. Which OHLCV-column to compute the labels from. Default 'close'
    :return: A list of pentary labels.
    """
    series = df[column]

    return [2 if delta >= strong_buy_margin else
            1 if delta >= weak_buy_margin else
            0 if delta >= weak_sell_margin*-1 else
            -1 if delta >= strong_sell_margin*-1 else
            -2 for delta in [_percent_change(series[t], series[t + n])
                            if t+n < len(series) else 0
                            for t in range(len(series))]]

