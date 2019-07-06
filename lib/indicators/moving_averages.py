from lib.utility.statistics import avg

def simple_moving_average(values, lookback):
    return values.rolling(window=lookback, min_periods=0).mean()

def smoothed_moving_average(values, lookback):
    result = []
    for i in range(len(values)):
        if i <= lookback:
            result.append(simple_moving_average(values[:i + 1], lookback)[-1])
        else:
            prev_sum = sum(values[i-lookback:i])
            prev_smma = result[-1]
            current_smma = (prev_sum - prev_smma + values[i])/lookback
            result.append(current_smma)
    return result

def exponential_moving_average(values, lookback):
    return values.ewm(span=lookback, min_periods=0).mean()
