
from api import *
from time import time

history = Chart('C:/Users/User/Desktop/darmstadt/Master-Thesis/Data/Histories/btc_usd_jan2012-jun2018.csv', True, 'min', 'interpolate') \
    .upscale('60min') \
    .slice('2016-06', '2018-06')

start_time = time()

rsi = Indicator('rsi', 'rsi')
macd = Indicator('macd', 'macd_main_line', 'macd_signal_line')
uo = Indicator('uo', 'uo')
stoch = Indicator('stoch', 'stoch', return_as='delta')

features = Features(history)\
        .add_features(rsi, macd, uo, stoch)\
        .normalize_features('min_max', ('macd_main_line', 'macd_signal_line'), lookback=14, only_positive=False)\
        .normalize_features(lambda x : x/100, ('rsi', 'uo', 'stoch'))\
        .rolling_window(0, 3, 'uo', 'stoch')\
        .label_data('ternary', 24, 0.05, 0.05)

end_time = time()

print(features.X.tail(10))

print('time elapsed: ' + str(end_time - start_time) + ' seconds')
