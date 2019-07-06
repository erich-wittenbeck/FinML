from api import *

from time import time

start_time = time()

history = Chart('C:/Users/User/Desktop/darmstadt/Master-Thesis/Data/Histories/btc_usd_jan2012-jun2018.csv', True, 'min', 'interpolate') \
    .upscale('60min') \
    .slice('2017-03', '2018-06')\
    .smooth(0.5, 'open', 'close')

feature_matrix = Features(history)\
    .label_data('ternary', 'close', 24, 0.05, 0.05)\
    .add_indicator('rsi', 'rsi', lookback=5)\
    .add_indicator('macd', 'macd', return_as='delta')\
    .add_indicator('trix', 'trix', return_as='trix_only')\
    .add_indicator('uo', 'uo')\
    .add_indicator(['bbands_upper_delta', 'bbands_lower_delta'], 'bbands', return_as='deltas')\
    .add_indicator('adx', 'adx')\
    .add_indicator('obv', 'obv')

print(feature_matrix.X.isnull().values.any())
print(feature_matrix.X)

end_time = time()

print('time elapsed: ' + str(end_time - start_time) + ' seconds')