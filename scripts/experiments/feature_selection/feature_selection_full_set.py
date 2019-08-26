
from api import *

history = Chart('C:/Users/User/Desktop/darmstadt/Master-Thesis/Data/Histories/btc_usd_jan2012-jun2018.csv') \
    .fill_missing_data('min', 'interpolate')\
    .upscale('60min')

# TODO Indicators!!!

macd = Indicator('macd', 'macd_main_line', 'macd_signal_line')\
        .transform(lambda main_line, signal_line: signal_line - main_line, 'macd')
trix = Indicator('trix', 'trix_main_line', 'trix_signal_line')\
        .transform(lambda main_line, signal_line: signal_line - main_line, 'trix')
adx = Indicator('adx', 'adx')
rsi = Indicator('rsi', 'rsi')
mfi = Indicator('mfi', 'mfi')
wpr = Indicator('wpr', 'wpr')
ao = Indicator('ao', 'ao')
uo = Indicator('uo', 'uo')
stoch = Indicator('stoch', 'stoch_k', 'stoch_d')
bbands = Indicator('bbands', 'bb_u', 'bb_m', 'bb_l')\
        .transform(lambda upper, middle, lower: upper - lower, 'bbands')
kltch = Indicator('kltch', 'klt_u', 'klt_m', 'klt_l')\
        .transform(lambda upper, middle, lower: upper - lower, 'kltch')

features = Features(history) \
    .label_data('ternary', 1, 0, 0)\
    .add_features(macd, trix, adx, rsi, mfi, wpr, ao, uo, stoch, bbands, kltch)\
    .prune_features(5)

print(features.X.columns)

