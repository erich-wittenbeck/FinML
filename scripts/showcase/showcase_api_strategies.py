
from api import *

history = Chart('C:/Users/User/Desktop/darmstadt/Master-Thesis/Data/Histories/btc_usd_jan2012-jun2018.csv') \
    .fill_missing_data('min', 'interpolate')\
    .upscale('60min') \
    .slice('2016-06', '2018-06')

n = 24

rsi = Indicator('rsi', 'rsi')
macd = Indicator('macd', 'macd_main_line', 'macd_signal_line')
bbands = Indicator('bbands', 'bbands_upper', 'bbands_mid', 'bbands_low')

feature_matrix = Features(history)\
    .label_data('ternary', n, 0.0125, 0.0125)\
    .add_features(rsi, macd, bbands)

train_data, test_data = feature_matrix.split('2017-09')

some_random_forest = Classifier('some_random_forest', 'randf') \
        .set_hyper_parameters(n_estimators=[100],
                              max_features=[1, 2, 3, 4],
                              bootstrap=[True, False]) \
        .configure_hpo('f1_macro', n_jobs=3, verbose=2) \
        .train(train_data)

strat_rsi_macd_str = """
if rsi > 70:
    if macd_main_line < macd_signal_line:
        __signal__ = -3
    else:
        __signal__ = -2
elif rsi < 30:
    if macd_main_line > macd_signal_line:
        __signal__ = 3
    else:
        __signal__ = 2
else:
    if macd_main_line < macd_signal_line:
        __signal__ = -1
    else:
        __signal__ = 1
"""

strat_rsi_macd_1 = Strategy('strat_rsi_macd_1', strat_rsi_macd_str)

simulator = Simulator()\
    .on_signal(3, 'buy', 0.5)\
    .on_signal(2, 'buy', 0.25)\
    .on_signal(1, 'buy', 0.1)\
    .on_signal(-1, 'sell', 0.1)\
    .on_signal(-2, 'sell', 0.25)\
    .on_signal(-3, 'sell', 0.5)\
    .run_simulation(feature_matrix, 50000, 500, strat_rsi_macd_1, some_random_forest, transaction_rate=0.03, act_every=n)

# print(simulator.logs['strat_rsi_macd_1'].trail)

simulator.logs['strat_rsi_macd_1'].plot('2016-06', '2016-06-07')
simulator.logs['some_random_forest'].plot('2016-06', '2016-06-07')