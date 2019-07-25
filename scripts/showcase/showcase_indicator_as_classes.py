
from api import *
from time import time

alphas = [1, 0.875, 0.75, 0.625]

history = Chart('C:/Users/User/Desktop/darmstadt/Master-Thesis/Data/Histories/btc_usd_jan2012-jun2018.csv') \
    .fill_missing_data('min', 'interpolate')\
    .upscale('60min') \
    .slice('2016-06', '2018-06')

evaluator = Evaluator()

start_time = time()

for alpha in alphas:

    rsi = Indicator('rsi', 'rsi')\
        .smooth_chart(alpha, 'open', 'close')

    macd = Indicator('macd', 'macd_main_line', 'macd_signal_line')\
        .smooth_chart(alpha, 'open', 'close')

    uo = Indicator('uo', 'uo')\
        .smooth_chart(alpha, 'open', 'close')

    stoch = Indicator('stoch')\
        .smooth_chart(alpha, 'open', 'close')\
        .transform('sub', 'stoch')

    features = Features(history)\
        .label_data('ternary', 24, 0.0125, 0.0125)\
        .add_features(rsi, macd, uo, stoch)\
        .normalize_features('min_max', ('macd_main_line', 'macd_signal_line'), lookback=14, only_positive=False)\
        .normalize_features(lambda x : x/100, ('rsi', 'uo', 'stoch'))\
        .prune_features(2)\
        .rolling_window(0, 3)

    print(list(features.X))

    train_data, test_data = features.split('2017-09')

    randf = Classification('randf_alpha_' + str(alpha), 'randf') \
        .set_hyper_parameters(n_estimators=[100],
                              max_features=[1, 2, 3, 4],
                              bootstrap=[True, False]) \
        .configure_hpo('f1_macro', n_jobs=3, verbose=0) \
        .train(train_data)

    evaluator.evaluate(test_data, randf)

end_time = time()

evaluator.plot()

print('time elapsed: ' + str(end_time - start_time) + ' seconds')