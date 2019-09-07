
from api import *
from time import time

# Setup meta-parameters

alphas = [0.875, 0.5, 0.125]

# Setup Base History and evaluator

history = Chart('C:/Users/User/Desktop/darmstadt/Master-Thesis/Data/Histories/btc_usd_jan2012-jun2018.csv')\
        .fill_missing_data('min', 'interpolate')\
        .upscale('60min') \
        .slice('2016-06', '2018-06')

evaluator = Evaluator()

rsi = Indicator('rsi', 'rsi')
mfi = Indicator('mfi', 'mfi')
uo  = Indicator('uo', 'uo')
stoch = Indicator('stoch', 'stoch', return_as='delta')

start_time = time()

model_count = 1

for alpha in alphas:

    feature_matrix = Features(history) \
        .smooth_chart(alpha, 'open', 'close', 'high', 'low')\
        .label_data('ternary', 1, 0.0025, 0.0025) \
        .add_indicators(rsi, mfi, uo, stoch)\
        .normalize_features(lambda x: x / 100, ('rsi', 'mfi', 'uo', 'stoch'))

    train_data, test_data = feature_matrix.split('2017-09')

    randf = Classification('rndf_' + str(model_count), 'randf') \
        .set_hyper_parameters(n_estimators = [100],
                              max_features = [1,2,3,4],
                              bootstrap = [True, False]) \
        .configure_hpo('f1_macro', n_jobs=2, verbose=2) \
        .train(train_data)

    model_count += 1

    evaluator.evaluate(test_data, randf)

end_time = time()

print('time elapsed: ' + str(end_time - start_time) + ' seconds')

evaluator.plot()