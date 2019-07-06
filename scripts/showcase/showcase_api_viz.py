
import api

from time import time

import lib.models

start_time = time()

history = api.Chart('C:/Users/User/Desktop/darmstadt/Master-Thesis/Data/Histories/btc_usd_jan2012-jun2018.csv', True, 'min', 'interpolate') \
    .upscale('60min') \
    .slice('2017-06', '2018-06')\
    .smooth(0.5, 'open', 'close')

feature_matrix = api.Features(history)\
    .label_data('ternary', 'close', 24, 0.05, 0.05)\
    .add_indicator('rsi', 'rsi')\
    .add_indicator(['macd'], 'macd', return_as='delta')\
    .add_indicator(['bbands_upper_delta', 'bbands_lower_delta'], 'bbands', return_as='deltas')\
    .normalize_features(['macd', 'bbands_upper_delta', 'bbands_lower_delta'], 'min_max', 14, only_positive=False)\
    .normalize_features('rsi', lambda x : x/100)

train_data, test_data = feature_matrix.split('2018')

svm_johnny = lib.models.Classifier('johnny', 'svm')\
    .set_hyper_parameters(kernel='poly', gamma=2 ** -8, C=64)\
    .train(train_data)

svm_billy = lib.models.Classifier('billy', 'svm')\
    .set_hyper_parameters(kernel='rbf', gamma=2 ** -5, C=32)\
    .train(train_data)

randf_bob = lib.models.Classifier('bob', 'randf')\
    .train(train_data)

evaluator = api.Evaluator(test_data, svm_johnny, svm_billy, randf_bob) \
    .evaluate('reports', 'confmats', 'roc_curves')

report_johnny, report_billy, report_bob = (evaluator.reports['johnny'], evaluator.reports['billy'], evaluator.reports['bob'])

print('Results for johnny:')
print()
print(report_johnny)
print()
print('Results for billy:')
print()
print(report_billy)
print()
print('Results for bob')
print()
print(report_bob)

end_time = time()

evaluator.plot()

print('time elapsed: ' + str(end_time - start_time) + ' seconds')