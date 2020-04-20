
from finml import *
from time import time


# Setup meta-parameters

alphas = [1, 0.875, 0.75, 0.625]
# alphas = [0.5, 0.375, 0.25, 0.125]

# Setup Base History and evaluator

history = Chart('C:/Users/User/Desktop/darmstadt/Master-Thesis/Data/Histories/btc_usd_jan2012-jun2018.csv',
                True, 'min', 'interpolate')\
        .upscale('60min') \
        .slice('2016-06', '2018-06')
evaluator = Evaluator()

start_time = time()

for alpha in alphas:

    feature_matrix = Features(history) \
        .smooth_chart_data(alpha, 'open', 'close', 'high', 'low')\
        .label_data('ternary', 'close', 4, 0.25, 0.25) \
        .add_indicator('rsi', 'rsi') \
        .add_indicator('mfi', 'mfi') \
        .add_indicator('uo', 'uo') \
        .add_indicator('stoch', 'stoch', return_as='delta') \
        .normalize_features(['rsi', 'mfi', 'uo', 'stoch'], lambda x: x / 100)

    train_data, test_data = feature_matrix.split('2017-09')

    svm = Classification('svm_alpha_' + str(alpha), 'svm') \
        .set_hyper_parameters(kernel=['poly', 'rbf', 'sigmoid'],
                              gamma=[2 ** -13, 2 ** -8, 2 ** -5, 2 ** -3],
                              C=[2 ** 1, 2 ** 3, 2 ** 5, 2 ** 8, 2 ** 9]) \
        .configure_hpo('f1_macro', n_jobs=2, verbose=2) \
        .train(train_data)

    evaluator.evaluate(test_data, svm)

end_time = time()

evaluator.plot()

print('time elapsed: ' + str(end_time - start_time) + ' seconds')