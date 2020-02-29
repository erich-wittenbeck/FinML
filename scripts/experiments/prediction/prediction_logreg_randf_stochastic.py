
from api import *

import numpy as np
from matplotlib import pyplot as plt

history = Chart('C:/Users/User/Desktop/darmstadt/Master-Thesis/Data/Histories/btc_usd_jan2012-jun2018.csv') \
    .fill_missing_data('min', 'interpolate')\
    .upscale('60min')\
    .slice('2012-01-01', '2013-12-31')

features = Features(history) \
        .label_data('ternary', 1, 0, 0)\
        .add_indicators(Indicator('macd', 'macd'),
                        Indicator('trix', 'trix'),
                        Indicator('adx', 'adx'),
                        Indicator('vrtx', 'vrtx'),
                        Indicator('rsi', 'rsi'),
                        Indicator('mfi', 'mfi'),
                        Indicator('wpr', 'wpr'),
                        Indicator('ao', 'ao'),
                        Indicator('uo', 'uo'),
                        Indicator('stock_k', 'stoch', return_as='k'),
                        Indicator('stoch_d', 'stoch', return_as='d'),
                        Indicator('midx', 'midx'),
                        Indicator('bbands_total', 'bbands', return_as='total_delta'),
                        Indicator('bbands_upper', 'bbands', return_as='upper_delta'),
                        Indicator('bbands_lower', 'bbands', return_as='lower_delta'),
                        Indicator('kltch_total', 'kltch', return_as='total_delta'),
                        Indicator('kltch_upper', 'kltch', return_as='upper_delta'),
                        Indicator('kltch_lower', 'kltch', return_as='lower_delta'),
                        Indicator('obv', 'obv'),
                        Indicator('fidx', 'fidx'),
                        Indicator('cmf', 'cmf'),
                        Indicator('adl', 'adl'))



spans = ['year', 'quarter', 'month']
n = {'year': 1, 'quarter': 4, 'month': 12}

for span in spans:
    auc_timelines = {identifier: [] for identifier in ['logreg', 'randf', 'stochastic']}
    date_times = []

    matrices = features.split(span)

    for matrix in matrices:
        training_data, test_data = matrix.split(0.75)
        training_data.standardize()
        test_data.standardize(training_data)

        logreg = Classification('logreg', 'logreg') \
            .set_hyper_parameters(C=[10 ** exp for exp in range(-1, 2)]) \
            .configure_hpo('exhaustive', 'f1_macro', n_jobs=3, verbose=2) \
            .train(training_data, prune_features=True, rfe_scoring='f1_macro')

        stochastic = Stochastic('stochastic')\
            .determine_distribution(training_data)

        evaluator = Evaluator()\
            .evaluate(test_data, logreg, stochastic, evaluations=['roc_curves'])

        for model in [logreg, stochastic]:
            identifier = model.name
            auc_timelines[identifier] += [evaluator.roc_curves[identifier]['roc_auc']]

        date_times += [matrix.X.index[-1]]

    auc_timelines["date"] = [date_time.strftime('%Y-%m-%d') for date_time in date_times]
    df = pd.DataFrame.from_dict(auc_timelines)
    df = df.set_index("date")
    df = df.T

print('finish!')