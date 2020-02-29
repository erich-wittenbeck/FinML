
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

labels = ['-1', '0', '1']

for span in spans:
    auc_timelines = {label: {identifier: [] for identifier in ['logreg', 'randf', 'stochastic']} for label in labels}
    date_times = []

    matrices = features.split(span)

    for matrix in matrices:
        training_data, test_data = matrix.split(0.75)
        training_data.standardize()
        test_data.standardize(training_data)

        # logreg = Classification('logreg', 'logreg') \
        #     .set_hyper_parameters(C=[10 ** exp for exp in range(-1, 2)]) \
        #     .configure_hpo('exhaustive', 'f1_macro', n_jobs=3, verbose=2) \
        #     .train(training_data, prune_features=True, rfe_scoring='f1_macro')

        randf = Classification('randf', 'randf') \
            .set_hyper_parameters(n_estimators=100,
                                  bootstrap=True) \
            .train(training_data, prune_features=True, rfe_scoring='f1_macro')

        stochastic = Stochastic('stochastic')\
            .determine_distribution(training_data)

        evaluator = Evaluator()\
            .evaluate(test_data, randf, stochastic, evaluations=['roc_curves'])

        for model in [randf, stochastic]:
            identifier = model.name
            for label in model.classes:
                label_str = str(label)
                auc_timelines[label_str][identifier] += [evaluator.roc_curves[identifier][label_str]['roc_auc']]

        date_times += [matrix.X.index[-1]]

    # auc_timelines["date"] = [date_time.strftime('%Y-%m-%d') for date_time in date_times]
    # df = pd.DataFrame.from_dict(auc_timelines)
    date_times = [date_time.strftime('%Y-%m-%d') for date_time in date_times]
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)

    fig.set_size_inches(10, 8)
    fig.suptitle(span + 's')

    for label_idx, label in enumerate(labels):
        ax = axes[label_idx]
        for model in [randf, stochastic]:
            ax.set_title('AUC for class: ' + label)
            ax.plot(auc_timelines[label][model.name])

    plt.show()
    plt.clf()

print('finish!')