
from api import *

from matplotlib import pyplot as plt

history = Chart('/home/files/charts/btc_usd_jan2012-jun2018.csv') \
    .fill_missing_data('min', 'interpolate')\
    .upscale('60min')\
    .slice('2012-01-01')

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

for avg in ['macro', 'micro']:
    for span in spans:
        auc_scores = {identifier: [] for identifier in ['logreg', 'randf', 'baseline']}
        date_times = []

        matrices = features.split(span)

        for matrix in matrices:
            training_data, test_data = matrix.split(0.75)
            training_data.standardize()
            test_data.standardize(training_data)

            logreg = Classification('logreg', 'logreg') \
                .set_hyper_parameters(C=[10 ** exp for exp in range(-6, 7)], max_iter=[20000]) \
                .configure_hpo('exhaustive', 'f1_macro', n_jobs=6, verbose=2) \
                .train(training_data, prune_features=True, rfe_scoring='f1_macro')

            randf = Classification('randf', 'randf') \
                .set_hyper_parameters(n_estimators=100,
                                      bootstrap=True) \
                .train(training_data, prune_features=True, rfe_scoring='f1_macro')

            baseline = Stochastic('baseline')\
                .determine_distribution(training_data)

            evaluator = Evaluator()\
                .evaluate(test_data, logreg, randf, baseline, evaluations=['roc_curves'])

            for model in [logreg, randf, baseline]:
                identifier = model.name
                auc_scores[identifier] += [evaluator.roc_curves[identifier]['auc_'+avg]]

            date_times += [matrix.X.index[-1]]

        plt.figure(figsize=(10, 8))
        plt.title(span + 's')

        for model in [logreg, randf, baseline]:
            identifier = model.name
            color = 'red' if identifier == 'baseline' else 'blue' if identifier == 'logreg' else 'cyan'
            plt.plot(auc_scores[identifier], label=identifier, linestyle='-.', marker='o', color=color)

        plt.ylim(0.15, 0.95)
        plt.axhline(0.5, linestyle=':', color='black')

        date_times = [date_time.strftime('%Y-%m-%d') for date_time in date_times]

        xticks = [pos for pos in range(0, len(date_times)) if pos % n[span] == 0]
        labels = [date_times[xtick] for xtick in xticks]
        plt.xticks(xticks, labels)
        plt.ylabel('Area-under-Curve ('+ avg + '-averaged')

        plt.grid(True, which='both', axis='x', linestyle='--')
        plt.legend(loc='lower right')

        plt.savefig('/home/files/output/png/eval/real-data/auc/real_data_eval_results_auc_' + avg + '_' + span + 's.png', dpi=300, bbox_inches="tight")
        plt.clf()

print('finish!')