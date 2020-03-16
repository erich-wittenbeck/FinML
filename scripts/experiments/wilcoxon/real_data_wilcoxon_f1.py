
from api import *
from scipy.stats import wilcoxon

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


spans = ['year']#, 'quarter', 'month']
wil_dict = {identifier: {} for identifier in ['logreg', 'randf']}

for span in spans:
    f1_scores = {identifier: [] for identifier in ['logreg', 'randf', 'baseline']}

    matrices = features.split(span)

    for matrix in matrices:
        training_data, test_data = matrix.split(0.75)
        training_data.standardize()
        test_data.standardize(training_data)

        logreg = Classification('logreg', 'logreg') \
            .set_hyper_parameters(C=[10 ** exp for exp in range(-6, 4)]) \
            .configure_hpo('exhaustive', 'f1_macro', n_jobs=6, verbose=2) \
            .train(training_data, prune_features=True, rfe_scoring='f1_macro')

        randf = Classification('randf', 'randf') \
            .set_hyper_parameters(n_estimators=100,
                                  bootstrap=True) \
            .train(training_data, prune_features=True, rfe_scoring='f1_macro')

        baseline = Stochastic('baseline')\
            .determine_distribution(training_data)

        evaluator = Evaluator()\
            .evaluate(test_data, logreg, randf, baseline, evaluations=['reports'])

        for model in [logreg, randf, baseline]:
            identifier = model.name
            f1_scores[identifier] += [evaluator.reports[identifier]['macro avg']['f1-score']]

    f1_baseline = f1_scores['baseline']

    for identifier in ['logreg', 'randf']:
        f1_model = f1_scores[identifier]
        statistic, pvalue = wilcoxon(f1_model, f1_baseline)
        wil_dict[identifier][span] = pvalue

table = pd.DataFrame(wil_dict).to_latex(buf='/home/files/output/latex/feature_selection_rfe_logreg.txt')

print('finish!')