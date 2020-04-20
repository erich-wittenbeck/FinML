
from finml import *
from scipy.stats import wilcoxon

history = Chart('/home/files/charts/btc_usd_jan2012-jun2018.csv') \
    .fill_missing_data('min', 'interpolate')\
    .upscale('60min')\
    .slice('2012-01-01')\
    .randomize()

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
wil_dict_f1, wil_dict_auc = ({identifier: {} for identifier in ['logreg', 'randf']},)*2

for span in spans:
    f1_scores = {identifier: [] for identifier in ['logreg', 'randf', 'baseline']}
    # auc_scores = {identifier: [] for identifier in ['logreg', 'randf', 'baseline']}

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

        evaluator_f1 = Evaluator()\
            .evaluate(test_data, logreg, randf, baseline, evaluations=['reports'])
        #
        # try:
        #     evaluator_auc = Evaluator() \
        #         .evaluate(test_data, logreg, randf, baseline, evaluations=['roc_curves'])
        # except (ValueError, IndexError) as e:
        #     evaluator_auc = None

        for model in [logreg, randf, baseline]:
            identifier = model.name
            f1_scores[identifier] += [evaluator_f1.reports[identifier]['macro avg']['f1-score']]
            # if evaluator_auc is not None:
            #     auc_scores[identifier] += [evaluator_auc.roc_curves[identifier]['auc_macro']]
            # else:
            #     auc_scores[identifier] += [auc_scores[identifier][-1]]

    f1_baseline = f1_scores['baseline']
    # auc_baseline = auc_scores['baseline']

    for identifier in ['logreg', 'randf']:
        f1_model = f1_scores[identifier]
        # auc_model = auc_scores[identifier]

        statistic, pvalue = wilcoxon(f1_model, f1_baseline)
        wil_dict_f1[identifier][span] = pvalue

        # statistic, pvalue = wilcoxon(auc_model, auc_baseline)
        wil_dict_auc[identifier][span] = pvalue

table_f1 = pd.DataFrame(wil_dict_f1).to_latex(buf='/home/files/output/latex/random_data_wilcoxon_f1.txt')
# table_auc = pd.DataFrame(wil_dict_auc).to_latex(buf='/home/files/output/latex/random_data_wilcoxon_auc.txt')

print('finish!')