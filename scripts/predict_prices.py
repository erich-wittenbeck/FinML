

from finml import *

history = Chart('/path/to/your/chart/data.csv') \
    .fill_missing_data('min', 'interpolate')\
    .upscale('60min')

features = Features(history) \
        .label_data('ternary', 1, 0, 0)\
        .add_indicators(Indicator('macd', 'macd'),
                        Indicator('rsi', 'rsi'),
                        Indicator('wpr', 'wpr'),
                        Indicator('ao', 'ao'),
                        Indicator('uo', 'uo')
                        # Add or remove indicators
                        )

training_data, test_data = features.split(0.75)

training_data.standardize()
test_data.standardize(training_data)

logreg = Classification('logreg', 'logreg') \
            .set_hyper_parameters(C=[10 ** exp for exp in range(-4, 4)]) \
            .configure_hpo('exhaustive', 'f1_macro', n_jobs=1, verbose=2) \
            .train(training_data, prune_features=True, rfe_scoring='f1_macro')

randf = Classification('randf', 'randf')\
                    .set_hyper_parameters(n_estimators = 100,
                                          bootstrap = True)\
                    .train(training_data, prune_features=True, rfe_scoring='f1_macro')

evaluator = Evaluator()\
            .evaluate(test_data, logreg, randf, evaluations=['reports'])

evaluator.plot(save_as='/path/to/destination/file.png')