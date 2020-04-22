
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

randf = Classification('randf', 'randf')\
                    .set_hyper_parameters(n_estimators = 100,
                                          bootstrap = True)\
                    .train(training_data, prune_features=True, rfe_scoring='f1_macro')

print(randf.features)

