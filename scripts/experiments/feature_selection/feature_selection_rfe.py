
from api import *

history = Chart('C:/Users/User/Desktop/darmstadt/Master-Thesis/Data/Histories/btc_usd_jan2012-jun2018.csv') \
    .fill_missing_data('min', 'interpolate')\
    .upscale('60min')\
    .slice('2012', '2014')

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

spans = ['year']#, 'quarter', 'month', 'week']
selected_features = {feature: {span: 0 for span in spans} for feature in features.X.columns}

for span in spans:
    matrices = features.split(span)
    increment = 1/len(matrices)

    for matrix in matrices:
        training_data, test_data = matrix.split(0.75)
        training_data.standardize()

        randf = Classification('randf', 'randf')\
                    .set_hyper_parameters(n_estimators = [100],
                                          bootstrap = [True, False])\
                    .configure_hpo('exhaustive', 'f1_macro', n_jobs=3, verbose=2) \
                    .train(training_data, prune_features=True, k=5)

        predictions = randf.predict(test_data)

        for feature in randf.features:
            selected_features[feature][span] += increment

table = pd.DataFrame(selected_features).T[spans].to_latex()

print(table)