from api import *

history = Chart('C:/Users/User/Desktop/darmstadt/Master-Thesis/Data/Histories/btc_usd_jan2012-jun2018.csv') \
    .fill_missing_data('min', 'interpolate')\
    .upscale('60min')

features = Features(history) \
        .label_data('ternary', 1, 0, 0)\
        .add_features(Indicator('macd', 'macd'),
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

spans = ['year', 'quarter', 'month', 'week']
selected_features = {feature: {span: 0 for span in spans} for feature in features.X.columns}

for span in spans:
    matrices = features.split(span)
    increment = 1/len(matrices)

    for matrix in matrices:
        matrix.prune_features(5)

        for feature in matrix.X.columns:
            selected_features[feature][span] += increment

print(selected_features)