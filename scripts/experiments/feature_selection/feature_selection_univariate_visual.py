
from finml import *
from lib.indicators.dummy import random_standard_variable as baseline

import numpy as np
from matplotlib import pyplot as plt

# history = Chart('/home/files/charts/btc_usd_jan2012-jun2018.csv') \
history = Chart('C:/Users/User/Desktop/darmstadt/Master-Thesis/Data/Histories/btc_usd_jan2012-jun2018.csv') \
    .fill_missing_data('min', 'interpolate')\
    .upscale('60min')

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
                        Indicator('adl', 'adl'),
                        Indicator('baseline', baseline))

spans = ['year', 'quarter', 'month', 'week']
n = {'year': 1, 'quarter': 4, 'month': 12, 'week': 52}

for span in spans:
    selection_timelines = {feature: [] for feature in features.X.columns}
    date_times = []

    matrices = features.split(span)
    num_of_matrices = len(matrices)

    for matrix in matrices:
        matrix.standardize()\
              .prune_features('baseline')

        selected_features = matrix.X.columns

        for feature in selection_timelines:
            selection_timelines[feature] += [1 if feature in selected_features else 0]

        date_times += [matrix.X.index[-1]]

    selection_timelines = {feature: selection_timelines[feature] for feature in selection_timelines if not all(v == 0 for v in selection_timelines[feature])}
    selection_timelines["date"] = [date_time.strftime('%Y-%m-%d') for date_time in date_times]
    df = pd.DataFrame.from_dict(selection_timelines)
    df = df.set_index("date")
    df = df.T

    plt.figure(figsize=(10, 8))

    plt.pcolormesh(df, edgecolor='black', linewidths=0.1, cmap='bwr', alpha=0.75)
    plt.yticks(np.arange(0.5, len(df.index), 1), df.index)

    xticks = [pos for pos in range(1, len(df.columns)+1) if pos %n[span] == 0]
    labels = [df.columns[xtick-1] for xtick in xticks]
    plt.xticks(xticks, labels)

    plt.title(span + 's')
    plt.savefig('E:/image.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.clf()



print("finish!")
