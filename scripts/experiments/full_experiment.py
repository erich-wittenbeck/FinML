
from api import *

histories = Chart('C:/Users/User/Desktop/darmstadt/Master-Thesis/Data/Histories/btc_usd_jan2012-jun2018.csv') \
    .fill_missing_data('min', 'interpolate')\
    .upscale('60min')\
    .split(*[0.03125*i for i in range(1,32)])

# TODO Indicators!!!

macd = Indicator('macd', 'macd_main_line', 'macd_signal_line')
trix = Indicator('trix', 'trix_main_line', 'trix_signal_line')
adx = Indicator('adx', 'adx')
#vrtx = Indicator('vrtx', 'vrtx_plus', 'vrtx_minus')
rsi = Indicator('rsi', 'rsi')
mfi = Indicator('mfi', 'mfi')
wpr = Indicator('wpr', 'wpr')
ao = Indicator('ao', 'ao')
uo = Indicator('uo', 'uo')
stoch = Indicator('stoch', 'stoch_k', 'stoch_d')
#midx = Indicator('midx', 'midx')
bbands = Indicator('bbands', 'bb_u', 'bb_m', 'bb_l')
kltch = Indicator('kltch', 'klt_u', 'klt_m', 'klt_l')

selected_features = {}

metaparams = Metaparameters(histories=histories,
                            margins=[0.05, 0.025, 0.0125, 0.00625]
                            )

for history, margin in metaparams.cartesian:

    features = Features(history) \
        .label_data('ternary', 1, margin, margin)\
        .add_features(macd, trix, adx, rsi, mfi, wpr, ao, uo, stoch, bbands, kltch)\
        .prune_features(5)

    for feature in features.X.columns:
        if feature in selected_features:
            selected_features[feature] += 1
        else:
            selected_features[feature] = 1

    # train_data, test_data = features.split(0.75)
    #
    # randf = Classification('randf', 'randf')\
    #     .set_hyper_parameters(n_estimators=[100],
    #                           max_features=[1, 2, 3, 4],
    #                           bootstrap=[True, False])\
    #     .configure_hpo('exhaustive', 'f1_macro', n_jobs=3, verbose=2)\
    #     .train(train_data)
    #
    # print(len(history.data), margin, margin)

    # TODO the rest

print(selected_features)