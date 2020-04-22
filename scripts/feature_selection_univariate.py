
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
                        )\
        .prune_features(3)

print(features.X.columns)