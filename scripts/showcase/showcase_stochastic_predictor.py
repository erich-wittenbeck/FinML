
from api import *

history = Chart('C:/Users/User/Desktop/darmstadt/Master-Thesis/Data/Histories/btc_usd_jan2012-jun2018.csv') \
    .fill_missing_data('min', 'interpolate')\
    .upscale('60min') \
    .slice('2016-06', '2018-06')

features = Features(history)\
        .label_data('ternary', 24, 0.0125, 0.0125)

train_data, test_data = features.split('2017-09')

stoch_model = StochasticModel('Fred', 0.51)\
            .determine_distribution(train_data.y)

predictions = stoch_model.predict(test_data)

print(predictions)