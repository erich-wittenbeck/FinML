from api import *

history = Chart('C:/Users/User/Desktop/darmstadt/Master-Thesis/Data/Histories/btc_usd_jan2012-jun2018.csv') \
    .fill_missing_data('min', 'interpolate')\
    .upscale('60min') \
    .slice('2016-06', '2018-06')

features = Features(history)\
        .label_data('ternary', 24, 0.0125, 0.0125)

print(len(features.split(4000, 8000, 10000)))
print(len(features.split(0.25, 0.5, 0.75)))
print(len(features.split('2016-09', '2017', '2018-02-14')))

print(len(features.slice(4000).y))
print(len(features.slice(8000, 12000).y))
print(len(features.slice(0.5, 0.75).y))
print(len(features.slice('2016-11', '2017-01-03').y))