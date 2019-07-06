
from api import *

history = Chart('C:/Users/User/Desktop/darmstadt/Master-Thesis/Data/Histories/btc_usd_jan2012-jun2018.csv') \
    .fill_missing_data('min', 'interpolate')\
    .upscale('60min') \
    .slice('2016-06', '2018-06')

history.plot('2016-06-02', '2016-06-06')