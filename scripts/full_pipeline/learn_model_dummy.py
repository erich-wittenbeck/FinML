import numpy as np
import matplotlib.pyplot as plt

from api.utility.in_out import read_from_csvfile, load_config
from sklearn.metrics import accuracy_score, precision_score, recall_score
from api.utility.statistics import exponential_smoothing as exs

# Load history

history_data_dir, training_data_dir = load_config('history_data_dir', 'training_data_dir', section='PATHS')
close_prices, volumes = zip(*[(float(row[5]), float(row[6])) for row in read_from_csvfile(history_data_dir + 'coinbase_btc_usd_dec2014-jun2018_hours.csv')])

# smooth out closing prices

close_prices_smoothed = exs(close_prices, 0.5)

n = 1

deltas_in_percent = [(close_prices_smoothed[t + n] - close_prices_smoothed[t]) * (100 / close_prices_smoothed[t - n])
                     if t+n < len(close_prices_smoothed)
                     else 0
                     for t in range(len(close_prices_smoothed))]

# Create numpy-array

td_array = np.array(deltas_in_percent)

# Create discrete labels

margin = 0.2

discretize = lambda x : 1 if x >= margin else -1 if x <= -margin else 0
vdiscretize = np.vectorize(discretize)

test_y = vdiscretize(td_array)

# Create dummy predictions

predictions = np.array([test_y[t - n] for t in range(0, len(test_y))])

accuracy = accuracy_score(test_y, predictions)
precision = precision_score(test_y, predictions, average='micro')
recall = recall_score(test_y, predictions, average='micro')
# F1 = 2 * (precision * recall) / (precision + recall)

print('Accuracy: ' + str(accuracy))
print('Precision: ' + str(precision))
print('Recall: ' + str(recall))
# print('F1: ' + str(F1))

ticks = list(range(len(close_prices)))

plt.plot(ticks, test_y)
plt.plot(ticks, predictions)

plt.legend(['gold', 'dummy'], loc='upper right')

plt.show()