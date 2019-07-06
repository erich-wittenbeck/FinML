import numpy as np
import pandas as pd
# import seaborn as sn
import matplotlib.pyplot as plt

from lib.utility.in_out import read_from_csvfile, load_config
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from lib.utility.statistics import exponential_smoothing as exs
from lib.utility.statistics import min_max_norm
from lib.utility.ml import roc_auc_multiclass_scorer
from lib.utility.data import split_timeseries_data, shift_list_back
from lib.utility.mysql import MySQLAccessor
from lib.indicators.momentum import moving_average_convergence_divergence
from lib.indicators.oscillation import relative_strength_index
from lib.indicators.volatility import bollinger_bands
from lib.indicators.volume import on_balance_volume

# Set metaparameters

size_y = 1

metric = 'roc_auc_weighted'
currency = 'btc_usd'
candle_size = 'hours'
n = 24
margin = 0.25
alpha = 0.125
features = ['bbands', 'macd_5', 'rsi', 'obv']

# Load history

history_data_dir, training_data_dir, eval_dir = load_config('history_data_dir', 'training_data_dir', 'eval_dir', section='PATHS')
close_prices, volumes = zip(*[(float(row[5]), float(row[6])) for row in read_from_csvfile(history_data_dir + 'coinbase_btc_usd_dec2014-jun2018_'+ candle_size +'.csv',
                                                                                          end=24000)])

# smooth out closing prices

close_prices_smoothed = exs(close_prices, alpha)

# indicators/features

uband, mband, lband = bollinger_bands(close_prices_smoothed)
udelta = [u - m for u, m in zip(uband, mband)]
ldelta = [m - l for m, l in zip(mband, lband)]

MACD_main_line, MACD_signal_line = moving_average_convergence_divergence(close_prices_smoothed)
MACD_delta = [a - b for a, b in zip(MACD_main_line, MACD_signal_line)]

RSI = relative_strength_index(close_prices_smoothed)

OBV = on_balance_volume(close_prices_smoothed, volumes)

# normalize/rescale

udelta = min_max_norm(udelta, 14)
ldelta = min_max_norm(ldelta, 14)

MACD_delta = min_max_norm(MACD_delta, 9, only_positive=False)

RSI_mini = [v/100 for v in RSI]

OBV = min_max_norm(OBV, lookback=24, only_positive=False)

# Labels

deltas_in_percent = [(close_prices_smoothed[t + n] - close_prices_smoothed[t]) * (100 / close_prices_smoothed[t - n])
                     if t+n < len(close_prices_smoothed)
                     else 0
                     for t in range(len(close_prices_smoothed))]

# assemble data points

feature_data = [list(data_point) for data_point in zip(*[shift_list_back(MACD_delta, n) for n in range(5)],
                                                       udelta, ldelta, RSI_mini, OBV, deltas_in_percent)]

# Split feature data into train set and test set

train_data, test_data = split_timeseries_data(feature_data, 40, 20, 40)

# cutoff = int(len(feature_data)*0.7)
# train_data, test_data = (feature_data[0:cutoff], feature_data[cutoff+n:])

print(len(train_data))
print(len(test_data))

# Create numpy-arrays

train_array = np.array(train_data)
test_array = np.array(test_data)

# Seperate input features (x) from output features (y)

train_array_x = train_array[:, :-size_y]
train_array_y = np.reshape(train_array[:, -size_y:], -1)

test_array_x = test_array[:, :-size_y]
test_array_y = np.reshape(test_array[:, -size_y:], -1)

# Discretize y

discretize = lambda x : 1 if x >= margin else -1 if x <= -margin else 0
vdiscretize = np.vectorize(discretize)

train_array_y = vdiscretize(train_array_y)
test_array_y = vdiscretize(test_array_y)

# Setup grid-search based HPO

search_grid = {'kernel':['poly', 'rbf', 'sigmoid'],
                'gamma':[2 ** -13, 2 ** -8, 2 ** -5, 2 ** -3],
               #'tol':[1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
                'shrinking':[True],
                'C': [2 ** 1, 2 ** 3, 2 ** 5, 2 ** 8, 2 ** 9]}

# Create the scorer

scorer = roc_auc_multiclass_scorer(classes=[-1, 0, 1], average=metric.split('_')[-1]) if metric.startswith('roc_auc_') \
    else metric

# Train optimized classifiers

svclf = SVC(kernel='poly')

svclf = GridSearchCV(SVC(), search_grid, cv=5,
                     scoring=scorer, verbose=3)

svclf.fit(train_array_x, train_array_y)

classes = svclf.classes_

# Predict some stuff

predictions = svclf.predict(test_array_x)

# Determined, optimal hyperparameters

hyperparams = svclf.best_params_

# Evaluation scores

precision = precision_score(y_true=test_array_y, y_pred=predictions, average=None)
recall = recall_score(y_true=test_array_y, y_pred=predictions, average=None)
f1 = f1_score(y_true=test_array_y, y_pred=predictions, average=None)

print(classification_report(y_true=test_array_y, y_pred=predictions))

#### Write results to database

maccess = MySQLAccessor(host='localhost', usr='root', pwd='root', db='crypto_ml_eval')

# Feature configuration

features_uuid = maccess.insert_features(sorted(features))

# The evaluation report itself

metaparams = {'metric' : metric,
              'currency' : currency,
              'candle_size' : candle_size,
              'n' : n,
              'margin' : margin,
              'alpha' : alpha}

existsAlready, svm_evaluation_uuid = maccess.insert_evaluation(model='svm',
                                                               metaparams=metaparams,
                                                               features_uuid=features_uuid)

if existsAlready:

    # Retrieve foreign keys

    hyperparams_uuid, f1_scores_uuid, precision_scores_uuid, recall_scores_uuid = maccess.get_foreign_keys_from_evaluation('svm', svm_evaluation_uuid)

    # Update hyperparameters and scores

    maccess.update_hyperparams('svm', hyperparams_uuid, hyperparams)

    maccess.update_scores(metric='f1', uuid=f1_scores_uuid, neg=f1[0], ntr=f1[1], pos=f1[2])
    maccess.update_scores(metric='precision', uuid=precision_scores_uuid, neg=precision[0], ntr=precision[1], pos=precision[2])
    maccess.update_scores(metric='recall', uuid=recall_scores_uuid, neg=recall[0], ntr=recall[1], pos=recall[2])

else:

    # Create new rows for evaluation scores

    f1_scores_uuid = maccess.insert_scores(metric='f1', neg=f1[0], ntr=f1[1], pos=f1[2])
    precision_scores_uuid = maccess.insert_scores(metric='precision', neg=precision[0], ntr=precision[1], pos=precision[2])
    recall_scores_uuid = maccess.insert_scores(metric='recall', neg=recall[0], ntr=recall[1], pos=recall[2])

    # Create new row for hyperparameters

    hyperparams_uuid = maccess.insert_hyperparams('svm', hyperparams)

    # Add them to the evaluation instance

    maccess.update_evaluation('svm',
                              svm_evaluation_uuid,
                              hyperparams_uuid,
                              f1_scores_uuid,
                              precision_scores_uuid,
                              recall_scores_uuid)


# Commit transactions

maccess.commit()


# Confusion matrix

cm = confusion_matrix(y_true=test_array_y, y_pred=predictions)
cm_df = pd.DataFrame(cm, range(cm.shape[0]), range(cm.shape[1]))

# Plot confusion matrix

# cm_plot = sn.heatmap(cm_df, cmap=plt.cm.Blues, xticklabels=classes, yticklabels=classes, annot=True)
# cm_plot.set_xlabel('predicted class')
# cm_plot.set_ylabel('true class')
# plt.savefig(eval_dir +'/Plots/ConfMat/svm_' + metric + '_n' + str(n) + '_m' + str(int(margin * 100)) + '_confmat.png')

# ROC curves

print('Plotting ROC curves!')

plt.gcf().clear()

for cls, cls_idx in zip(classes, range(len(classes))):
    scores = svclf.decision_function(test_array_x)[:, cls_idx]

    fpr, tpr, threshold = roc_curve(test_array_y, scores, pos_label=cls)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label='Class: '+str(cls)+'; AUC %0.2f' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc="lower right")
plt.savefig(eval_dir +'/Plots/ROC/svm_' + metric + '_n' + str(n) + '_m' + str(int(margin * 100)) + '_roc_curves.png')
