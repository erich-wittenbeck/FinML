
import matplotlib.pyplot as plt
import numpy as np

from lib.utility.data import create_diagonal_mask

def plot_candlesticks(ax, chart_df):

    bottom_wax = chart_df[['open', 'close']].min(axis=1)
    height_wax = chart_df[['open', 'close']].max(axis=1) - bottom_wax

    highs = chart_df['high']
    lows = chart_df['low']

    highest_value = highs.max()
    lowest_value = lows.min()

    bottom_wick = lows
    height_wick = highs - lows

    diffs = chart_df['close'] - chart_df['open']
    colors = diffs.map(lambda x: 'g' if x >= 0 else 'r')

    ylim_min = lowest_value - 5
    ylim_max = highest_value + 5

    x_coords = list(range(len(chart_df.index)))

    ax.bar(x=x_coords, height=height_wax, bottom=bottom_wax, color=colors)
    ax.bar(x=x_coords, height=height_wick, bottom=bottom_wick, width=0.15, color=colors)

    xticklabels = [date_time.strftime('%Y-%m-%d') for date_time in chart_df.index]
    ax.set_xticklabels(xticklabels, rotation=45)

    ax.set_ylim(ylim_min, ylim_max)
    ax.set_ylabel('Prices')

def plot_simulation(ax_history, ax_assets, ax_budget, history_df, trail_df, freq):

    plot_candlesticks(ax_history, history_df)

    shared_x = list(range(len(history_df.index)))
    xticks = np.arange(min(shared_x), max(shared_x)+freq, step=freq)

    ax_assets.plot(shared_x, trail_df['assets'])
    ax_assets.set_xticks(xticks)
    ax_assets.set_ylabel('Assets')

    ax_budget.plot(shared_x, trail_df['budget'])
    ax_budget.set_xticks(xticks)
    ax_budget.get_xaxis().set_ticklabels(['' for tick in xticks])
    ax_budget.set_ylabel('Budget')


def plot_classification_report(ax, classification_report):
    keys = classification_report.keys()
    items = classification_report.items()

    scores_prec = [scores['precision'] for cls, scores in items]
    scores_reca = [scores['recall'] for cls, scores in items]
    scores_fone = [scores['f1-score'] for cls, scores in items]

    w = 0.2
    x_reca = list(range(len(keys)))
    x_prec = [x-w for x in x_reca]
    x_fone = [x+w for x in x_reca]

    ax.bar(x=x_prec, height=scores_prec, width=w, color='r', label='Precision')
    ax.bar(x=x_reca, height=scores_reca, width=w, color='b', label='Recall')
    ax.bar(x=x_fone, height=scores_fone, width=w, color='g', label='F1-Score')

    ax.set_xticks(x_reca)
    ax.set_xticklabels(keys)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="upper right")

    for ticklabel in ax.get_xticklabels():
        ticklabel.set_rotation(25)

def plot_confustion_matrix(ax, confusion_matrix, classes):

    mask_correct = create_diagonal_mask(*confusion_matrix.shape)
    mask_incorrect = create_diagonal_mask(*confusion_matrix.shape, invert=True)

    confmat_cor = np.ma.masked_array(confusion_matrix, mask_correct)
    confmat_inc = np.ma.masked_array(confusion_matrix, mask_incorrect)

    # ax.imshow(confusion_matrix, cmap=plt.cm.Blues)

    ax.imshow(confmat_cor, cmap=plt.cm.Blues)
    ax.imshow(confmat_inc, cmap=plt.cm.Reds)

    ax.set_xlabel('predicted class')
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_ylabel('true class')
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes)

    for i, j in [(i, j) for i in list(range(confusion_matrix.shape[0]))
                   for j in list(range(confusion_matrix.shape[1]))]:
        ax.text(j, i, confusion_matrix[i, j],
                       ha="center", va="center", color="grey")

    return ax

def plot_roc_curves(ax, roc_dict):

    for cls, roc_curve in roc_dict.items():
        ax.plot(roc_curve['fpr'], roc_curve['tpr'], label='Class: '+str(cls)+'; AUC %0.2f' % roc_curve['roc_auc'])

    ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower right")

    return ax