import os
import sys

import pandas as pd
from sklearn.metrics import classification_report
from metrics import calculate_accuracy, calculate_gwets_ac2, calculate_qwk, calculate_within_1, get_confusion_matrix



def get_three_way(label):

    if label == '1.0':

        return label
    
    elif label == 'contradictory':

        return label

    else:

        return 'incorrect'


def get_two_way(label):

    if label == '1.0':

        return label
    
    else:

        return 'incorrect'


def get_confusion_matrix(first, second, first_name='Annotator 1', second_name='Annotator 2'):

    first_series = pd.Series(first, name=first_name)
    second_series = pd.Series(second, name=second_name)
    
    return pd.crosstab(first_series, second_series)


def write_stats(target_dir, y_true, y_pred, labels=None):

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    # Stats file and confusion Matrix
    with open(os.path.join(target_dir, 'stats.txt'), 'w') as stats_file:

        stats_file.write(classification_report(y_true=y_true, y_pred=y_pred, zero_division=0, labels=labels))
        stats_file.write('\n\n')
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            stats_file.write(str(get_confusion_matrix(first=y_true, second=y_pred, first_name='true', second_name='predicted')))
        stats_file.write('\n\n')
        stats_file.write('Acc:\t' + str(calculate_accuracy(y_true=y_true, y_pred=y_pred)) + '\n')
        # stats_file.write("Within-1:\t" + str(calculate_within_1(y_true=y_true, y_pred=y_pred)) + '\n')
        stats_file.write('QWK:\t' + str(calculate_qwk(y_true=y_true, y_pred=y_pred)) + '\n')
        stats_file.write("Gwet's AC2:\t" + str(calculate_gwets_ac2(y_true=y_true, y_pred=y_pred)) + '\n')
