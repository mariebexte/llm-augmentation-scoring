import pandas as pd
from sklearn.metrics import f1_score


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

def get_macro_f1(y_true, y_pred):

    return  f1_score(y_true=y_true, y_pred=y_pred, average='macro')