from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score
import numpy as np
import pandas as pd
from irrCAC.raw import CAC

## Metrics

# QWK
def calculate_qwk(y_true, y_pred):

    return cohen_kappa_score(y1=y_true, y2=y_pred, weights='quadratic')

# Acc
def calculate_accuracy(y_true, y_pred):

    return accuracy_score(y_true=y_true, y_pred=y_pred)

# F1
def calculate_macro_f1(y_true, y_pred):

    return f1_score(y_true=y_true, y_pred=y_pred, average='macro')

def calculate_gwets_ac1(y_true, y_pred):

    # Imitate the kind of object that the example use shows
    df_votes = pd.DataFrame.from_dict({'Rater1': y_true, 'Rater2': y_pred})
    df_votes.index.name = 'Units'
    df_votes.index = range(1, len(df_votes)+1)

    # Transform into the object we need to derive agreement from
    cac_votes = CAC(df_votes)
    result = cac_votes.gwet()

    print(result)

    return result['est']['coefficient_value']

def calculate_gwets_ac2(y_true, y_pred):

    # Imitate the kind of object that the example use shows
    df_votes = pd.DataFrame.from_dict({'Rater1': y_true, 'Rater2': y_pred})
    df_votes.index.name = 'Units'
    df_votes.index = range(1, len(df_votes)+1)

    try:
        # Transform into the object we need to derive agreement from
        cac_votes = CAC(df_votes, weights='quadratic')
        result = cac_votes.gwet()
        return result['est']['coefficient_value']
    
    except ZeroDivisionError:
        return -100

# Within-1
def calculate_within_1(y_true, y_pred):

    diff = np.array(y_true) - np.array(y_pred)
    diff = np.absolute(diff)
    correct = diff <= 1

    return sum(correct)/len(correct)

def get_confusion_matrix(first, second, first_name='Annotator 1', second_name='Annotator 2'):

    first_series = pd.Series(first, name=first_name)
    second_series = pd.Series(second, name=second_name)
    
    return pd.crosstab(first_series, second_series)


# y_true = [3,4,6,12,5]
# y_pred = [4,3,5,7,8]
# y_true = [1,1,0,1,0]
# y_pred = [0,1,0,0,0]
# print(calculate_within_1(y_true=y_true, y_pred=y_pred))
# print(calculate_gwets_ac1(y_true=y_true, y_pred=y_pred))
# print(calculate_gwets_ac2(y_true=y_true, y_pred=y_pred))
# print(calculate_macro_f1(y_true=y_true, y_pred=y_pred))
# print(calculate_qwk(y_true=y_true, y_pred=y_pred))
# print(calculate_accuracy(y_true=y_true, y_pred=y_pred))