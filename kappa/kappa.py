import pandas as pd
from sklearn.metrics import cohen_kappa_score

import sys


def calculate_kappa(df):

    # Compute Cohen's kappa between all columns
    columns = df.columns
    kappa_matrix = pd.DataFrame(index=columns, columns=columns)

    for col1 in columns:
        for col2 in columns:
            if col1 != col2:
                kappa_matrix.loc[col1, col2] = str("%.2f" % cohen_kappa_score(df[col1], df[col2]))
            else:
                kappa_matrix.loc[col1, col2] = 1.0  # Kappa with itself is always 1

    print(kappa_matrix)



# calculate_kappa('ratings.csv')

def get_majority(row):

    row = row[['ScoreRater1', 'ScoreRater2', 'ScoreRater3']]

    label_counts = dict(row.value_counts())
    label_counts_reversed = {count: label for label, count in label_counts.items()}
    # print(label_counts_reversed)

    if len(label_counts) == 1:

        return label_counts_reversed[3]

    elif len(label_counts) == 2:

        return label_counts_reversed[2]

    else:

        return 'disagree'


# for file in ['ME27b_trial_with_agreement.csv', 'PS4bp_trial_with_agreement.csv', 'VB1_trial_with_agreement.csv']:

#     df = pd.read_csv(file)
#     df = df[['Score', 'Annotation (KS)', 'Annotation (MB)', 'Annotation (TZ)', 'Gold_annotation']]

#     for col in df.columns:

#         df[col] = df[col].astype(str)

#     # df['majority'] = df.apply(get_majority, axis=1)

#     print(file)
#     print(calculate_kappa(df))
#     print('---')


for file in ['ME27b_deepseek_ALL_with_agreement_adjudicated.csv', 'PS4bp_deepseek_ALL_with_agreement_adjudicated.csv', 'VB1_deepseek_ALL_V2_with_agreement_adjudicated.csv']:

    df = pd.read_csv(file)
    df = df[['Label_LLM', 'Annotation (KS)', 'Annotation (MB)', 'Annotation (TZ)', 'Gold_annotation']]

    for col in df.columns:

        df[col] = df[col].astype(str)

    # df['majority'] = df.apply(get_majority, axis=1)

    print(file)
    print(calculate_kappa(df))
    print('---')