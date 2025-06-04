import os

import pandas as pd
from metrics import calculate_macro_f1

data_source = 'data/SRA_SEB'
target_column = 'Score'
results_dir = 'results_majority'

if not os.path.exists(results_dir):
    os.mkdir(results_dir)

df_prompts = pd.read_csv('data/prompts_without_pictures.csv')

results = {}

for prompt in df_prompts['id']:

    df = pd.read_csv(os.path.join(data_source, 'SRA_allAnswers_prompt' + prompt + '.tsv'), sep='\t')
    label_dict = dict(df[target_column].value_counts())

    most_frequent_label = max(label_dict, key=label_dict.get)
    
    results[prompt] = calculate_macro_f1(y_true=df[target_column], y_pred=[most_frequent_label]*len(df))

df_results = pd.DataFrame.from_dict(results, orient='index')
df_results.columns=['f1_majority']
df_results.index.name='prompt'

df_results.to_csv(os.path.join(results_dir, 'majority.csv'))