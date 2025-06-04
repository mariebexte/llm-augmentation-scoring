import pandas as pd
import os
import sys
import random

from sklearn.metrics import f1_score
from train_similarity import train_similarity
from utils import get_confusion_matrix, get_three_way, get_two_way
from metrics import calculate_macro_f1

from statistics import stdev


data_path_llm = 'data/generated_llm_answers'
data_path_orig = 'data/SRA_SEB'

target_column_test = 'Score'

dict_results = {}


def compare_same_distribution(base_model, result_dir, num_runs=20, num_labels=5, sizes=[2,3,4,5,6,7,14,28], random_state=2356, num_epochs=5, batch_size=8, target_column='label_clean'):

    method_result_dir = result_dir + '_' + base_model

    if not os.path.exists(method_result_dir):
        os.mkdir(method_result_dir)

    for task in ['ME_27b_llm.csv', 'PS_4bp_llm.csv', 'VB_1_llm.csv']:

        if task.endswith('.csv'):

            print(task)
            task_name = task[:task.index('_llm')]

            current_results = {}

            prompt_result_dir = os.path.join(method_result_dir, task_name)
            if not os.path.exists(prompt_result_dir):
                os.mkdir(prompt_result_dir)

            for num in sizes:

                performances = []
                performances_max = []

                for run in range(1, num_runs+1):

                    df_preds = pd.read_csv(os.path.join(prompt_result_dir, 'sample_' + str(num) + '_' + str(run), 'test_preds.csv'))
                    df_preds_max = pd.read_csv(os.path.join(prompt_result_dir, 'sample_' + str(num) + '_' + str(run), 'test_preds_max.csv'))

                    pred_sample = df_preds['pred'].tolist()
                    pred_sample_max = df_preds_max['pred'].tolist()

                    true_scores = df_preds['label_clean'].tolist()
                    true_scores_max = df_preds_max['label_clean'].tolist()

                    f1_sample = calculate_macro_f1(y_true=true_scores, y_pred=pred_sample)
                    performances.append(f1_sample)
                    current_results[str(num)+'_'+str(run)] = f1_sample

                    f1_sample_max = calculate_macro_f1(y_true=true_scores_max, y_pred=pred_sample_max)
                    performances_max.append(f1_sample_max)
                    current_results['max_' + str(num)+'_'+str(run)] = f1_sample_max
            
                current_results[str(num) + '_avg'] = sum(performances)/len(performances)
                current_results[str(num) + '_max'] = max(performances)
                current_results[str(num) + '_min'] = min(performances)
                current_results[str(num) + '_sd'] = stdev(performances)

                current_results['max_' + str(num) + '_avg'] = sum(performances_max)/len(performances_max)
                current_results['max_' + str(num) + '_max'] = max(performances_max)
                current_results['max_' + str(num) + '_min'] = min(performances_max)
                current_results['max_' + str(num) + '_sd'] = stdev(performances_max)

            dict_results[task_name] = current_results

    df_results = pd.DataFrame.from_dict(dict_results).T
    df_results.index.name = 'prompt'
    print(df_results)
    df_results.to_csv(os.path.join(method_result_dir, 'llm_balanced' + base_model + '_clean_' + str(num_labels) + '_way.csv'))


compare_same_distribution(base_model='all-MiniLM-L6-v2', result_dir='results_CURVES_clean_balanced_all-MiniLM-L6-v2', num_labels=5, target_column='label_clean')
compare_same_distribution(base_model='all-MiniLM-L6-v2', result_dir='results_CURVES_dirty_balanced_all-MiniLM-L6-v2', num_labels=5, target_column='label')
