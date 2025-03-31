import pandas as pd
import os
import sys

from sklearn.metrics import f1_score
from train_shallow import train_shallow
from utils import get_macro_f1, get_confusion_matrix, get_three_way, get_two_way

from statistics import stdev


data_path_llm = 'data/generated_llm_answers'
data_path_orig = 'data/SRA_SEB'

target_column_test = 'Score'
target_column = 'label_clean'

dict_results = {}


def compare_same_distribution(method, result_dir, num_runs=20, num_labels=5):

    method_results_dir = result_dir + '_' + method

    if not os.path.exists(method_results_dir):
        os.mkdir(method_results_dir)

    for task in ['ME_27b_llm.csv', 'PS_4bp_llm.csv', 'VB_1_llm.csv']:

        if task.endswith('.csv'):

            print(task)
            task_name = task[:task.index('_llm')]

            df_train=pd.read_csv(os.path.join(data_path_llm, task))
            df_test=pd.read_csv(os.path.join(data_path_orig, 'SRA_allAnswers_prompt'+task_name+'.tsv'), sep='\t')
            df_train.rename(columns={'text_clean':'AnswerText'}, inplace=True)

            if num_labels == 3:

                df_test[target_column_test] = df_test[target_column_test].apply(get_three_way)
                df_train[target_column] = df_train[target_column].apply(get_three_way)
            
            elif num_labels == 2:
            
                df_test[target_column_test] = df_test[target_column_test].apply(get_two_way)
                df_train[target_column] = df_train[target_column].apply(get_two_way)
            
            elif num_labels != 5:

                print('Unknown number of labels:', num_labels)
                sys.exit(0)

            true_scores = list(df_test[target_column_test])

            pred_full = train_shallow(method=method, df_train=df_train, df_test=df_test, answer_column='AnswerText', target_column=target_column)

            pred_orig = []
            for test_idx, df_test_orig in df_test.iterrows():

                df_test_orig = pd.DataFrame(df_test_orig).T
                df_train_orig = df_test.drop(df_test_orig.index)

                pred_orig = pred_orig + list(train_shallow(method='LR', df_train=df_train_orig, df_test=df_test_orig, answer_column='AnswerText', target_column=target_column_test))


            current_results = {'LOOCV_orig': get_macro_f1(y_true=true_scores, y_pred=pred_orig), 'LLM_labels_full_clean': get_macro_f1(y_true=true_scores, y_pred=pred_full)}

            # n runs on same distribution as in target data
            target_dist = dict(df_test[target_column_test].value_counts())

            performances = []
            for run in range(1, num_runs+1):

                dfs = []
                for label, count in target_dist.items():

                    df_label = df_train[df_train[target_column] == label]

                    # print(label, len(df_label), count)

                    df_label_sample = df_label.sample(count)
                    dfs.append(df_label_sample)

                df_train_sampled = pd.concat(dfs)

                # print(target_dist)
                # print(dict(df_train_sampled[target_column].value_counts()))

                pred_sample = train_shallow(method=method, df_train=df_train_sampled, df_test=df_test, answer_column='AnswerText', target_column=target_column)
                f1_sample = get_macro_f1(y_true=true_scores, y_pred=pred_sample)
                performances.append(f1_sample)
                current_results['sample_'+str(run)] = f1_sample
            
            current_results['sample_avg'] = sum(performances)/len(performances)
            current_results['sample_max'] = max(performances)
            current_results['sample_min'] = min(performances)
            current_results['sample_sd'] = stdev(performances)

            dict_results[task_name] = current_results

    df_results = pd.DataFrame.from_dict(dict_results).T
    df_results.index.name = 'prompt'
    print(df_results)
    df_results.to_csv(os.path.join(method_results_dir, 'same_distribution' + method + '_clean_' + str(num_labels) + '_way.csv'))


compare_same_distribution(method='LR', num_labels=5, result_dir='results_clean')
compare_same_distribution(method='LR', num_labels=3, result_dir='results_clean')
compare_same_distribution(method='LR', num_labels=2, result_dir='results_clean')
