import pandas as pd
import os
import sys

from sklearn.metrics import f1_score
from train_shallow import train_shallow
from train_similarity_pretrained import train_similarity_pretrained
from utils import get_confusion_matrix, get_three_way, get_two_way
from metrics import calculate_macro_f1

from statistics import stdev
from copy import deepcopy


data_path_llm = 'data/generated_llm_answers'
data_path_llm_2 = 'data/generated_llm_answers_2'
data_path_orig = 'data/SRA_SEB'

target_column_test = 'Score'
target_column = 'label'


## Compare training on original data and SRA-gen data with same distribution as in original data.
## SRA-gen with as-generated labels
## LR and pretrained SBERT model

def compare_same_distribution(method, num_runs=20, num_labels=5, results_dir='results'):

    method_results_dir = os.path.join(results_dir + '_' + method)

    if not os.path.exists(method_results_dir):
        os.mkdir(method_results_dir)

    dict_results = {}

    for task in os.listdir(data_path_llm):

        if task.endswith('.csv'):

            print(task)
            task_name = task[:task.index('_llm')]

            prompt_result_dir = os.path.join(method_results_dir, task_name)

            if not os.path.exists(prompt_result_dir):
                os.mkdir(prompt_result_dir)

            dfs_train = []
            df_train_1=pd.read_csv(os.path.join(data_path_llm, task))
            dfs_train.append(df_train_1)
            df_train_2=pd.read_csv(os.path.join(data_path_llm_2, task))
            dfs_train.append(df_train_2)
            df_train = pd.concat(dfs_train)

            df_test=pd.read_csv(os.path.join(data_path_orig, 'SRA_allAnswers_prompt'+task_name+'.tsv'), sep='\t')
            df_train.rename(columns={'text_clean':'AnswerText', 'id': 'AnswerId', 'question': 'PromptId'}, inplace=True)

            df_test_renamed = deepcopy(df_test)
            df_test_renamed.rename(columns={target_column_test: target_column}, inplace=True)

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

            if method == 'LR':
                pred_full = train_shallow(method=method, df_train=df_train, df_test=df_test, answer_column='AnswerText', target_column=target_column)
            
            elif method == 'all-MiniLM-L6-v2':
                pred_full, pred_full_max = train_similarity_pretrained(run_path=os.path.join(prompt_result_dir, 'clean'), df_train=df_train, df_val=None, df_test=df_test_renamed, base_model=method, id_column='AnswerId', target_column=target_column, prompt_column='PromptId', answer_column='AnswerText', cross_prompt=False)
            
            else:
                print('Unknown method', method)
                sys.exit(0)

            pred_orig = []
            pred_orig_max = []
            for test_idx, df_test_orig in df_test.iterrows():

                df_test_orig = pd.DataFrame(df_test_orig).T
                df_train_orig = df_test.drop(df_test_orig.index)

                if method == 'LR':
                    pred_orig = pred_orig + list(train_shallow(method='LR', df_train=df_train_orig, df_test=df_test_orig, answer_column='AnswerText', target_column=target_column_test))

                elif method == 'all-MiniLM-L6-v2':
                    pred, pred_max = train_similarity_pretrained(run_path=os.path.join(prompt_result_dir, 'orig'), df_train=df_train_orig, df_val=None, df_test=df_test_orig, base_model=method, id_column='AnswerId', target_column=target_column_test, prompt_column='PromptId', answer_column='AnswerText', cross_prompt=False)
                    pred_orig = pred_orig + pred
                    pred_orig_max = pred_orig_max + pred_max

                else:
                    print('Unknown method', method)
                    sys.exit(0)

            current_results = {'LOOCV_orig': calculate_macro_f1(y_true=true_scores, y_pred=pred_orig), 'LLM_labels_full': calculate_macro_f1(y_true=true_scores, y_pred=pred_full)}

            if pred_full_max is not None:
                current_results['LOOCV_orig_max'] = calculate_macro_f1(y_true=true_scores, y_pred=pred_orig_max)
                current_results['LLM_labels_full_clean_max'] = calculate_macro_f1(y_true=true_scores, y_pred=pred_full_max)

            # n runs on same distribution as in target data
            target_dist = dict(df_test[target_column_test].value_counts())

            performances = []
            performances_max = []
            for run in range(1, num_runs+1):

                dfs = []
                for label, count in target_dist.items():

                    df_label = df_train[df_train[target_column] == label]

                    # print(label, len(df_label), count)

                    df_label_sample = df_label.sample(count)
                    dfs.append(df_label_sample)

                df_train_sampled = pd.concat(dfs)

                if method == 'LR':
                    pred_sample = train_shallow(method=method, df_train=df_train_sampled, df_test=df_test, answer_column='AnswerText', target_column=target_column)

                elif method == 'all-MiniLM-L6-v2':
                    pred_sample, pred_sample_max = train_similarity_pretrained(run_path=os.path.join(prompt_result_dir, 'sample_'+str(run)), df_train=df_train_sampled, df_val=None, df_test=df_test_renamed, base_model=method, id_column='AnswerId', target_column=target_column, prompt_column='PromptId', answer_column='AnswerText', cross_prompt=False)

                else:
                    print('Unknown method', method)
                    sys.exit(0)

                f1_sample = calculate_macro_f1(y_true=true_scores, y_pred=pred_sample)
                performances.append(f1_sample)
                current_results['sample_'+str(run)] = f1_sample

                if pred_full_max is not None:
                    f1_sample_max = calculate_macro_f1(y_true=true_scores, y_pred=pred_sample_max)
                    performances_max.append(f1_sample_max)
                    current_results['max_sample_'+str(run)] = f1_sample_max
            
            current_results['sample_avg'] = sum(performances)/len(performances)
            current_results['sample_max'] = max(performances)
            current_results['sample_min'] = min(performances)
            current_results['sample_sd'] = stdev(performances)

            if pred_full_max is not None:
                current_results['max_sample_avg'] = sum(performances_max)/len(performances_max)
                current_results['max_sample_max'] = max(performances_max)
                current_results['max_sample_min'] = min(performances_max)
                current_results['max_sample_sd'] = stdev(performances_max)

            dict_results[task_name] = current_results

    df_results = pd.DataFrame.from_dict(dict_results).T
    df_results.index.name = 'prompt'
    print(df_results)
    df_results.to_csv(os.path.join(method_results_dir, 'same_distribution' + method + '_' + str(num_labels) + '_way.csv'))


compare_same_distribution(method='LR', num_labels=5, results_dir='results_same_dist_LR')
# compare_same_distribution(method='LR', num_labels=3, results_dir='results_same_dist_LR')
# compare_same_distribution(method='LR', num_labels=2, results_dir='results_same_dist_LR')

compare_same_distribution(method='all-MiniLM-L6-v2', num_labels=5, results_dir='results_same_dist_pretrained')