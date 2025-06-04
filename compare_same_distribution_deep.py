import pandas as pd
import os
import sys
import random

from sklearn.metrics import f1_score
from train_bert import train_bert
from train_similarity import train_similarity
from utils import get_confusion_matrix, get_three_way, get_two_way
from metrics import calculate_macro_f1

from statistics import stdev
from copy import deepcopy


data_path_llm = 'data/generated_llm_answers'
data_path_llm_2 = 'data/generated_llm_answers_2'
data_path_orig = 'data/SRA_SEB'

target_column_test = 'Score'
target_column = 'label'

dict_results = {}


def compare_same_distribution(base_model, result_dir, num_runs=20, num_labels=5, batch_size=8, num_epochs=5, random_state=2356):

    base_model_name = base_model
    if '/' in base_model:
        base_model_name = base_model[base_model.index('/')+1:]

    method_results_dir = result_dir + '_' + base_model_name

    if not os.path.exists(method_results_dir):
        os.mkdir(method_results_dir)

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

            current_results = {}

            pred_orig = []
            pred_orig_max = []
            for test_idx, df_test_orig in df_test.iterrows():

                df_test_orig = pd.DataFrame(df_test_orig).T
                df_train_orig = df_test.drop(df_test_orig.index)

                # Sample val
                df_val_orig = df_train_orig.sample(frac=.1, random_state=random_state)
                df_train_orig = df_train_orig.drop(df_val_orig.index)

                if base_model == 'bert-base-uncased':
                    pred_orig = pred_orig + train_bert(run_path=os.path.join(prompt_result_dir, 'orig'), df_train=df_train_orig, df_val=df_val_orig, df_test=df_test_orig, base_model=base_model, id_column='AnswerId', target_column=target_column_test, prompt_column='PromptId', answer_column='AnswerText', num_epochs=num_epochs, batch_size=batch_size)


                elif base_model == 'all-MiniLM-L6-v2':
                    pred, pred_max = train_similarity(run_path=os.path.join(prompt_result_dir, 'orig'), df_train=df_train_orig, df_val=df_val_orig, df_test=df_test_orig, base_model=base_model, id_column='AnswerId', target_column=target_column_test, prompt_column='PromptId', answer_column='AnswerText', cross_prompt=False, num_epochs=num_epochs, batch_size=batch_size)
                    pred_orig = pred_orig + pred
                    pred_orig_max = pred_orig_max + pred_max

                else:
                    print('Unknown method', method)
                    sys.exit(0)

            current_results = {'LOOCV_orig': calculate_macro_f1(y_true=true_scores, y_pred=pred_orig)}

            if base_model == 'all-MiniLM-L6-v2':
                current_results['LOOCV_orig_max'] = calculate_macro_f1(y_true=true_scores, y_pred=pred_orig_max)


            # n runs on same distribution as in target data
            target_dist = dict(df_test[target_column_test].value_counts())

            performances = []
            performances_max = []
            random_states = [random.randint(0,9992356) for i in range(num_runs)]
            for run in range(1, num_runs+1):

                dfs = []
                for label, count in target_dist.items():

                    df_label = df_train[df_train[target_column] == label]

                    df_label_sample = df_label.sample(count, random_state=random_states[run-1])
                    dfs.append(df_label_sample)

                df_train_sampled = pd.concat(dfs)

                # Sample val
                df_val_sampled = df_train_sampled.sample(frac=.1, random_state=random_state)
                df_train_sampled = df_train_sampled.drop(df_val_sampled.index)

                if base_model == 'bert-base-uncased':
                    pred_sample = train_bert(run_path=os.path.join(prompt_result_dir, 'sample_' + str(run)), df_train=df_train_sampled, df_val=df_val_sampled, df_test=df_test_renamed, base_model=base_model, id_column='AnswerId', target_column=target_column, prompt_column='PromptId', answer_column='AnswerText', num_epochs=num_epochs, batch_size=batch_size)

                elif base_model == 'all-MiniLM-L6-v2':
                    pred_sample, pred_sample_max = train_similarity(run_path=os.path.join(prompt_result_dir, 'sample_'+str(run)), df_train=df_train_sampled, df_val=df_val_sampled, df_test=df_test_renamed, base_model=base_model, id_column='AnswerId', target_column=target_column, prompt_column='PromptId', answer_column='AnswerText', cross_prompt=False, num_epochs=num_epochs, batch_size=batch_size)

                else:
                    print('Unknown method', base_model)
                    sys.exit(0)

                f1_sample = calculate_macro_f1(y_true=true_scores, y_pred=pred_sample)
                performances.append(f1_sample)
                current_results['sample_'+str(run)] = f1_sample

                if base_model == 'all-MiniLM-L6-v2':
                    f1_sample_max = calculate_macro_f1(y_true=true_scores, y_pred=pred_sample_max)
                    performances_max.append(f1_sample_max)
                    current_results['max_sample_'+str(run)] = f1_sample_max
                    
            
            current_results['sample_avg'] = sum(performances)/len(performances)
            current_results['sample_max'] = max(performances)
            current_results['sample_min'] = min(performances)
            current_results['sample_sd'] = stdev(performances)

            if base_model == 'all-MiniLM-L6-v2':
                current_results['max_sample_avg'] = sum(performances_max)/len(performances_max)
                current_results['max_sample_max'] = max(performances_max)
                current_results['max_sample_min'] = min(performances_max)
                current_results['max_sample_sd'] = stdev(performances_max)


            dict_results[task_name] = current_results

            df_results = pd.DataFrame.from_dict(dict_results).T
            df_results.index.name = 'prompt'
            print(df_results)
            df_results.to_csv(os.path.join(method_results_dir, 'same_distribution' + base_model_name + str(num_labels) + '_way.csv'))


compare_same_distribution(base_model='all-MiniLM-L6-v2', num_labels=5, result_dir='results_same_dist_SBERT', num_epochs=5)
compare_same_distribution(base_model='bert-base-uncased', num_labels=5, result_dir='results_same_dist_BERT', num_epochs=10)