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

            df_train=pd.read_csv(os.path.join(data_path_llm, task))
            df_test=pd.read_csv(os.path.join(data_path_orig, 'SRA_allAnswers_prompt'+task_name+'.tsv'), sep='\t')
            df_train.rename(columns={'text_clean':'AnswerText', 'id': 'AnswerId', 'question': 'PromptId'}, inplace=True)
            df_test.rename(columns={'Score': target_column}, inplace=True)

            if num_labels == 3:

                df_test[target_column_test] = df_test[target_column_test].apply(get_three_way)
                df_train[target_column] = df_train[target_column].apply(get_three_way)
            
            elif num_labels == 2:
            
                df_test[target_column_test] = df_test[target_column_test].apply(get_two_way)
                df_train[target_column] = df_train[target_column].apply(get_two_way)
            
            elif num_labels != 5:

                print('Unknown number of labels:', num_labels)
                sys.exit(0)

            true_scores = list(df_test[target_column])

            # Find label with lowest count
            nums_per_label = dict(df_train[target_column].value_counts())
            min_label = min(nums_per_label, key=nums_per_label.get)
            num_per_label = nums_per_label[min_label]

            print(task_name, num_per_label)

            for num in sizes:

                performances = []
                performances_max = []

                random_states = [random.randint(0,9992356) for i in range(num_runs)]

                for run in range(1, num_runs+1):

                    dfs = []
                    for label in df_train[target_column].unique():

                        df_label = df_train[df_train[target_column] == label]
                        df_label_sample = df_label.sample(num, random_state=random_states[run-1])
                        dfs.append(df_label_sample)

                    df_train_sampled = pd.concat(dfs)

                    # print(dict(df_train_sampled[target_column].value_counts()))

                    # Sample val
                    df_val_sampled = df_train_sampled.sample(frac=.1, random_state=random_state)
                    df_train_sampled = df_train_sampled.drop(df_val_sampled.index)

                    pred_sample, pred_sample_max = train_similarity(run_path=os.path.join(prompt_result_dir, 'sample_' + str(num) + '_' +str(run)), df_train=df_train_sampled, df_val=df_val_sampled, df_test=df_test, base_model=base_model, id_column='AnswerId', target_column=target_column, prompt_column='PromptId', answer_column='AnswerText', cross_prompt=False, num_epochs=num_epochs, batch_size=batch_size)
                    
                    f1_sample = calculate_macro_f1(y_true=true_scores, y_pred=pred_sample)
                    performances.append(f1_sample)
                    current_results[str(num)+'_'+str(run)] = f1_sample

                    f1_sample_max = calculate_macro_f1(y_true=true_scores, y_pred=pred_sample_max)
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


compare_same_distribution(base_model='all-MiniLM-L6-v2', result_dir='results_CURVES_clean_balanced', num_labels=5, sizes=[5,10,15,20,25])
compare_same_distribution(base_model='all-MiniLM-L6-v2', result_dir='results_CURVES_dirty_balanced', num_labels=5, target_column='label', sizes=[5,10,15,20,25])
# compare_same_distribution(base_model='all-MiniLM-L6-v2', result_dir='results_CURVES_dirty_balanced', num_labels=5, target_column='label', sizes=[2,3,4,5,6,7,14,28,50,100])
