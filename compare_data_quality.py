import pandas as pd
import os
import sys

from sklearn.metrics import f1_score
from train_shallow import train_shallow
from utils import get_macro_f1, get_confusion_matrix, get_three_way, get_two_way



data_path_llm = 'data/generated_llm_answers'
data_path_orig = 'data/SRA_SEB'

target_column_test = 'Score'
target_column_llm = 'label'
target_column_clean = 'label_clean'

dict_results = {}


def compare_quality(method, result_dir, num_labels=5):

    method_result_dir = result_dir + '_' + method
    
    if not os.path.exists(method_result_dir):
        os.mkdir(method_result_dir)

    for task in ['ME_27b_llm.csv', 'PS_4bp_llm.csv', 'VB_1_llm.csv']:

        if task.endswith('.csv'):

            print(task)
            task_name = task[:task.index('_llm')]
            # task_name = task[:task.index('_deepseek')]

            df_train=pd.read_csv(os.path.join(data_path_llm, task))
            df_test=pd.read_csv(os.path.join(data_path_orig, 'SRA_allAnswers_prompt'+task_name+'.tsv'), sep='\t')
            df_train.rename(columns={'text_clean':'AnswerText'}, inplace=True)

            if num_labels == 3:

                df_test[target_column_test] = df_test[target_column_test].apply(get_three_way)
                df_train[target_column_llm] = df_train[target_column_llm].apply(get_three_way)
                df_train[target_column_clean] = df_train[target_column_clean].apply(get_three_way)
            
            elif num_labels == 2:
            
                df_test[target_column_test] = df_test[target_column_test].apply(get_two_way)
                df_train[target_column_llm] = df_train[target_column_llm].apply(get_two_way)
                df_train[target_column_clean] = df_train[target_column_clean].apply(get_two_way)
            
            elif num_labels != 5:

                print('Unknown number of labels:', num_labels)
                sys.exit(0)

            true_scores = list(df_test[target_column_test])

            pred_messy = train_shallow(method=method, df_train=df_train, df_test=df_test, answer_column='AnswerText', target_column=target_column_llm)
            pred_clean = train_shallow(method=method, df_train=df_train, df_test=df_test, answer_column='AnswerText', target_column=target_column_clean)
            
            pred_orig = []
            for test_idx, df_test_orig in df_test.iterrows():

                df_test_orig = pd.DataFrame(df_test_orig).T
                df_train_orig = df_test.drop(df_test_orig.index)

                pred_orig = pred_orig + list(train_shallow(method='LR', df_train=df_train_orig, df_test=df_test_orig, answer_column='AnswerText', target_column='Score'))

            
            # print(get_confusion_matrix(first=true_scores, second=pred_orig, first_name='True', second_name='Pred Orig'))
            # print(get_confusion_matrix(first=true_scores, second=pred_messy, first_name='True', second_name='Pred LLM'))
            # print(get_confusion_matrix(first=true_scores, second=pred_clean, first_name='True', second_name='Pred LLM (clean)'))
            
            current_results = {'LOOCV orig': get_macro_f1(y_true=true_scores, y_pred=pred_orig), 'LLM_labels_full': get_macro_f1(y_true=true_scores, y_pred=pred_messy), 'LLM_labels_full_cleaned': get_macro_f1(y_true=true_scores, y_pred=pred_clean)}
            dict_results[task_name] = current_results

    df_results = pd.DataFrame.from_dict(dict_results).T
    print(df_results)
    df_results.to_csv(os.path.join(method_result_dir, 'gold_vs_llm_training_' + method + '_' + str(num_labels) +'_way.csv'))


compare_quality(method='LR', result_dir='results_clean', num_labels=2)
compare_quality(method='LR', result_dir='results_clean', num_labels=3)
compare_quality(method='LR', result_dir='results_clean', num_labels=5)
