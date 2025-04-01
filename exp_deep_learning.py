import pandas as pd
import os
import sys

from train_similarity import train_similarity
from train_bert import train_bert
from utils import get_confusion_matrix, get_three_way, get_two_way, write_stats
from metrics import calculate_macro_f1

from copy import deepcopy

## For the three cleaned prompts:
# Train deep learning models on the original data (LOOCV) 
# Train deep leanring models on the cleaned generated data

# Validation data is always a 10 percent sample of the training data


data_path_llm = 'data/generated_llm_answers'
data_path_orig = 'data/SRA_SEB'

target_column_test = 'Score'
target_column_llm = 'label'
target_column_clean = 'label_clean'

dict_results = {}


def compare_quality(base_model, result_dir, num_labels=5, num_epochs=10, batch_size=8, random_state=2356):

    base_model_name = base_model[base_model.index('/')+1:]

    method_result_dir = result_dir + '_' + base_model_name
    
    if not os.path.exists(method_result_dir):
        os.mkdir(method_result_dir)

    for task in ['ME_27b_llm.csv', 'PS_4bp_llm.csv', 'VB_1_llm.csv']:

        print(task)
        task_name = task[:task.index('_llm')]

        prompt_result_dir = os.path.join(method_result_dir, task_name)
        if not os.path.exists(prompt_result_dir):
            os.mkdir(prompt_result_dir)

        df_train=pd.read_csv(os.path.join(data_path_llm, task))
        df_test=pd.read_csv(os.path.join(data_path_orig, 'SRA_allAnswers_prompt'+task_name+'.tsv'), sep='\t')
        df_train.rename(columns={'text_clean':'AnswerText', 'id': 'AnswerId', 'question': 'PromptId'}, inplace=True)

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


        # Sample val
        df_val = df_train.sample(frac=.1, random_state=random_state)
        df_train = df_train.drop(df_val.index)

        # df_train=df_train.head(7)
        # df_val=df_val.head(7)
        # df_test=df_test.head(7)

        true_scores = list(df_test[target_column_test])

        # LOOCV on original data
        pred_loocv = []

        for test_idx, df_test_orig in df_test.iterrows():

            df_test_orig = pd.DataFrame(df_test_orig).T
            df_train_orig = df_test.drop(df_test_orig.index)

            # Sample val
            df_val_orig = df_train_orig.sample(frac=.1, random_state=random_state)
            df_train_orig = df_train_orig.drop(df_val_orig.index)

            if base_model == 'all-MiniLM-L6-v2':
                pred_loocv = pred_loocv + train_similarity(run_path=os.path.join(prompt_result_dir, 'orig'), df_train=df_train_orig, df_val=df_val_orig, df_test=df_test_orig, base_model=base_model, id_column='AnswerId', target_column=target_column_test, prompt_column='PromptId', answer_column='AnswerText', num_epochs=num_epochs, batch_size=batch_size, cross_prompt=False)
            
            else:
                pred_loocv = pred_loocv + train_bert(run_path=os.path.join(prompt_result_dir, 'orig'), df_train=df_train_orig, df_val=df_val_orig, df_test=df_test_orig, base_model=base_model, id_column='AnswerId', target_column=target_column_test, prompt_column='PromptId', answer_column='AnswerText', num_epochs=num_epochs, batch_size=batch_size)

        write_stats(target_dir=os.path.join(prompt_result_dir, 'orig'), y_true=true_scores, y_pred=pred_loocv)

        # Train on generated data
        if base_model == 'all-MiniLM-L6-v2':
            df_test = df_test.rename(columns={target_column_test: target_column_llm})
            pred_messy = train_similarity(run_path=os.path.join(prompt_result_dir, 'messy'), df_train=df_train, df_val=df_val, df_test=df_test, base_model=base_model, id_column='AnswerId', target_column=target_column_llm, prompt_column='PromptId', answer_column='AnswerText', num_epochs=num_epochs, batch_size=batch_size, cross_prompt=False)
            df_test = df_test.rename(columns={target_column_llm: target_column_clean})
            pred_clean = train_similarity(run_path=os.path.join(prompt_result_dir, 'clean'), df_train=df_train, df_val=df_val, df_test=df_test, base_model=base_model, id_column='AnswerId', target_column=target_column_clean, prompt_column='PromptId', answer_column='AnswerText', num_epochs=num_epochs, batch_size=batch_size, cross_prompt=False)
        
        else:
            df_test = df_test.rename(columns={target_column_test: target_column_llm})
            pred_messy = train_bert(run_path=os.path.join(prompt_result_dir, 'messy'), df_train=df_train, df_val=df_val, df_test=df_test, base_model=base_model, id_column='AnswerId', target_column=target_column_llm, prompt_column='PromptId', answer_column='AnswerText', num_epochs=num_epochs, batch_size=batch_size)
            df_test = df_test.rename(columns={target_column_llm: target_column_clean})
            pred_clean = train_bert(run_path=os.path.join(prompt_result_dir, 'clean'), df_train=df_train, df_val=df_val, df_test=df_test, base_model=base_model, id_column='AnswerId', target_column=target_column_clean, prompt_column='PromptId', answer_column='AnswerText', num_epochs=num_epochs, batch_size=batch_size)

        df_test_copy = deepcopy(df_test)
        df_test_copy['pred_messy'] = pred_messy
        df_test_copy['pred_clean'] = pred_clean
        df_test_copy['pred_LOOCV'] = pred_loocv
        df_test_copy.to_csv(os.path.join(prompt_result_dir, str(num_labels) + '_way_preds.csv'))
        
        current_results = {'LOOCV': calculate_macro_f1(y_true=true_scores, y_pred=pred_loocv), 'LLM_labels_full': calculate_macro_f1(y_true=true_scores, y_pred=pred_messy), 'LLM_labels_full_cleaned': calculate_macro_f1(y_true=true_scores, y_pred=pred_clean)}
        dict_results[task_name] = current_results

        df_results = pd.DataFrame.from_dict(dict_results).T
        df_results.to_csv(os.path.join(method_result_dir, 'gold_vs_llm_training_' + base_model + '_' + str(num_labels) +'_way.csv'))


# compare_quality(base_model='all-MiniLM-L6-v2', result_dir='results')
compare_quality(base_model='answerdotai/ModernBERT-base', result_dir='results')
# compare_quality(base_model='bert-base-uncased', result_dir='results')