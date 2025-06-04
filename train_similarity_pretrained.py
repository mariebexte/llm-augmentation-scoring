from torch.utils.data import DataLoader
import torch.nn.functional as F

import torch
import sys
import os
import shutil

import pandas as pd
import numpy as np
from tqdm import tqdm
import glob

from copy import deepcopy
from metrics import calculate_accuracy, calculate_gwets_ac2, calculate_macro_f1, calculate_qwk
from utils import write_stats

import datasets
from sentence_transformers import SentenceTransformer, losses, evaluation, SentenceTransformerTrainer, SentenceTransformerTrainingArguments


def train_similarity_pretrained(run_path, df_train, df_val, df_test, base_model, id_column, prompt_column, target_column, answer_column, cross_prompt=False):

    if not os.path.exists(run_path):
        os.mkdir(run_path)
    
    if df_val is not None:
        df_train = pd.concat([df_train, df_val])
    df_train_paired, _, df_test_paired = get_paired_data_from_dataframes(df_train=df_train, df_val=df_val, df_test=df_test, id_column=id_column, prompt_column=prompt_column, target_column=target_column, cross_prompt=cross_prompt)
    
    model = SentenceTransformer(base_model)
    model.cuda()

    with torch.no_grad():

        df_train_copy = deepcopy(df_train)
        df_test_copy = deepcopy(df_test)

        df_train_copy['embedding'] = df_train_copy[answer_column].apply(model.encode)
        df_test_copy['embedding'] = df_test_copy[answer_column].apply(model.encode)

    df_inference = cross_dataframes(df=df_test_copy, df_ref=df_train_copy)
    df_inference['sim'] = df_inference.apply(lambda row: row['embedding_1'] @ row['embedding_2'], axis=1)

    answer_ids, test_predictions, test_true_scores = get_preds_from_pairs(df=df_inference, id_column=id_column+'_1', pred_column='sim', ref_label_column=target_column+'_2', true_label_column=target_column+'_1')
    df_test_aggregated = pd.DataFrame({'submission_id': answer_ids, 'pred': test_predictions, target_column: test_true_scores})
    df_test_aggregated.to_csv(os.path.join(run_path, 'test_preds.csv'))
    write_stats(target_dir=run_path, y_true=test_true_scores, y_pred=test_predictions)

    answer_ids_max, test_predictions_max, test_true_scores_max = get_preds_from_pairs_max(df=df_inference, id_column=id_column+'_1', pred_column='sim', ref_label_column=target_column+'_2', true_label_column=target_column+'_1')
    df_test_aggregated_max = pd.DataFrame({'submission_id': answer_ids_max, 'pred': test_predictions_max, target_column: test_true_scores_max})
    df_test_aggregated_max.to_csv(os.path.join(run_path, 'test_preds_max.csv'))
    write_stats(target_dir=run_path, y_true=test_true_scores, y_pred=test_predictions_max, prefix='max_')

    df_test_merged = pd.merge(left=df_test, right=df_test_aggregated, left_on=id_column, right_on='submission_id')
    df_test_merged_max = pd.merge(left=df_test, right=df_test_aggregated_max, left_on=id_column, right_on='submission_id')

    return list(df_test_merged['pred']), list(df_test_merged_max['pred'])


# _2 is ref!
def cross_dataframes(df, df_ref):

    return pd.merge(left=df, right=df_ref, how='cross', suffixes=('_1', '_2'))


def get_paired_data_from_dataframes(df_train, df_val, df_test, target_column, prompt_column='task_id', id_column='submission_id', answer_column='text', cross_prompt=False):

    dfs = [] 

    if cross_prompt:

        prompts = list(df_train[prompt_column].unique())
        num_prompts = len(prompts)

        for prompt_num in range(num_prompts):

            prompt_1 = prompts.pop()
            df_prompt_1 = df_train[df_train[prompt_column] == prompt_1]

            for prompt_2 in prompts:
                
                df_prompt_2 = df_train[df_train[prompt_column] == prompt_2]
                dfs.append(cross_dataframes(df=df_prompt_1, df_ref=df_prompt_2))
    
    else:

        for prompt, df_prompt in df_train.groupby(prompt_column):

            dfs.append(cross_dataframes(df=df_prompt, df_ref=df_prompt))
        
    df_train_pairs = pd.concat(dfs)

    if df_val is not None:
        df_val_pairs = cross_dataframes(df=df_val, df_ref=df_train)
    df_test_pairs = cross_dataframes(df=df_test, df_ref=df_train)

    if df_val is not None:
        for df_split in [df_train_pairs, df_val_pairs, df_test_pairs]:
            df_split[target_column] = (df_split[target_column+'_1'] == df_split[target_column+'_2']).astype(int)
    
        return df_train_pairs, df_val_pairs, df_test_pairs

    else:
        for df_split in [df_train_pairs, df_test_pairs]:
            df_split[target_column] = (df_split[target_column+'_1'] == df_split[target_column+'_2']).astype(int)
        
        return df_train_pairs, None, df_test_pairs




def get_preds_from_pairs(df, id_column, pred_column, ref_label_column, true_label_column):

    answer_ids = []
    pred_labels = []
    true_labels = []

    # For each test instance
    for answer, df_answer in df.groupby(id_column):

        true_label = list(df_answer[true_label_column].unique())
        if len(true_label) > 1:
            print('True label not unique!', true_label)
            sys.exit(0)

        else:
            true_label = true_label[0]

        score_probs = {}

        for label, df_label in df_answer.groupby(ref_label_column):

            # print(list(df_label[pred_column]))
            score_probs[label] = df_label[pred_column].mean()
        
        pred_label = max(score_probs, key=score_probs.get)
        # print('probabilities', score_probs)
        # print('prediction', pred_label)

        answer_ids.append(answer)
        pred_labels.append(pred_label)
        true_labels.append(true_label)

    return answer_ids, pred_labels, true_labels


def get_preds_from_pairs_max(df, id_column, pred_column, ref_label_column, true_label_column):

    answer_ids = []
    pred_labels = []
    true_labels = []

    # For each test instance
    for answer, df_answer in df.groupby(id_column):

        true_label = list(df_answer[true_label_column].unique())
        if len(true_label) > 1:
            print('True label not unique!', true_label)
            sys.exit(0)

        else:
            true_label = true_label[0]

        # Find most similar answer and its label
        max_idx = df_answer[pred_column].idxmax()
        pred_label = df_answer.loc[max_idx][ref_label_column]

        # print(max_idx, pred_label, answer)
        # print(df_answer.loc[max_idx]['AnswerText_1'])
        # print(df_answer.loc[max_idx]['AnswerText_2'])

        answer_ids.append(answer)
        pred_labels.append(pred_label)
        true_labels.append(true_label)

    return answer_ids, pred_labels, true_labels