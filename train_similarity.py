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


def train_similarity(run_path, df_train, df_val, df_test, base_model, id_column, prompt_column, target_column, answer_column, num_epochs, batch_size, cross_prompt):

    df_train_paired, df_val_paired, df_test_paired = get_paired_data_from_dataframes(df_train=df_train, df_val=df_val, df_test=df_test, id_column=id_column, prompt_column=prompt_column, target_column=target_column, cross_prompt=cross_prompt)

    train_dataset = datasets.Dataset.from_dict({
        'text1': list(df_train_paired[answer_column + '_1']),
        'text2': list(df_train_paired[answer_column + '_2']),
        'label': list(df_train_paired[target_column])
    })
    eval_dataset = datasets.Dataset.from_dict({
        'text1': list(df_val_paired[answer_column + '_1']),
        'text2': list(df_val_paired[answer_column + '_2']),
        'label': list(df_val_paired[target_column])
    })

    model = SentenceTransformer(base_model)
    model.cuda()

    loss = losses.OnlineContrastiveLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=run_path,
        # Optional training parameters:
        num_train_epochs=num_epochs,
        load_best_model_at_end=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy='epoch',
        save_strategy='epoch'
    )

    dev_evaluator = evaluation.EmbeddingSimilarityEvaluator(df_val_paired[answer_column + "_1"].tolist(), df_val_paired[answer_column + "_2"].tolist(), df_val_paired[target_column].tolist(), write_csv=True)
    dev_evaluator(model)

    # 7. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=dev_evaluator,
    )
    trainer.train()
    
    # Obtain test predictions
    model.eval()
    
    with torch.no_grad():

        df_train_copy = deepcopy(df_train)
        df_test_copy = deepcopy(df_test)

        df_train_copy['embedding'] = df_train_copy[answer_column].apply(model.encode)
        df_test_copy['embedding'] = df_test_copy[answer_column].apply(model.encode)

    df_inference = cross_dataframes(df=df_test_copy, df_ref=df_train_copy)
    df_inference['sim'] = df_inference.apply(lambda row: row['embedding_1'] @ row['embedding_2'], axis=1)
    test_answers, test_predictions, test_true_scores = get_preds_from_pairs(df=df_inference, id_column=id_column+'_1', pred_column='sim', ref_label_column=target_column+'_2', true_label_column=target_column+'_1')

    df_test_aggregated = pd.DataFrame({'submission_id': test_answers, 'pred': test_predictions, target_column: test_true_scores})
    df_test_aggregated.to_csv(os.path.join(run_path, 'test_preds.csv'))
    write_stats(target_dir=run_path, y_true=test_true_scores, y_pred=test_predictions)

    for checkpoint in glob.glob(os.path.join(run_path, 'checkpoint*')):
        shutil.rmtree(checkpoint)

    df_test_merged = pd.merge(left=df_test, right=df_test_aggregated, left_on=id_column, right_on='submission_id')

    return list(df_test_merged['pred'])


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

    df_val_pairs = cross_dataframes(df=df_val, df_ref=df_train)
    df_test_pairs = cross_dataframes(df=df_test, df_ref=df_train)

    print(df_train_pairs.columns)
    print(df_val_pairs.columns)
    print(df_test_pairs.columns)

    for df_split in [df_train_pairs, df_val_pairs, df_test_pairs]:
    
        df_split[target_column] = (df_split[target_column+'_1'] == df_split[target_column+'_2']).astype(int)

    return df_train_pairs, df_val_pairs, df_test_pairs


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

    return answer, pred_labels, true_labels