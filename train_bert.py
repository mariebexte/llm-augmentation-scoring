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
from transformers import ModernBertForSequenceClassification, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback


def train_bert(run_path, df_train, df_val, df_test, base_model, id_column, prompt_column, target_column, answer_column, num_epochs, batch_size):

    # Model evaluation throws error if val/test data contains more labels than train
    labels_in_training = df_train[target_column].unique().tolist()
    labels_in_validation = df_val[target_column].unique().tolist()
    labels_in_test = df_test[target_column].unique().tolist()

    label_set = set(labels_in_training + labels_in_validation + labels_in_test)
    
    label_to_id = {label: label_id for label, label_id in zip(label_set, range(len(label_set)))}
    id_to_label = {label_id: label for label, label_id in label_to_id.items()}

    # If all training instances have the same label: Return this label as prediction for all testing instances
    if len(df_train[target_column].unique()) == 1:
        target_label = list(df_train[target_column].unique())
        logging.warn("All training instances have the same label '"+str(target_label[0])+"'. Predicting this label for all testing instances!")
        print("All training instances have the same label '"+str(target_label[0])+"'. Predicting this label for all testing instances!")
        return target_label*len(df_test)

    model = ModernBertForSequenceClassification.from_pretrained(base_model, num_labels=len(label_set), id2label=id_to_label, label2id=label_to_id)
    # model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=len(label_set), id2label=id_to_label, label2id=label_to_id)
    model.cuda()

    print('model was loaded')
    sys.exit(0)

    args = TrainingArguments(
        output_dir=run_path,
        # Optional training parameters:
        num_train_epochs=num_epochs,
        load_best_model_at_end=True,
        per_device_train_batch_size=batch_size,
        eval_strategy='epoch',
        save_strategy='epoch'
    )

    # 7. Create a trainer & train
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
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

    return df_test_aggregated


# To log loss of training and evaluation to file
class WriteCsvCallback(TrainerCallback):

    def __init__(self, *args, csv_train, csv_eval, **kwargs):
        super().__init__(*args, **kwargs)
        self.csv_train_path = csv_train
        self.csv_eval_path = csv_eval
        self.df_eval = pd.DataFrame()
        self.df_train_eval = pd.DataFrame()

    def on_log(self, args, state, control, logs=None, **kwargs):

        df_log = pd.DataFrame([logs])

        # Has info about performance on training data
        if "loss" in logs:
            self.df_train_eval = pd.concat([self.df_train_eval, df_log])
        
        # Has info about performance on validation data
        else:
            best_model = state.best_model_checkpoint
            df_log["best_model_checkpoint"] = best_model
            self.df_eval = pd.concat([self.df_eval, df_log])

    def on_train_end(self, args, state, control, **kwargs):
        self.df_eval.to_csv(self.csv_eval_path)
        self.df_train_eval.to_csv(self.csv_train_path)


# Which metrics to compute on evaluation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = calculate_macro_f1(labels, preds)
    return {
      'acc': acc,
      'macro f1': f1,
    }