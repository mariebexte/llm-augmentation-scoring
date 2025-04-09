import torch
import sys
import os
import shutil

import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
import logging

from copy import deepcopy
from metrics import calculate_accuracy, calculate_gwets_ac2, calculate_macro_f1, calculate_qwk
from utils import write_stats

from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback


def train_bert(run_path, df_train, df_val, df_test, base_model, id_column, prompt_column, target_column, answer_column, num_epochs, batch_size):

    # Model evaluation throws error if val/test data contains more labels than train
    labels_in_training = df_train[target_column].unique().tolist()
    labels_in_validation = df_val[target_column].unique().tolist()
    labels_in_test = df_test[target_column].unique().tolist()

    label_set = set(labels_in_training + labels_in_validation + labels_in_test)
    
    label_to_id = {label: label_id for label, label_id in zip(label_set, range(len(label_set)))}
    id_to_label = {label_id: label for label, label_id in label_to_id.items()}

    train_texts = list(df_train.loc[:, answer_column])
    train_labels = list(df_train.loc[:, target_column])
    train_labels = [label_to_id[label] for label in train_labels]
    val_texts = list(df_val.loc[:, answer_column])
    val_labels = list(df_val.loc[:, target_column])
    val_labels = [label_to_id[label] for label in val_labels]
    test_texts = list(df_test.loc[:, answer_column])
    test_labels = list(df_test.loc[:, target_column])
    test_labels = [label_to_id[label] for label in test_labels]

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=len(label_set), id2label=id_to_label, label2id=label_to_id)
    model.cuda()

    # Tokenize the dataset, truncate if longer than max_length, pad with 0's when less than `max_length`
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    # Convert tokenized data into torch Dataset
    train_dataset = Dataset(train_encodings, train_labels)
    eval_dataset = Dataset(val_encodings, val_labels)
    test_dataset = Dataset(test_encodings, test_labels)

    # train_dataset = Dataset.from_dict({'text': train_encodings, 'label': train_labels})
    # eval_dataset = Dataset.from_dict({'text': val_encodings, 'label': val_labels})
    # test_dataset = Dataset.from_dict({'text': test_encodings, 'label': test_labels})

    # If all training instances have the same label: Return this label as prediction for all testing instances
    if len(df_train[target_column].unique()) == 1:
        target_label = list(df_train[target_column].unique())
        logging.warn("All training instances have the same label '"+str(target_label[0])+"'. Predicting this label for all testing instances!")
        print("All training instances have the same label '"+str(target_label[0])+"'. Predicting this label for all testing instances!")
        return target_label*len(df_test)

    args = TrainingArguments(
        output_dir=run_path,
        num_train_epochs=num_epochs,
        load_best_model_at_end=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy='epoch',
        save_strategy='epoch'
    )

    # Create a trainer & train
    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    dict_val_loss = {}
    dict_test_preds = {}

    trainer.add_callback(WriteCsvCallback(csv_train=os.path.join(run_path, "train_stats.csv"), csv_eval=os.path.join(run_path, "eval_stats.csv"), dict_val_loss=dict_val_loss))
    trainer.add_callback(GetTestPredictionsCallback(dict_test_preds=dict_test_preds, save_path=os.path.join(run_path, "test_stats.csv"), trainer=trainer, test_data=test_dataset))
    trainer.train()
    
    # Determine epoch with lowest validation loss
    best_epoch = min(dict_val_loss, key=dict_val_loss.get)

    # For this epoch, return test predictions
    predictions = dict_test_preds[best_epoch]

    # Obtain test predictions
    model.eval()

    for checkpoint in glob.glob(os.path.join(run_path, 'checkpoint*')):
        shutil.rmtree(checkpoint)

    predictions = list(predictions)
    predictions = [id_to_label[pred] for pred in predictions]

    write_stats(target_dir=run_path, y_true=list(df_test[target_column]), y_pred=predictions)

    return list(predictions)


## Callback to monitor performance
class GetTestPredictionsCallback(TrainerCallback):

    def __init__(self, *args, dict_test_preds, save_path, trainer, test_data, **kwargs):

        super().__init__(*args, **kwargs)

        self.dict_test_preds = dict_test_preds
        self.save_path = save_path
        self.trainer = trainer
        self.test_data=test_data
        self.df_test_stats = pd.DataFrame()

    def on_log(self, args, state, control, logs=None, **kwargs):

        pred = self.trainer.predict(self.test_data)
        predictions = pred.predictions.argmax(axis=1)
        self.dict_test_preds[logs['epoch']] = predictions
        self.df_test_stats = pd.concat([self.df_test_stats, pd.DataFrame(pred.metrics, index=[int(logs['epoch'])])])

    def on_train_end(self, args, state, control, **kwargs):

        self.df_test_stats.to_csv(self.save_path, index_label='epoch')


## Callback to log loss of training and evaluation to file
class WriteCsvCallback(TrainerCallback):

    def __init__(self, *args, csv_train, csv_eval, dict_val_loss, **kwargs):

        super().__init__(*args, **kwargs)

        self.csv_train_path = csv_train
        self.csv_eval_path = csv_eval
        self.df_eval = pd.DataFrame()
        self.df_train_eval = pd.DataFrame()
        self.dict_val_loss = dict_val_loss

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

            if 'eval_loss' in logs:

                self.dict_val_loss[logs['epoch']] = logs['eval_loss']

    def on_train_end(self, args, state, control, **kwargs):

        self.df_eval.to_csv(self.csv_eval_path)
        self.df_train_eval.to_csv(self.csv_train_path)


# Which metrics to compute on evaluation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = calculate_accuracy(labels, preds)
    f1 = calculate_macro_f1(labels, preds)
    return {
      'acc': acc,
      'macro f1': f1,
    }


class Dataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels):

        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):

        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        # item = {k: v[idx] for k, v in self.encodings.items()}
        # item = {k: v[idx].clone().detach() for k, v in self.encodings.items()}

        return item

    def __len__(self):

        return len(self.labels)