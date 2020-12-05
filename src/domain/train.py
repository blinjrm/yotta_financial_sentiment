"""Module to train the model.

Example
-------
Script could be run with the following command line from the shell :

    $ python src/domain/train.py -f data.csv

Script could be run with the following command line from a python interpreter :

    >>> run src/domain/train.py -f data.csv

Attributes
----------
PARSER: argparse.ArgumentParser

"""

import argparse
import logging

import fire
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

import src.settings.base as stg
from src.infrastructure.make_dataset import DatasetBuilder


class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def create_dataset(X, y, tokenizer):
    train_texts, test_texts, train_labels, test_labels = train_test_split(X, y, test_size=0.15)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = SentimentDataset(train_encodings, train_labels)
    test_dataset = SentimentDataset(test_encodings, test_labels)

    return train_dataset, test_dataset


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def train_model(
    filename,
    path=stg.TRAINING_DATA_DIR,
    headline_col=stg.HEADLINE_COL,
    sentiment_col=stg.SENTIMENT_COL,
    model_name=stg.MODEL_NAME,
    epochs=5,
):
    """[summary]

    Args:
        filename ([type]): [description]
        path ([type], optional): [description]. Defaults to stg.TRAINING_DATA_DIR.
        headline_col ([type], optional): [description]. Defaults to stg.HEADLINE_COL.
        sentiment_col ([type], optional): [description]. Defaults to stg.SENTIMENT_COL.
        model_name ([type], optional): [description]. Defaults to stg.MODEL_NAME.
        epochs (int, optional): [description]. Defaults to 5.
    """

    stg.enable_logging(log_filename="project_logs.log", logging_level=logging.INFO)

    logging.info("_" * 20)
    logging.info("_________ Launch new training ___________\n")

    # try:
    #     df = pd.read_csv(df_path)
    # except FileNotFoundError:
    #     raise

    logging.info("Loading data..")
    df = DatasetBuilder(filename, path).data

    X = df["headline"].astype(str).tolist()
    y = df["sentiment"].map({"neutral": 0, "positive": 1, "negative": 2}).tolist()

    config = AutoConfig.from_pretrained(model_name, num_labels=3)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset, test_dataset = create_dataset(X, y, tokenizer)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    logging.info("Training model..")
    print("Training model...")
    trainer.train()
    trainer.evaluate()

    trainer.save_model(f"./model/classifier_{model_name}")
    tokenizer.save_pretrained(f"./model/tokenizer_{model_name}")


if __name__ == "__main__":
    fire.Fire(train_model)
