"""Module to train the model.

Example
-------
Script could be run with the following command line from the shell :

    $ python src/domain/train.py training_data.csv
    $ python src/domain/train.py training_data.csv --epochs=5

Script could be run with the following command line from a python interpreter :

    >>> run src/domain/train.py training_data.csv
    >>> run src/domain/train.py training_data.csv --epochs=5

"""

import logging
import os

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
    """Dataset object used by the pytorch model"""

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
    """Creates a dataset to be passed to the model

    Args:
        X (list of strings): list of headlines
        y (list of ints): list of sentiment (0='neutral', 1='positive', 2='negative')
        tokenizer (huggingface tokenizer): the tokenizer to be used to tokenize the data

    Returns:
        Dataset: data ready to be used in the pytorch model
    """
    train_texts, test_texts, train_labels, test_labels = train_test_split(X, y, test_size=0.15)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = SentimentDataset(train_encodings, train_labels)
    test_dataset = SentimentDataset(test_encodings, test_labels)

    return train_dataset, test_dataset


def compute_metrics(pred):
    """Compute the metrics for the model, when calling model.evaluate()

    Args:
        pred (int): prediction for the sentiment of a headline

    Returns:
        dict: dictionnary containing the chosen metrics
    """
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
    epochs=5,
):
    """Create a model based on model_name and fine-tunes it with the data from filename

    Args:
        filename (string): File containing the data - must be of type csv.
        path (string, optional): Path where filename is located. Defaults to stg.TRAINING_DATA_DIR.
        headline_col (string, optional): Name of the columns containing the headlines. Defaults to stg.HEADLINE_COL.
        sentiment_col (string, optional): Name of the columns containing the target (sentiment). Defaults to stg.SENTIMENT_COL.
        epochs (int, optional): Number of epochs for fine-tuning. Defaults to 5.
    """

    stg.enable_logging(log_filename="project_logs.log", logging_level=logging.INFO)

    logging.info("40" * 20)
    logging.info("_________ Launch new training ___________\n")

    logging.info("Loading data..")
    df = DatasetBuilder(filename, path).data

    X = df[headline_col].astype(str).tolist()
    y = df[sentiment_col].map({"neutral": 0, "positive": 1, "negative": 2}).tolist()

    config = AutoConfig.from_pretrained(
        stg.MODEL_NAME, num_labels=3, id2label=stg.ID2LABEL, label2id=stg.LABEL2ID
    )
    model = AutoModelForSequenceClassification.from_pretrained(stg.MODEL_NAME, config=config)
    tokenizer = AutoTokenizer.from_pretrained(stg.MODEL_NAME)

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
    trainer.train()
    trainer.evaluate()

    logging.info("Exporting model..")
    trainer.save_model(os.path.join(stg.MODEL_DIR, f"classifier_{stg.MODEL_NAME}"))
    tokenizer.save_pretrained(os.path.join(stg.MODEL_DIR, f"tokenizer_{stg.MODEL_NAME}"))
    config.save_pretrained(os.path.join(stg.MODEL_DIR, f"config_{stg.MODEL_NAME}"))


if __name__ == "__main__":
    fire.Fire(train_model)
