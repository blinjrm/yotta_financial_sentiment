"""Module to train the model.

Example
-------
Script could be run with the following command line from the shell :

    $ python src/application/predict_string.py "After a violent crash, the stock doesn't seem to recover"

Script could be run with the following command line from a python interpreter :

    >>> run src/application/predict_string.py "After a violent crash, the stock doesn't seem to recover"

"""

import logging
import os

import fire
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, RobertaTokenizer, pipeline

import src.settings.base as stg
from src.infrastructure.make_dataset import DatasetBuilder


def make_prediction_string(sentence):
    """Use the pretrained model to make evaluate the sentiment of a string.

    Args:
        sentence (string): a prediction will be made for the sentence and printed in the terminal.

    Raises:
        TypeError: an error is raised it the data in not a string.
    """

    stg.enable_logging(log_filename="project_logs.log", logging_level=logging.INFO)

    logging.info("-" * 40)
    logging.info("_________ New prediction ___________\n")

    logging.info("Loading model, tokenizer and config..")
    config = AutoConfig.from_pretrained(os.path.join(stg.MODEL_DIR, f"config_{stg.MODEL_NAME}"))
    tokenizer = RobertaTokenizer.from_pretrained(
        os.path.join(stg.MODEL_DIR, f"tokenizer_{stg.MODEL_NAME}")
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(stg.MODEL_DIR, f"classifier_{stg.MODEL_NAME}"), config=config
    )

    logging.info("Creating classifier..")
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    if isinstance(sentence, str):
        logging.info("Making prediction on a string..")
        preds = classifier(sentence)
        for pred in preds:
            print("\nSentiment analysis:")
            print(f"Label: {pred['label']}, with score: {round(pred['score'], 4)}")
    else:
        raise TypeError("Data must be a single sentence or a Pandas DataFrame.")


if __name__ == "__main__":
    fire.Fire(make_prediction_string)
