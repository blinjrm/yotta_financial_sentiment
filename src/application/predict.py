"""Module to train the model.

Example
-------
Script could be run with the following command line from the shell :

    $ python src/application/predict.py

Script could be run with the following command line from a python interpreter :

    >>> run src/application/predict.py

"""

import logging
import os

import fire
import pandas as pd
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    RobertaTokenizer,  # AutoTokenizer
    pipeline,
)

import src.settings.base as stg
from src.infrastructure.make_dataset import DatasetBuilder


config = AutoConfig.from_pretrained(os.path.join(stg.MODEL_DIR, f"config_{stg.MODEL_NAME}"))
tokenizer = RobertaTokenizer.from_pretrained(
    os.path.join(stg.MODEL_DIR, f"tokenizer_{stg.MODEL_NAME}")
)
model = AutoModelForSequenceClassification.from_pretrained(
    os.path.join(stg.MODEL_DIR, f"classifier_{stg.MODEL_NAME}"), config=config
)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

test_sentence = "After a violent crash, the stock doesn't seem to recover"

print(classifier(test_sentence))
