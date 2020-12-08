"""Module to evaluate the sentiment for a single sentence.

Example
-------
Script could be run with the following command line from the shell :

    $ python src/application/predict_string.py "After a violent crash, the stock doesn't seem to recover"

Script could be run with the following command line from a python interpreter :

    >>> run src/application/predict_string.py "After a violent crash, the stock doesn't seem to recover"

Use the flag --help to show usage information

"""

import logging

import fire

# import pandas as pd
from transformers import pipeline

import src.settings.base as stg
from src.infrastructure.infra import TrainedModelLoader


def make_prediction_string(sentence, model_name=stg.MODEL_NAME):
    """Use the pretrained model to make evaluate the sentiment of a string.

    Args:
        sentence (string): a prediction will be made for the sentence and printed in the terminal.
        model_name (string, optional): Name of the model to use for predictions. Defaults to distilroberta-base.

    Raises:
        TypeError: an error is raised it the data in not a string.
    """

    stg.enable_logging(log_filename="project_logs.log", logging_level=logging.INFO)

    logging.info("_" * 40)
    logging.info("_____ New prediction on a sentence _____\n")

    logging.info("Load model and tokenizer")
    m = TrainedModelLoader(model_name)

    logging.info("Create classifier")
    classifier = pipeline("sentiment-analysis", model=m.model, tokenizer=m.tokenizer)

    if isinstance(sentence, str):
        logging.info("Make prediction on a string")
        pred = classifier(sentence)[0]

        print("\nSentiment analysis:")
        print(f"Label: {pred['label']}, with score: {round(pred['score'], 4)}")

        return pred

    else:
        logging.error("Typerror")
        raise TypeError("Data must be a single sentence.")


if __name__ == "__main__":
    prediction = fire.Fire(make_prediction_string)


# zero_shot_classifier = pipeline("zero-shot-classification")
# labels = list(stg.LABEL2ID.keys())
# results = zero_shot_classifier(sentece, labels)
# SCORES = results["scores"]
# CLASSES = results["labels"]
# BEST_INDEX = argmax(SCORES)
# predicted_class = CLASSES[BEST_INDEX]