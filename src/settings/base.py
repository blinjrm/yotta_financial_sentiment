"""
Contains all configurations for the project.

>>> import src.settings as stg

"""

import logging
import os

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
TRAINING_DATA_DIR = os.path.join(REPO_DIR, "data/training/")
PREDICTION_DATA_DIR = os.path.join(REPO_DIR, "data/prediction/")
OUTPUTS_DIR = os.path.join(REPO_DIR, "outputs")
LOGS_DIR = os.path.join(REPO_DIR, "logs")
MODEL_DIR = os.path.join(REPO_DIR, "model")


def enable_logging(log_filename, logging_level=logging.DEBUG):
    """Set loggings parameters.


    Args:
        log_filename (str): name of the file where the logs are written
        logging_level (logging.level, optional): Min level to log. Defaults to logging.DEBUG.
    """

    with open(os.path.join(LOGS_DIR, log_filename), "a") as file:
        file.write("\n")
        file.write("\n")

    LOGGING_FORMAT = "[%(asctime)s][%(levelname)s][%(module)s] - %(message)s"
    LOGGING_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        format=LOGGING_FORMAT,
        datefmt=LOGGING_DATE_FORMAT,
        level=logging_level,
        filename=os.path.join(LOGS_DIR, log_filename),
    )


HEADLINE_COL = "headline"
SENTIMENT_COL = "sentiment"
MODEL_NAME = "distilroberta-base"
SAVED_MODEL_FILE = os.path.join(MODEL_DIR, "model.pkl")

ID2LABEL = {"0": "neutral", "1": "positive", "2": "negative"}
LABEL2ID = {"neutral": 0, "positive": 1, "negative": 2}
