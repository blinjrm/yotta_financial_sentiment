"""Contains all configurations for the project.

>>> import src.settings as stg

"""

import logging
import os

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
TRAINING_DATA_DIR = os.path.join(REPO_DIR, "data/training/")
PREDICTION_DATA_DIR = os.path.join(REPO_DIR, "data/prediction/")
OUTPUTS_DIR = os.path.join(REPO_DIR, "data/output/")
APP_DATA_DIR = os.path.join(REPO_DIR, "data/app/")
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
MODEL_NAME = "roberta-base"
SCRAPING_FILENAME = "scraped_headlines.csv"
OUTPUT_FILENAME = "data_with_predictions.csv"

ID2LABEL = {"0": "neutral", "1": "positive", "2": "negative"}
LABEL2ID = {"neutral": 0, "positive": 1, "negative": 2}
NUM_LABEL = len(LABEL2ID)

MODELS = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-large",
    "distilroberta": "distilroberta-base",
    "finbert": "ipuneetrathore/bert-base-cased-finetuned-finBERT",
}

SCRAPING_WEBSITES = ["reuters", "financial_times"]
SCRAPING_START_DATE = "2020-12-10"

PARAMS_REUTERS = {
    "newspaper": "Reuters",
    "early_date": SCRAPING_START_DATE,
    "url": "https://www.reuters.com/news/archive/marketsNews?view=page&page={i}&pageSize=10",
    "html_articles": ["div", "story-content"],
    "html_headline": ["h3", "story-title"],
    "html_date": ["span", "timestamp"],
}

PARAMS_FT = {
    "newspaper": "Financial Times",
    "early_date": SCRAPING_START_DATE,
    "url": "https://www.ft.com/markets?page={i}",
    "html_articles": ["li", "o-teaser-collection__item o-grid-row"],
    "html_headline": ["a", "js-teaser-heading-link"],
    "html_date": ["time", "o-date o-teaser__timestamp"],
}
