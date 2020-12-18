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

PROJECT_DESCRIPTION = """ 
2020 felt like a rollercoaster 🎢, with the global pandemic causing a sudden economic downturn 📉, which didn't prevent the stock market to rise to new heights ⛰️.
---
We were curious about how financial newspapers have been dealing with this dichotomy, and decided to conduct an analysis of the sentiment of the headlines they printed throughout the year. 

By leveraging recent advancements of *transfer learning* in the field of natural language processing, we can take a model pretrained on a large, generic language corpus (think Wikipedia), and fine-tune it on a domain-specific task: in our case classification of financial sentences. 

For this project, we focused on the United States, being the first economy in the world and home to the largest financial markets, while at the same time being the most affected by the pandemic in terms of total cases and deaths. 

This website presents in a **dashboard** the results of our analysis, regarding:
* The evolution of the sentiment expressed in financial headlines throughout 2020
* The relation between that sentiment and:
    * The stock market
    * The development of the COVID-19 pandemic

It also provides you with the opportunity to use the models we fine-tuned on the **headlines** of your choice.  

Finaly, we compare provide some insights into the **performance** of several language models. 

*👈 Use the menu on the left to access these screens.*
"""
