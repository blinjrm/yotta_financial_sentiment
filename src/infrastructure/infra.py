"""This module povides classes to load data and models. 

Classes
-------
DatasetBuilder
ModelLoader
TrainedModelLoader

"""


import logging
import os
import re
import sys
from datetime import date

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dateutil.parser import parse
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

import src.settings.base as stg


class DatasetBuilder:
    """Creates dataet from CSV file.

    Args:
        filename (string): the name of the file to load - must be of type .csv.
        path (string): the path to find the file

    Raises:
        FileExistsError: The given filename does not end with .csv.
        FileNotFoundError: The file does not exist in the given directory.

    Returns:
        pandas.DataFrame: DataFrame containing the data
    """

    def __init__(self, filename, path):
        self.data = self._load_data_from_csv(filename, path)

    def _load_data_from_csv(self, filename, path):
        if self._check_file_extension(filename):
            df = self._open_file(filename, path)
            return df

    @staticmethod
    def _check_file_extension(filename):
        logging.info("Confirm file extension is .csv")
        if filename.endswith(".csv"):
            return True
        else:
            logging.error("Extension must be .csv")
            raise FileExistsError("Extension must be .csv")

    @staticmethod
    def _open_file(filename, path):
        logging.info("Load data")
        try:
            df = pd.read_csv("".join((path, filename)), delimiter="@")
            return df
        except FileNotFoundError as error:
            logging.error("FileNotFoundError")
            raise FileNotFoundError(f"Error in SalesDataset initialization - {error}")


class ModelLoader:
    """Initialize a new model and tokenizer from huggingface.co"""

    def __init__(self, model_name):
        self.model_name = model_name
        self.model, self.tokenizer, self.config = self._create_model()

    def _create_model(self):
        config = AutoConfig.from_pretrained(
            self.model_name, num_labels=stg.NUM_LABEL, id2label=stg.ID2LABEL, label2id=stg.LABEL2ID
        )
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=config)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        return model, tokenizer, config

    def save_model(self, trainer):
        trainer.save_model(os.path.join(stg.MODEL_DIR, self.model_name))
        self.tokenizer.save_pretrained(os.path.join(stg.MODEL_DIR, self.model_name))
        self.config.save_pretrained(os.path.join(stg.MODEL_DIR, self.model_name))


class TrainedModelLoader:
    """Load a fine-tuned model and tokenizer for classification"""

    def __init__(self, model_name):
        self.model_name = model_name
        self.model, self.tokenizer = self._load_model()

    def _load_model(self):
        config = AutoConfig.from_pretrained(os.path.join(stg.MODEL_DIR, stg.MODEL_NAME))
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(stg.MODEL_DIR, stg.MODEL_NAME))
        model = AutoModelForSequenceClassification.from_pretrained(
            os.path.join(stg.MODEL_DIR, stg.MODEL_NAME), config=config
        )

        return model, tokenizer


def clean_headline_date(headline, date):
    headline_clean = re.sub(r"[-()\"#/@;:<>{}=~|.?,]", "", headline).strip()
    date_clean = parse(date).strftime("%Y-%m-%d")

    return headline_clean, date_clean


def get_headlines_reuters(early_date):
    """Scrapes the headlines for financial news from reuters.com

    Args:
        early_date (string): Earliest date to crape headlines from.

    Returns:
        df: DataFrame containing the headlines and publication dates.
    """

    headlines = {"headline": [], "date": []}
    date_clean = date.today().strftime("%Y-%m-%d")
    i = 1

    while date_clean >= early_date:

        print(f"Current page:{i}, current date: {date_clean}", end="\r")

        url = f"https://www.reuters.com/news/archive/marketsNews?view=page&page={i}&pageSize=10"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        try:
            articles = soup.findAll("div", class_="story-content")
        except:
            break

        for article in articles:
            try:
                headline = article.find("h3", class_="story-title").text
                date_ = article.find("span", class_="timestamp").text

                headline_clean, date_clean = clean_headline_date(headline, date_)

                if date_clean < early_date:
                    break

                headlines["headline"].append(headline_clean)
                headlines["date"].append(date_clean)
            except:
                pass

        i += 1

    df = pd.DataFrame(headlines)
    df["source"] = "Reuters"
    return df


def get_headlines_ft(early_date):
    """Scrapes the headlines for financial news from ft.com

    Args:
        early_date (string): Earliest date to crape headlines from.

    Returns:
        df: DataFrame containing the headlines and publication dates.
    """

    headlines = {"headline": [], "date": []}
    date_clean = date.today().strftime("%Y-%m-%d")
    i = 1

    while date_clean >= early_date:

        # print(f"Getting headlines from Financial Times - page {i}", end="\r")

        url = f"https://www.ft.com/markets?page={i}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        try:
            articles = soup.findAll("li", class_="o-teaser-collection__item o-grid-row")

        except:
            break

        for article in articles:
            try:
                headline = article.find("a", class_="js-teaser-heading-link").text
                date_ = article.find("time", class_="o-date o-teaser__timestamp").text

                print(f"Current page:{i}, current date: {date_clean}", end="\r")

                headline_clean, date_clean = clean_headline_date(headline, date_)

                if date_clean < early_date:
                    break

                headlines["headline"].append(headline_clean)
                headlines["date"].append(date_clean)
            except:
                pass

        i += 1

    df = pd.DataFrame(headlines)
    df["source"] = "Financial Times"
    return df
