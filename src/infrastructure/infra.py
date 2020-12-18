"""This module povides classes to load data and models. 

Classes
-------
DatasetBuilder
ModelLoader
TrainedModelLoader

Functions
-------
tensorflowGPU
torchGPU

"""


import logging
import os
import re
from datetime import date

import tensorflow as tf
import torch

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dateutil.parser import parse
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

import src.settings.base as stg


class DatasetBuilder:
    """Creates dataset from CSV file.

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


class HeadlinesReuters:
    """Web scraping from reuters.com"""

    def __init__(self, edition, domain, n_pages):
        self.edition = edition
        self.domain = domain
        self.n_pages = n_pages
        self.data = self._get_data()

    def _get_data(self):
        print(f"Getting headlines for {self.domain}")

        headlines = {"headline": [], "date": [], "edition": []}

        for i in tqdm(range(1, self.n_pages + 1)):

            url = f"{self.domain}/news/archive/marketsNews?view=page&page={i}&pageSize=10"
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            articles = soup.findAll("div", class_="story-content")

            for article in articles:
                try:
                    headline = article.find("h3", class_="story-title").text
                    date = article.find("span", class_="timestamp").text

                    headline_clean = re.sub(r"[-()\"#/@;:<>{}=~|.?,]", "", headline).strip()
                    date_clean = parse(date).strftime("%Y-%m-%d")

                    headlines["headline"].append(headline_clean)
                    headlines["date"].append(date_clean)
                    headlines["edition"].append(self.edition)

                except:
                    pass

        return pd.DataFrame(headlines)


def tensorflowGPU():
    # Get the GPU device name.
    device_name = tf.test.gpu_device_name()

    # The device name should look like the following:
    if device_name == "/device:GPU:0":
        print("Found GPU at: {}".format(device_name))
    else:
        raise SystemError("GPU device not found")  #


def torchGPU():

    # If there's a GPU available...
    if torch.cuda.is_available():

        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")

        print("There are %d GPU(s) available." % torch.cuda.device_count())

        print("We will use the GPU:", torch.cuda.get_device_name(0))

    # If not...
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
