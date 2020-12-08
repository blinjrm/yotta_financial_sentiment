"""This module povides classes to load data and models. 

Classes
-------
DatasetBuilder
ModelLoader
TrainedModelLoader

"""


import logging
import os

import pandas as pd
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
    """[summary]"""

    def __init__(self, model_name):
        self.model_name = model_name
        self.model, self.tokenizer, self.config = self._load_model()

    def _load_model(self):
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
    """[summary]"""

    def __init__(self, model_name):
        self.model_name = model_name
        self.model, self.tokenizer = self._load_model()

    def _load_model(self):
        config = AutoConfig.from_pretrained(os.path.join(stg.MODEL_DIR, stg.MODEL_NAME))
        # tokenizer = RobertaTokenizer.from_pretrained(os.path.join(stg.MODEL_DIR, stg.MODEL_NAME))
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(stg.MODEL_DIR, stg.MODEL_NAME))
        model = AutoModelForSequenceClassification.from_pretrained(
            os.path.join(stg.MODEL_DIR, stg.MODEL_NAME), config=config
        )

        return model, tokenizer
