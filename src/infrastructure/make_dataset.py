"""This module creates and cleans the dataset from a flat file.

Classes
-------
DatasetBuilder

"""


import logging

import numpy as np
import pandas as pd

import src.settings.base as stg


class DatasetBuilder:
    """Creates dataet from CSV file.

    Attributes
    ----------
    data: dataset in a Pandas dataframe

    """

    def __init__(self, filename, path):
        self.data = self._load_data_from_csv(filename, path)

    def _load_data_from_csv(self, filename, path):
        if self._check_file_extension(filename):
            df = self._open_file(filename, path)
            return df

    def _check_file_extension(self, filename):
        logging.info("Confirm file extension is .csv ..")
        if filename.endswith(".csv"):
            logging.info(".. Done \n")
            return True
        else:
            logging.info(".. ERROR: Extension must be .csv")
            raise FileExistsError("Extension must be .csv")

    def _open_file(self, filename, path):
        logging.info("Load data ..")
        try:
            df = pd.read_csv("".join((path, filename)), delimiter="@")
            logging.info(".. Done \n")
            return df
        except FileNotFoundError as error:
            logging.info(".. FileNotFoundError")
            raise FileNotFoundError(f"Error in SalesDataset initialization - {error}")
