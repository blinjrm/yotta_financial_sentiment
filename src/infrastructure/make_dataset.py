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

    def _check_file_extension(self, filename):
        logging.info("Confirm file extension is .csv")
        if filename.endswith(".csv"):
            return True
        else:
            logging.error("Extension must be .csv")
            raise FileExistsError("Extension must be .csv")

    def _open_file(self, filename, path):
        logging.info("Load data")
        try:
            df = pd.read_csv("".join((path, filename)), delimiter="@")
            return df
        except FileNotFoundError as error:
            logging.error("FileNotFoundError")
            raise FileNotFoundError(f"Error in SalesDataset initialization - {error}")
