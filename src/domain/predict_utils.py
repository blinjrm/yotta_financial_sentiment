"""Utilities to use the model for predictions.

"""

import logging

import pandas as pd

from src.infrastructure.infra import DatasetBuilder


def load_data(filename, path, headline_col):
    """Loads the data from a csv and prepare the headlines for sentiment analysis.

    Args:
        filename (string): File containing the data - must be of type csv.
        path (string): Path where filename is located. Defaults to ./data/prediction/.
        headline_col (string): Name of the column containing the headlines. "headline".

    Raises:
        TypeError: raises an error is filename is not a string.
        KeyError: raises an error if the column 'headline_col' is not find in the file 'filname'.

    Returns:
        df (pandas DataFrame): DataFrame containing the data from the .csv file
        x (list of string): list containing the headline to pass to the model
    """

    if not isinstance(filename, str):
        logging.error("Typerror")
        raise TypeError("Data must be the name of a file.")

    df = DatasetBuilder(filename, path).data

    try:
        X = df[headline_col].astype(str).tolist()
    except:
        raise KeyError(f"Column {headline_col} was not found in the file {filename}")

    return df, X


def convert_predictions_dataframe(preds):
    """Convert the list of predictions in a DataFrame.

    Args:
        preds (list): list of dictionaries containing the predicted labels and scores.

    Returns:
        df_preds (pandas DataFrame): DataFrame containing the labels and score for each headline.
    """

    predictions = {"label": [], "score": []}

    for pred in preds:
        predictions["label"].append(pred["label"])
        predictions["score"].append(pred["score"])

    df_preds = pd.DataFrame(predictions)

    return df_preds