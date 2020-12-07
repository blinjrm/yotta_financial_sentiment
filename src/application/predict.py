"""Module to evaluate the sentiment for every item in a .csv file.

Example
-------
Script could be run with the following command line from the shell :

    $ python src/application/predict.py data_4_prediction.csv

Script could be run with the following command line from a python interpreter :

    >>> run src/application/predict.py data_4_prediction.csv

Use the flag --help to show usage information

"""

import logging
import os

import fire
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, RobertaTokenizer, pipeline

import src.settings.base as stg
from src.infrastructure.make_dataset import DatasetBuilder


def load_pretrained_model():
    """[summary]

    Returns:
        [type]: [description]
    """

    config = AutoConfig.from_pretrained(os.path.join(stg.MODEL_DIR, f"config_{stg.MODEL_NAME}"))
    tokenizer = RobertaTokenizer.from_pretrained(
        os.path.join(stg.MODEL_DIR, f"tokenizer_{stg.MODEL_NAME}")
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(stg.MODEL_DIR, f"classifier_{stg.MODEL_NAME}"), config=config
    )

    return model, tokenizer


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


def make_prediction(filename, path=stg.PREDICTION_DATA_DIR, headline_col=stg.HEADLINE_COL):
    """Use the pretrained model to make predictions on the headlines contained in a csv file.
    The results are saved in a .csv fil in the output/ directory.

    Args:
        filename (string): File containing the data - must be of type csv.
        path (string, optional): Path where filename is located. Defaults to ./data/prediction/.
        headline_col (string, optional): Name of the column containing the headlines. "headline".
    """

    stg.enable_logging(log_filename="project_logs.log", logging_level=logging.INFO)

    logging.info("_" * 40)
    logging.info("_________ New prediction ___________\n")

    logging.info("Loading model, tokenizer and config")
    model, tokenizer = load_pretrained_model()

    logging.info("Creating classifier")
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    logging.info("Loading data")
    df, X = load_data(filename, path, headline_col)

    logging.info("Making predictions")
    preds = classifier(X)

    logging.info("Converting predictions to dataframe")
    df_preds = convert_predictions_dataframe(preds)

    logging.info("Concatenating original data with predictions")
    df_with_predictions = pd.concat([df, df_preds], axis=1)

    logging.info("Exporting results")
    df_with_predictions.to_csv(os.path.join(stg.OUTPUTS_DIR, stg.OUTPUT_FILENAME), index=False)

    print(
        f"\n{df.shape[0]} predictions where successfully saved in the file: {stg.OUTPUT_FILENAME}, \nin the directory: {stg.OUTPUTS_DIR}"
    )


if __name__ == "__main__":
    fire.Fire(make_prediction)
