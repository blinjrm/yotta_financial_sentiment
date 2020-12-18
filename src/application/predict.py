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
from transformers import pipeline

import src.settings.base as stg
from src.domain.predict_utils import convert_predictions_dataframe, load_data
from src.infrastructure.infra import TrainedModelLoader


def make_prediction(
    filename, model_name=stg.MODEL_NAME, path=stg.PREDICTION_DATA_DIR, headline_col=stg.HEADLINE_COL
):
    """Use the pretrained model to make predictions on the headlines contained in a csv file.
    The results are saved in a .csv fil in the output/ directory.

    Args:
        filename (string): File containing the data - must be of type csv.
        model_name (string, optional): Name of the model to use for predictions. Defaults to distilroberta-base.
        path (string, optional): Path where filename is located. Defaults to ./data/prediction/.
        headline_col (string, optional): Name of the column containing the headlines. "headline".
    """

    stg.enable_logging(log_filename="project_logs.log", logging_level=logging.INFO)

    logging.info("_" * 40)
    logging.info("_________ New prediction ___________\n")

    logging.info("Load model and tokenizer")
    m = TrainedModelLoader(model_name)

    logging.info("Create classifier")
    classifier = pipeline("sentiment-analysis", model=m.model, tokenizer=m.tokenizer)

    logging.info("Load data")
    df, X = load_data(filename, path, headline_col)

    logging.info("Make predictions")
    preds = classifier(X)

    logging.info("Convert predictions to dataframe")
    df_preds = convert_predictions_dataframe(preds)

    logging.info("Concatenate original data with predictions")
    df_with_predictions = pd.concat([df, df_preds], axis=1)

    logging.info("Export results")
    df_with_predictions.to_csv(
        os.path.join(stg.OUTPUTS_DIR, stg.OUTPUT_FILENAME), sep="@", index=False
    )

    print(
        f"\n{df.shape[0]} predictions where successfully saved in the file: \
        {stg.OUTPUT_FILENAME}, \nin the directory: {stg.OUTPUTS_DIR}"
    )


if __name__ == "__main__":
    fire.Fire(make_prediction)
