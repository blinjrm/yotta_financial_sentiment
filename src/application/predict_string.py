"""Module to evaluate the sentiment for a single sentence.

Example
-------
Script could be run with the following command line from the shell :

    $ python src/application/predict_string.py "After a violent crash, the stock doesn't seem to recover"

Script could be run with the following command line from a python interpreter :

    >>> run src/application/predict_string.py "After a violent crash, the stock doesn't seem to recover"

Use the flag --help to show usage information

"""

import logging

import fire
import numpy as np
from transformers import pipeline

import src.settings.base as stg
from src.domain.predict_utils import list_trained_models
from src.infrastructure.infra import TrainedModelLoader


def make_prediction_string(sentence, model_name=stg.MODEL_NAME):
    """Use the pretrained model to make evaluate the sentiment of a string.

    Args:
        sentence (string): a prediction will be made for the sentence and printed in the terminal.
        model_name (string, optional): Name of the model to use for predictions. Defaults to distilroberta-base.

    Raises:
        TypeError: an error is raised it the data in not a string.
        FileNotFoundError : an error is raised if the selected model is does not exit in the model directory/ or a not a zero-shot classifier.

    Returns:
        Pred (dict): label and score given by the model.
    """

    stg.enable_logging(log_filename="project_logs.log", logging_level=logging.INFO)

    logging.info("_" * 40)
    logging.info("_____ New prediction on a sentence _____\n")

    if not isinstance(sentence, str):
        logging.error("Typerror")
        raise TypeError("Data must be a single sentence.")

    else:
        trained_models = list_trained_models()

        if model_name in trained_models:

            logging.info("Load model and tokenizer")
            m = TrainedModelLoader(model_name)

            logging.info("Create classifier")
            classifier = pipeline("sentiment-analysis", model=m.model, tokenizer=m.tokenizer)

            logging.info("Get sentiment")
            pred = classifier(sentence)[0]

        elif model_name == "zero-shot-classifier":

            logging.info("Create zero-shot classifier")
            zero_shot_classifier = pipeline("zero-shot-classification")

            logging.info("Get sentiment")
            labels = list(stg.LABEL2ID.keys())
            results = zero_shot_classifier(sentence, labels)

            results_scores = results["scores"]
            results_labels = results["labels"]
            best_index = np.argmax(results_scores)

            pred = {"label": results_labels[best_index], "score": results_scores[best_index]}

        else:
            trained_models.append("zero-shot-classifier")
            logging.error(f"Model {model_name} not found")
            raise FileNotFoundError(
                f"The model {model_name} was not found in the model/ directory. The available models are: {trained_models}"
            )

    print("\nSentiment analysis:")
    print(f"Label: {pred['label']}, with score: {round(pred['score'], 4)}")

    return pred


if __name__ == "__main__":
    prediction = fire.Fire(make_prediction_string)
