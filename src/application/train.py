"""Module to train the model.

Example
-------
Script could be run with the following command line from the shell :

    $ python src/application/train.py training_data.csv
    $ python src/application/train.py training_data.csv --epochs=5

Script could be run with the following command line from a python interpreter :

    >>> run src/application/train.py training_data.csv
    >>> run src/application/train.py training_data.csv --epochs=5

Use the flag --help to show usage information

"""

import logging

import fire
from transformers import Trainer, TrainingArguments

import src.settings.base as stg
from src.domain.train_utils import compute_metrics, create_dataset
from src.infrastructure.infra import DatasetBuilder, ModelLoader


def train_model(
    filename,
    model_name=stg.MODEL_NAME,
    path=stg.TRAINING_DATA_DIR,
    headline_col=stg.HEADLINE_COL,
    sentiment_col=stg.SENTIMENT_COL,
    epochs=5,
):
    """Create a model based on model_name and fine-tunes it with the data from filename

    Args:
        filename (string): File containing the data - must be of type csv.
        path (string, optional): Path where filename is located. Defaults to ./data/training/.
        headline_col (string, optional): Name of the column containing the headlines. Defaults to "headline".
        sentiment_col (string, optional): Name of the column containing the target (sentiment). Defaults to "sentiment".
        epochs (int, optional): Number of epochs for fine-tuning. Defaults to 5.
    """

    stg.enable_logging(log_filename="project_logs.log", logging_level=logging.INFO)

    logging.info("_" * 40)
    logging.info("_________ New training ___________\n")

    logging.info("Load data")
    df = DatasetBuilder(filename, path).data

    X = df[headline_col].astype(str).tolist()
    y = df[sentiment_col].map(stg.LABEL2ID).tolist()

    logging.info("Load model")
    m = ModelLoader(model_name)

    logging.info("Create dataset")
    train_dataset, test_dataset = create_dataset(X, y, m.tokenizer)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=m.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    logging.info("Train model")
    trainer.train()

    logging.info("Evaluate model")
    trainer.evaluate()

    logging.info("Export model")
    m.save_model(trainer)


if __name__ == "__main__":
    fire.Fire(train_model)
