"""Module to scrape headlines from a website.

Example
-------
Script could be run with the following command line from the shell :

    $ python src/application/predict_string.py "After a violent crash, the stock doesn't seem to recover"

Script could be run with the following command line from a python interpreter :

    >>> run src/application/scraping.py

Use the flag --help to show usage information

"""

import logging
import os

import fire
import pandas as pd

import src.settings.base as stg
from src.infrastructure.infra import HeadlinesReuters


def get_headlines(website="reuters", n_pages=10):
    """

    Args:
        website (str or list of str, optional): Website to srape headlines from. Defaults to "reuters".
        n_pages (int, optional): Number of pages to scrape for each website. Defaults to 10.
    """

    stg.enable_logging(log_filename="project_logs.log", logging_level=logging.INFO)

    logging.info("_" * 40)
    logging.info("_____ Scraping headlines _____\n")

    df_headlines = pd.DataFrame()

    if "reuters" in website:

        EDITIONS = {"us": "https://www.reuters.com", "uk": "https://uk.reuters.com"}

        for edition, domain in EDITIONS.items():
            scrapper = HeadlinesReuters(edition, domain, n_pages)
            df_headlines = pd.concat([df_headlines, scrapper.data])

    logging.info("Export results")
    df_headlines.to_csv(
        os.path.join(stg.PREDICTION_DATA_DIR, stg.SCRAPING_FILENAME), sep="@", index=False
    )

    print(
        f"\n{df_headlines.shape[0]} headlines where successfully saved in the file: \
        {stg.SCRAPING_FILENAME}, \nin the directory: {stg.PREDICTION_DATA_DIR}"
    )


if __name__ == "__main__":
    fire.Fire(get_headlines)
