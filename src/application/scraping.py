"""Module to scrape headlines from a website.
The website available for craping are Reuters and Financial Times

Example
-------
Script could be run with the following command line from the shell :

    $ python src/application/scraping.py
    $ python src/application/scraping.py --early_date="2020-11-01"

Script could be run with the following command line from a python interpreter :

    >>> run src/application/scraping.py
    >>> run src/application/scraping.py --early_date="2020-11-01"

Use the flag --help to show usage information

"""

import logging
import os

import fire
import pandas as pd

import src.settings.base as stg
from src.infrastructure.infra import WebScraper


def get_headlines(website=stg.SCRAPING_WEBSITES, early_date=stg.SCRAPING_START_DATE):
    """

    Args:
        website (str or list of str, optional): Website to srape headlines from. Defaults to ["reuters", "financial_times"].
        early_date (str, optional): Earliest date to crape headlines from.
    """

    stg.enable_logging(log_filename="project_logs.log", logging_level=logging.INFO)

    logging.info("_" * 40)
    logging.info("_____ Scraping headlines _____\n")

    df_headlines = pd.DataFrame()

    if not (website == stg.SCRAPING_WEBSITES) and not (website in stg.SCRAPING_WEBSITES):
        print(f"The site '{website}' is not available for scrapping.")
        print(f"Please choose at least one site frome {stg.SCRAPING_WEBSITES}.")

    else:

        if "reuters" in website:
            logging.info("Scrape headlines from Reuters")
            stg.PARAMS_REUTERS["early_date"] = early_date
            df_headlines_reuters = WebScraper(**stg.PARAMS_REUTERS).get_headlines()
            df_headlines = pd.concat(
                [df_headlines, df_headlines_reuters], axis=0, ignore_index=True
            )

        if "financial_times" in website:
            logging.info("Scrape headlines from Financial Times")
            stg.PARAMS_FT["early_date"] = early_date
            df_headlines_reuters = WebScraper(**stg.PARAMS_FT).get_headlines()
            df_headlines = pd.concat(
                [df_headlines, df_headlines_reuters], axis=0, ignore_index=True
            )

        logging.info("Export results")
        df_headlines.to_csv(
            os.path.join(stg.PREDICTION_DATA_DIR, stg.SCRAPING_FILENAME), sep="@", index=False
        )

        print(
            f"\n\n{df_headlines.shape[0]} headlines where successfully saved in the file: {stg.SCRAPING_FILENAME}, \
            \nin the directory: {stg.PREDICTION_DATA_DIR}"
        )


if __name__ == "__main__":
    fire.Fire(get_headlines)
