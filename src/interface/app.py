import datetime
import logging
import os

import pandas as pd
import streamlit as st

import src.settings.base as stg
from src.application.predict_string import make_prediction_string
from src.domain.app_utils import (
    latest_news_widget,
    newspapers_plot,
    raw_data_plot,
    tendency_heatmap,
    tendency_plot,
)
from src.domain.predict_utils import list_trained_models
from src.infrastructure.infra import WebScraper


def project_description():
    """Streamlit widget to show the project description on the home page."""

    st.title("Sentiment analysis of financial headlines")
    st.text("")
    st.markdown(stg.PROJECT_DESCRIPTION)


def latest_headlines():
    """Streamlit widget to show the latest headline from Reuters and Financial Times
    with sentiment analysis.
    """
    st.title("Latest headlines")
    st.text("")

    today = str(datetime.date.today())

    stg.PARAMS_REUTERS["early_date"] = today
    latest_news_widget(stg.PARAMS_REUTERS)
    st.markdown("---")

    stg.PARAMS_FT["early_date"] = today
    latest_news_widget(stg.PARAMS_FT)
    st.markdown("---")


def single_sentence():
    """Streamlit widget to show the sentiment analysis of a headline typed by the user."""

    st.title("Sentiment analysis on a headline")
    st.text("")

    trained_models = list_trained_models()
    trained_models.append("zero-shot-classifier")

    model = st.sidebar.radio("Choose a model:", (trained_models))
    st.sidebar.markdown("---")

    sentence = st.text_input(label="Type your sentence here:")

    make_pred = st.button("Start analysis")

    if make_pred:
        with st.spinner("Analyzing..."):
            prediction = make_prediction_string(sentence, model_name=model)

        st.text("\nSentiment analysis:")
        st.text(f"Label: {prediction['label']}, with score: {round(prediction['score'], 4)}")

    # TODO: add model description


def dashboard():
    """Streamlit widget showing a dashboard for the project, with the following pages:
    - Tendencies: graph and correlation heatmap of fiancial sentiement, stock market and new covid cases
    - Newspapers: graph showing the evolution of financilal sentiment for each newspaper
    - Raw data: insight in the raw data (scraped headlines) used for the project
    """

    st.title("Financial sentiment in the USA in 2020")
    st.text("")

    st.sidebar.text("Choose display options:")
    display_option = st.sidebar.radio(label="", options=["Tendencies", "Newspapers", "Raw data"])

    st.sidebar.markdown("---")

    if display_option == "Tendencies":
        st.text("")
        st.header(
            "Evolution of weekly financial sentiment, S&P500 index and trading volume (normalized)"
        )
        fig = tendency_plot()
        st.pyplot(fig)

        st.text("")
        st.header("Correlation between weekly financial sentiment, S&P500 index and trading volume")
        heatmap = tendency_heatmap()
        st.pyplot(heatmap)

    elif display_option == "Newspapers":

        col1, _, col2 = st.beta_columns([2, 1, 4])
        start_date = str(col1.date_input("Start date", datetime.date(2020, 3, 1)))
        end_date = str(col1.date_input("End date", datetime.date(2020, 11, 30)))

        newspaper = col2.select_slider(
            label="Which newspaper do you want to include?",
            options=["Reuters", "Both", "Financial Times"],
            value=(("Both")),
        )

        smooth = col2.select_slider(
            label="Granularity", options=["Day", "Week", "Month"], value=(("Week"))
        )
        st.text("")

        fig = newspapers_plot(start_date, end_date, newspaper, smooth)
        st.pyplot(fig)

    elif display_option == "Raw data":

        df, fig = raw_data_plot()

        st.text("")
        st.text(df)
        st.pyplot(fig)


def main():
    """Homepage for the Streamlit app, redirects to appropriate function based on user selection."""

    st.set_page_config(
        page_title="Financial Sentiment",
        page_icon="💵",
        layout="centered",
        initial_sidebar_state="auto",
    )

    st.sidebar.text("")
    st.sidebar.text("")

    option = st.sidebar.selectbox(
        "I want to...",
        [
            "",
            "see latest headlines",
            "view the dashboard",
            "analyze a single headline",
        ],
    )

    st.text(" ")
    st.sidebar.markdown("---")

    if option == "":
        project_description()
    elif option == "see latest headlines":
        latest_headlines()
    elif option == "view the dashboard":
        dashboard()
    elif option == "analyze a single headline":
        single_sentence()
    else:
        model_performance()


if __name__ == "__main__":
    stg.enable_logging(log_filename="project_logs.log", logging_level=logging.INFO)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
