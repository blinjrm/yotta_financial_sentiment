import logging
import os

import pandas as pd
import streamlit as st
import datetime

import src.settings.base as stg
from src.application.predict_string import make_prediction_string
from src.domain.predict_utils import list_trained_models
from src.domain.app_utils import newspapers_plot, raw_data_plot, tendency_plot


def project_description():

    st.title("Sentiment analysis of financial headlines")

    description = """ 
    ## Some text
    """

    st.markdown(description)


def single_sentence():

    st.title("Sentiment analysis on a sentence")
    st.text("")

    trained_models = list_trained_models()
    trained_models.append("zero-shot-classifier")

    model = st.sidebar.radio("Choose a model:", (trained_models))

    sentence = st.text_input(label="Type your sentence here:")

    if len(sentence) > 1:
        with st.spinner("Analyzing..."):
            prediction = make_prediction_string(sentence, model_name=model)

        st.text("\nSentiment analysis:")
        st.text(f"Label: {prediction['label']}, with score: {round(prediction['score'], 4)}")

    # TODO: add model description


def dashboard():

    st.title("Financial sentiment in the USA in 2020")
    st.text("")

    st.sidebar.text("Choose display options:")
    display_option = st.sidebar.radio(label="", options=["Tendencies", "Newspapers", "Raw data"])

    st.sidebar.markdown("---")

    if display_option == "Tendencies":
        st.text("")
        fig = tendency_plot()
        st.pyplot(fig)

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


def model_performance():

    st.title("Model performance")
    st.text("")

    trained_models = list_trained_models()
    model = st.sidebar.radio("Choose a model:", (trained_models))
    st.header(f"Showing {model}")


def main():

    st.set_page_config(
        page_title="Financial Sentiment",
        page_icon="💵",
        layout="centered",
        initial_sidebar_state="auto",
    )

    option = st.sidebar.selectbox(
        "I want to...",
        [
            "",
            "analyze a single sentence",
            "view the dashboard",
            "compare model performance",
            "see some ballons",
        ],
    )

    st.text(" ")
    st.sidebar.markdown("---")

    if option == "":
        project_description()
    elif option == "analyze a single sentence":
        single_sentence()
    elif option == "view the dashboard":
        dashboard()
    elif option == "compare model performance":
        model_performance()
    else:
        st.balloons()
        st.sidebar.markdown("Because 2020 wasn't *all* bad.")
        balloons = st.sidebar.button("Again! 🤩")
        if balloons:
            st.balloons()


if __name__ == "__main__":
    # stg.enable_logging(log_filename="project_logs.log", logging_level=logging.INFO)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
