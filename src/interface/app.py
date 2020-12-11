import logging

import pandas as pd
import streamlit as st

import src.settings.base as stg

from src.application.predict_string import make_prediction_string
from src.domain.predict_utils import list_trained_models


def single_sentence():

    st.header("Sentiment analysis on a sentence")

    trained_models = list_trained_models()
    trained_models.append("zero-shot-classifier")

    st.sidebar.markdown("-" * 17)
    model = st.sidebar.radio("Choose a model:", (trained_models))

    sentence = st.text_input(label="Type a sentence for sentiment analysis:")

    if len(sentence) > 1:
        with st.spinner("Analyzing..."):
            prediction = make_prediction_string(sentence, model_name=model)

        st.text("\nSentiment analysis:")
        st.text(f"Label: {prediction['label']}, with score: {round(prediction['score'], 4)}")


def project_description():
    description = """Some text
    """
    st.beta_container

    st.markdown(description)


def list_headlines():

    st.header("Financial sentiment in the USA throughout 2020")

    st.sidebar.text("")
    st.sidebar.text("Which neswpaper \ndo you want to include?")

    show_newspapers = {"Reuters": True, "Financial Times": True}
    for newspaper, value in show_newspapers.items():
        checkbox_show_newspapers = st.sidebar.checkbox(label=newspaper, value=value)
        if checkbox_show_newspapers:
            show_newspapers[newspaper] = True
        else:
            show_newspapers[newspaper] = False

    left_column, right_column = st.beta_columns(2)
    bt1 = left_column.button("Show newspapers?")
    if bt1:
        right_column.write(show_newspapers)


def model_performance():

    trained_models = list_trained_models()

    st.sidebar.text("")
    model = st.sidebar.radio("Choose a model:", (trained_models))

    if model == "roberta-base":
        st.header("Showing roberta-base")
    else:
        st.header("distilroberta-base")


def main():
    st.title("Sentiment analysis of financial headlines")
    st.text("")
    st.text("")

    expander = st.beta_expander("Project description")
    expander.write(
        "The goal of this project is to visualize the evolution of financial sentiment \
        in the global market as reflected by the newspaper headlines. "
    )

    st.text(" ")
    st.text(" ")
    option = st.sidebar.selectbox(
        "I want to...",
        ["", "analyze a single sentence", "see the dashboard", "see model performance"],
    )

    st.text(" ")

    if option == "":
        project_description()
    elif option == "analyze a single sentence":
        single_sentence()
    elif option == "see the dashboard":
        list_headlines()
    else:
        model_performance()


if __name__ == "__main__":
    stg.enable_logging(log_filename="project_logs.log", logging_level=logging.INFO)
    main()
