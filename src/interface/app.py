import logging
import time

import numpy as np
import pandas as pd
import pydeck
import streamlit as st

import src.settings.base as stg
from src.application.predict_string import make_prediction_string
from src.domain.predict_utils import list_trained_models


def single_sentence():

    sentence = st.text_input(label="Type a sentence for sentiment analysis:")

    if len(sentence) > 1:
        with st.spinner("Analyzing..."):
            prediction = make_prediction_string(sentence)

        st.text("\nSentiment analysis:")
        st.text(f"Label: {prediction['label']}, with score: {round(prediction['score'], 4)}")


def list_headlines():
    st.sidebar.text("")
    st.sidebar.text("Which countries \ndo you want to include?")

    show_country = {"USA": True, "UK": True, "China": True}
    for country in show_country.keys():
        checkbox_show_country = st.sidebar.checkbox(label=country, value=True)
        if checkbox_show_country:
            show_country[country] = True
        else:
            show_country[country] = False

    left_column, right_column = st.beta_columns(2)
    bt1 = left_column.button("Show countries?")
    if bt1:
        right_column.write(show_country)

    bt2 = st.button("Show map")
    if bt2:
        r = pydeck_map()
        r
        # fig = plotly_map()
        # st.text(fig.show())


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
        "I want to...", ["analyze a single sentence", "see the dashboard", "test model performance"]
    )
    "Currently enabling predictions for: ", option

    st.text(" ")

    if option == "analyze a single sentence":
        single_sentence()
    else if option == "see the dashboard":
        list_headlines()
    else:
        pass


if __name__ == "__main__":
    stg.enable_logging(log_filename="project_logs.log", logging_level=logging.INFO)
    main()
