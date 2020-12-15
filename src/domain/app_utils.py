import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from dateutil.parser import parse

import src.settings.base as stg
from src.application.predict_string import make_prediction_string
from src.infrastructure.infra import DatasetBuilder, WebScraper


@st.cache()
def load_data_app():

    df = DatasetBuilder(stg.OUTPUT_FILENAME, stg.OUTPUTS_DIR).data

    df.loc[df["label"] == "negative", "sentiment"] = -1
    df.loc[df["label"] == "neutral", "sentiment"] = 0
    df.loc[df["label"] == "positive", "sentiment"] = 1

    df["month"] = pd.DatetimeIndex(df["date"]).strftime("%m")
    df["week"] = pd.DatetimeIndex(df["date"]).strftime("%W")

    return df


@st.cache()
def newspapers_data(start_date, end_date, newspaper, smooth):

    df = load_data_app()
    df_with_dates = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

    if newspaper == "Both":
        df_with_newspaper = df_with_dates
    else:
        df_with_newspaper = df_with_dates[df_with_dates["source"] == newspaper]

    df_with_newspaper = df_with_newspaper.rename(columns={"date": "day"})

    pivot = df_with_newspaper.pivot_table(
        values="sentiment", index="source", columns=smooth.lower(), aggfunc=np.mean
    )

    return pivot


def newspapers_plot(start_date, end_date, newspaper, smooth):

    df = newspapers_data(start_date, end_date, newspaper, smooth)

    fig, ax = plt.subplots()
    sns.lineplot(data=df.T, palette="Accent", linewidth=1)
    sns.despine()

    if smooth == "Day":
        ax.xaxis.set_major_locator(plt.MaxNLocator(15))
    elif smooth == "Week":
        ax.xaxis.set_major_locator(plt.MaxNLocator(20))
    else:
        pass

    ax.set_ylim(-1, 1)
    plt.xticks(rotation=90)

    return fig


def raw_data_plot():

    df = load_data_app()

    label_repartition = df["label"].value_counts().rename_axis("label").reset_index(name="total")

    fig = plt.figure(figsize=(8, 4))
    sns.barplot(
        x="total",
        y="label",
        data=label_repartition,
        order=["negative", "neutral", "positive"],
        palette="Accent",
    )
    sns.despine(left=True, bottom=True)
    plt.title("Total number of headlines per sentiment class")

    return df, fig


def tendency_data():
    sentiment = load_data_app()
    sentiment = (
        sentiment.drop(columns=["headline", "label", "score", "source", "month", "week"])
        .groupby("date", as_index=False)
        .mean()
    )
    sentiment["week"] = pd.DatetimeIndex(sentiment["date"]).strftime("%W")

    stock = pd.read_csv("".join((stg.APP_DATA_DIR, "stock_sp500.csv")))
    stock = (
        stock.drop(columns=["Open", "High", "Low", "Adj Close"])
        .set_index("Date")
        .rename(columns={"Close": "S&P500"})
    )

    covid = pd.read_csv("".join((stg.APP_DATA_DIR, "us_covid.csv")), sep=";")
    covid["date"] = covid["submission_date"].apply(lambda x: parse(x).strftime("%Y-%m-%d"))
    covid = (
        covid.groupby("date", as_index=False)
        .sum()
        .set_index("date")
        .rename(columns={"new_case": "New cases", "new_death": "New deaths"})
    )

    data = sentiment.join(stock, on="date", how="left")
    data_with_covid = data.join(covid, on="date", how="left")
    data_with_covid = data_with_covid.drop(columns=["date"]).groupby("week", as_index=False).mean()

    return data_with_covid


def tendency_plot():
    data = tendency_data()
    data = data.set_index("week")

    min_max_scaler = MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(data)
    df = pd.DataFrame(data_scaled, columns=data.columns)

    fig, ax = plt.subplots()
    sns.lineplot(data=df, palette="Accent", linewidth=1)
    sns.despine()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)

    return fig


def tendency_heatmap():
    data = tendency_data()
    data = data.set_index("week")
    corr = data.corr()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax = sns.heatmap(
        corr,
        vmin=-1,
        vmax=1,
        annot=True,
        cmap=sns.color_palette("vlag", as_cmap=True),
    )
    plt.yticks(va="center")

    return fig


def latest_news_widget(PARAMS):
    scraper = WebScraper(**PARAMS)
    latest_headline = scraper.get_headlines().iloc[0, 0]

    st.text("")
    st.header(scraper.newspaper)
    st.markdown("**" + latest_headline + "**")
    with st.spinner("Analyzing..."):
        prediction = make_prediction_string(latest_headline, model_name="distilroberta-base")

    st.text("\nSentiment analysis:")
    st.text(f"Label: {prediction['label']}, with score: {round(prediction['score'], 4)}")
    st.text("")


if __name__ == "__main__":
    data = tendency_data()
