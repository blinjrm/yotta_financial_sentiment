import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

import src.settings.base as stg
from src.infrastructure.infra import DatasetBuilder


@st.cache
def load_data_app():

    df = DatasetBuilder(stg.OUTPUT_FILENAME, stg.OUTPUTS_DIR).data

    df.loc[df["label"] == "negative", "sentiment"] = -1
    df.loc[df["label"] == "neutral", "sentiment"] = 0
    df.loc[df["label"] == "positive", "sentiment"] = 1

    df["month"] = pd.DatetimeIndex(df["date"]).strftime("%m")
    df["week"] = pd.DatetimeIndex(df["date"]).strftime("%W")

    return df


@st.cache
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
    sns.despine(left=True, bottom=True, right=True)

    if smooth == "Day":
        ax.xaxis.set_major_locator(plt.MaxNLocator(15))
    elif smooth == "Week":
        ax.xaxis.set_major_locator(plt.MaxNLocator(20))
    else:
        pass

    ax.set_ylim(-1, 1)
    plt.xticks(rotation=90)

    return fig


if __name__ == "__main__":
    import datetime

    start_date = "2020-03-01"
    end_date = "2020-11-30"
    newspaper = "Both"
    smooth = "Month"

    df = newspapers_data(start_date, end_date, newspaper, smooth)
    print(df)
