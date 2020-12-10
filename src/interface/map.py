"""
https://pydeck.gl/index.html
"""


import plotly.express as px
import pydeck


def pydeck_map():

    # 2014 locations of car accidents in the UK
    UK_ACCIDENTS_DATA = (
        "https://raw.githubusercontent.com/uber-common/"
        "deck.gl-data/master/examples/3d-heatmap/heatmap-data.csv"
    )

    # Define a layer to display on a map
    layer = pydeck.Layer(
        "HexagonLayer",
        UK_ACCIDENTS_DATA,
        get_position="[lng, lat]",
        auto_highlight=True,
        elevation_scale=50,
        pickable=True,
        elevation_range=[0, 3000],
        extruded=True,
        coverage=1,
    )

    # Set the viewport location
    view_state = pydeck.ViewState(
        longitude=-1.415,
        latitude=52.2323,
        zoom=6,
        min_zoom=5,
        max_zoom=15,
        pitch=40.5,
        bearing=-27.36,
    )

    # Render
    r = pydeck.Deck(layers=[layer], initial_view_state=view_state)
    r.to_html("demo.html")

    # Please see the note about using a Mapbox API token here:
    # https://github.com/uber/deck.gl/tree/master/bindings/python/pydeck#mapbox-api-token

    return r


def plotly_map():

    country = px.data.gapminder()
    fig = px.scatter_geo(
        country,
        locations="iso_alpha",
        color="continent",
        hover_name="country",
        size="pop",
        animation_frame="year",
        projection="winkel tripel",
    )

    return fig
