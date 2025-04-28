import streamlit as st
import pandas as pd
import plotly.express as px
from functions import config_figure

# -------------------------------
# Load Styles
# -------------------------------
def load_styles():
    with open("assets/style.css") as f:
        return f"<style>{f.read()}</style>"

st.markdown(load_styles(), unsafe_allow_html=True)

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("data/IEP_2017_2018_Stations.xlsx")
    return df

stations = load_data()

# -------------------------------
# Categorize Grids
# -------------------------------
def categorize_by_prefix(grid_value):
    if grid_value.startswith("KML"):
        return "KML"
    elif grid_value.startswith("SHBML"):
        return "SHBML"
    elif grid_value.startswith("SML"):
        return "SML"
    elif grid_value.startswith("NML"):
        return "NML"
    else:
        return "HAB"

stations['Grid_Category'] = stations['Grid'].apply(categorize_by_prefix)

# -------------------------------
# Cached Station Figure
# -------------------------------
@st.cache_data
def create_station_figure(df):
    fig = px.scatter_map(
        df, 
        lat="Lat (°S)", 
        lon="Lon (°E)", 
        hover_name="Grid",
        hover_data=["Grid"],
        zoom=4.5,
        height=600,
        map_style="open-street-map",
        center={"lat": -33.0, "lon": 22.0},
        color="Grid_Category",
        color_discrete_sequence=px.colors.qualitative.Set1
    )

    fig.update_traces(
        marker=dict(
            size=8,
            symbol="circle",
            opacity=0.9,
        ),
    )

    fig.update_layout(
        title="Sampling Stations for IEP",
        legend={'title': 'Research Grids Lines'},
        showlegend=True,
    )
    return fig

# -------------------------------
# Page Header
# -------------------------------
st.markdown(
    """
    <div class="header" style="display: flex; justify-content: space-between; font-family:Roboto; align-items: center; background-color: #408a93; padding: 0px;">
        <div style="font-size: 1.4rem; font-weight: bold; color: white; margin-left: 0px;">
            State of the Ocean: Integrated Ecosystem Monitoring Programme 🐬 🔬.
        </div>
        <div style="font-size: 1rem; font-weight: bold; font-family:Roboto; margin-right: 100px;margin-top: 10px">
            <a href="https://data.ocean.gov.za/" target="_blank" style="color: white; text-decoration: none;">
                Access Historical Data
            </a>
        </div>
    </div>
    <div style="text-align: left; font-size: 0.9rem; margin-left: 0px; margin-bottom: 3px;font-family:Roboto; font-weight: bold; color: white; background-color: #408a93; padding: 0px;">
        Department of Forestry, Fisheries, and the Environment
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# Description Text
# -------------------------------
col, = st.columns([20])

with col:
    st.markdown(
        """
        <div style="max-width: 95%; margin-top: 15px; margin-right: 10px; text-align: justify;">
            <p class="small-text">
            Welcome to the Integrated Ecosystem Programme: Southern Benguela (IEP:SB) Analysis Dashboard.
            The IEP:SB is a multi-disciplinary project designed to undertake oceanographic research in the 
            Southern Benguela region. The primary objective of the IEP:SB is to develop ecosystem indicators that can be used to effectively monitor and 
            understand the Southern Benguela. These indicators cover a wide range of ecosystem components, 
            including physical, chemical, planktonic, microbial, seabird, and benthic elements. 
            The data and insights gained from this program are crucial for ecosystem-based management and conservation efforts in the Southern Benguela region.
            It serves as a platform for collaboration and learning, bringing together students and researchers from various disciplines to study the complex interactions within this marine ecosystem.
            <br><br>
            This dashboard allows you to explore and visualise the CTD data collected during IEP voyages. 
            You can filter the data by grid, date range, and season, and visualize it through various types of plots, including CTD profiles, TS diagrams, and correlation heatmaps. 
            Each plot comes with a download option, enabling you to save the figures in different formats for further use offline.
            The data used here is published online in South Africa's Marine Information Management System (MIMS). The link to the MIMS catalogue is available under the resource section.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------
# Station Map Plot
# -------------------------------
col0, col1, col2 = st.columns([1, 18, 1])  # Narrow margins

with col1:
    fig_stations = create_station_figure(stations)
    st.plotly_chart(fig_stations, use_container_width=True, config=config_figure)
