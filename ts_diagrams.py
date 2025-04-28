import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import gsw
from functions import config_figure

# -------------------------------
# Style Loading
# -------------------------------
#@st.cache_resource
def load_styles():
    with open("assets/style.css") as f:
        return f"<style>{f.read()}</style>"

st.markdown(load_styles(), unsafe_allow_html=True)

# ------------------------------
# Header Section
# ------------------------------
st.markdown(
    """
    <div class="header" style="display: flex; justify-content: space-between; font-family:Roboto; align-items: center; background-color: #408a93; padding: 0px;">
        <!-- Left Section -->
        <div style="font-size: 1.4rem; font-weight: bold; color: white; margin-left: 0px;">
             State of the Ocean: Temperature–Salinity Diagram 🌊
        </div>
        <!-- Right Section -->
        <div style="font-size: 1rem; font-weight: bold; font-family:Roboto; margin-right: 100px;margin-top: 10px">
            <a href="https://data.ocean.gov.za/" target="_blank" style="color: white; text-decoration: none;">
                Access Historical Data
            </a>
        </div>
    </div>
    <div style="text-align: left; font-size: 0.9rem; margin-left: 0px; margin-bottom: 10px; font-family:Roboto; font-weight: bold; color: white; background-color: #408a93; padding: 0px;">
        Department of Forestry, Fisheries, and the Environment
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------
# Data Processing Functions
# ------------------------------
@st.cache_data
def transform_data(df):
    """Preprocess and calculate required fields"""
    df = df.copy()
    df["latitude"] = df["latitude"].apply(lambda x: -abs(x))
    
    # Calculate density if missing
    if "density_sigma_theta" not in df.columns:
        df["density_sigma_theta"] = gsw.sigma0(
            df["salinity_psu"], 
            df["temperature_c"]
        )
    return df

@st.cache_data
def calculate_isopycnals(filtered_data):
    """Optimized isopycnal calculation"""
    t_min = filtered_data["temperature_c"].min() - 1
    t_max = filtered_data["temperature_c"].max() + 1
    s_min = filtered_data["salinity_psu"].min() - 1
    s_max = filtered_data["salinity_psu"].max() + 1

    ti = np.linspace(t_min, t_max, 100)
    si = np.linspace(s_min, s_max, 100)
    si_grid, ti_grid = np.meshgrid(si, ti)
    
    dens = gsw.sigma0(si_grid, ti_grid)
    return si, ti, dens

# ------------------------------
# Water Mass Definitions (verified)
# ------------------------------
WATER_MASSES = [
    # Antarctic Bottom Water (ABW)
    {
        "name": "Antarctic Bottom Water",
        "abbreviation": "ABW",
        "temp_min": -2,
        "temp_max": 2,
        "sal_min": 34.6,
        "sal_max": 34.8,
        "dens_min": 27.9,
        "dens_max": np.inf,
        "color": "black"
    },
    # North Atlantic Deep Water (NADW)
    {
        "name": "North Atlantic Deep Water",
        "abbreviation": "NADW",
        "temp_min": 2,
        "temp_max": 4,
        "sal_min": 34.9,
        "sal_max": 35.0,
        "dens_min": 27.8,
        "dens_max": np.inf,
        "color": "black"
    },
    # Antarctic Intermediate Water variants
    {
        "name": "Low Salinity Antarctic Intermediate Water",
        "abbreviation": "LSAIW",
        "temp_min": 3,
        "temp_max": 6,
        "sal_min": 34.3,
        "sal_max": 34.6,
        "dens_min": 27.2,
        "dens_max": 27.5,
        "color": "black"
    },
    {
        "name": "High Salinity Antarctic Intermediate Water",
        "abbreviation": "HSAIW",
        "temp_min": 5,
        "temp_max": 10,
        "sal_min": 34.5,
        "sal_max": 35.0,
        "dens_min": 27.3,
        "dens_max": 27.6,
        "color": "black"
    },
    # Central Water masses
    {
        "name": "Low Salinity Central Water",
        "abbreviation": "LSCW",
        "temp_min": 8,
        "temp_max": 15,
        "sal_min": 34.3,
        "sal_max": 34.8,
        "dens_min": 26.5,
        "dens_max": 27.0,
        "color": "black"
    },
    {
        "name": "High Salinity Central Water",
        "abbreviation": "HSCW",
        "temp_min": 8,
        "temp_max": 15,
        "sal_min": 34.8,
        "sal_max": 35.5,
        "dens_min": 26.8,
        "dens_max": 27.4,
        "color": "black"
    },
    # Surface/Upwelled waters
    {
        "name": "Modified Upwelled Water",
        "abbreviation": "MUW",
        "temp_min": 15,
        "temp_max": 20,
        "sal_min": 35.0,
        "sal_max": 36.0,
        "dens_min": 25.8,
        "dens_max": 26.5,
        "color": "black"
    },
    {
        "name": "Oceanic Surface Water",
        "abbreviation": "OSW",
        "temp_min": 20,
        "temp_max": 30,
        "sal_min": 34.5,
        "sal_max": 36.5,
        "dens_min": 24.0,
        "dens_max": 25.5,
        "color": "black"
    }
]

# ------------------------------
# Main Application Logic
# ------------------------------
def main():
    data = st.session_state.get("data")
    
    # Check if data is loaded and valid
    if data is None or data.empty:
        st.error("Data not found. Please load data in the main application.")
        st.stop()  # Stop execution if data is missing
    
    data = transform_data(data)

    # Session state management
    default_grid = "NML010" if "NML010" in data["grid"].unique() else "All Stations"
    st.session_state.grids_selected = [default_grid]

    with st.expander("Filter Data"):
        grid_options = ["All Stations"] + data["grid"].unique().tolist()
        grids_selected = st.multiselect(
            "Select Grid(s)", 
            grid_options, 
            default=st.session_state.grids_selected
        )
        show_water_masses = st.checkbox("Overlay Water Masses", value=True)  # <- ADD THIS
    st.session_state.grids_selected = grids_selected

    # Data filtering
    if "All Stations" in grids_selected:
        filtered_data = data
    else:
        filtered_data = data[data["grid"].isin(grids_selected)]

    # Check if the filtered data is empty
    if filtered_data.empty:
        st.warning("No data available after filtering.")
        return
    # Main display
    col1, col2 = st.columns([3, 8])

    with col1:
        if not filtered_data.empty:
            fig_map = px.scatter_map(
                filtered_data,
                lat="latitude",
                lon="longitude",
                hover_name="grid",
                zoom=4.5,
                height=600,
                map_style="open-street-map",
                center={"lat": -33.0, "lon": 17.0}
            )
            fig_map.update_traces(
                marker=dict(size=4, symbol="circle", opacity=0.7, color="black")
            )
            st.plotly_chart(fig_map, use_container_width=True, config=config_figure)

    with col2:
        if filtered_data.empty:
            st.warning("No data selected. Please select at least one grid.")
            return

        # Create T-S Diagram
        fig_wm = go.Figure()

        # Calculate and add isopycnals
        si, ti, dens = calculate_isopycnals(filtered_data)
        fig_wm.add_trace(
            go.Contour(
                x=si,
                y=ti,
                z=dens,
                contours_coloring="lines",
                showscale=False,
                contours=dict(
                    start=np.floor(dens.min()),
                    end=np.ceil(dens.max()),
                    size=0.5,
                    showlabels=True,
                    labelfont=dict(size=12, color="black")
                )
            )
        )

        # Add data points
        fig_wm.add_trace(
            go.Scatter(
                x=filtered_data["salinity_psu"],
                y=filtered_data["temperature_c"],
                mode="markers",
                marker=dict(
                    opacity=0.2,
                    color=filtered_data["pressure_db"],
                    colorscale="spectral",
                    colorbar=dict(
                        title=dict(text="pressure_db", side="bottom"),
                        orientation="h",
                        x=0.5,
                        y=-0.4
                    ),
                    size=3
                )
            )
        )

        # Add water mass annotations
        if show_water_masses:
            for wm in WATER_MASSES:
                mask = (
                    (filtered_data["temperature_c"] >= wm["temp_min"]) &
                    (filtered_data["temperature_c"] <= wm["temp_max"]) &
                    (filtered_data["salinity_psu"] >= wm["sal_min"]) &
                    (filtered_data["salinity_psu"] <= wm["sal_max"]) &
                    (filtered_data["density_sigma_theta"] >= wm["dens_min"]) &
                    (filtered_data["density_sigma_theta"] <= wm["dens_max"])
                )
                wm_data = filtered_data[mask]
                
                if not wm_data.empty:
                    avg_sal = wm_data["salinity_psu"].mean()
                    avg_temp = wm_data["temperature_c"].mean()
                    
                    fig_wm.add_annotation(
                        x=avg_sal,
                        y=avg_temp,
                        text=f"<b>{wm['abbreviation']}</b>",
                        showarrow=False,
                        font=dict(size=12, color=wm["color"]),
                        xanchor="center",
                        yanchor="bottom"
                    )
        fig_wm.update_layout(
            title="TS Diagram",
            xaxis_title="salinity_psu",
            yaxis_title="temperature_c",
            width=800,
            height=600,
            showlegend=False,
            title_font=dict(color="black")
        )
        st.plotly_chart(fig_wm, config=config_figure)

if __name__ == "__main__":
    main()