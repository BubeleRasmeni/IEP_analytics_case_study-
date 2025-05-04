import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from functions import generate_correlation_heatmap, config_figure
from utils import load_data
import matplotlib.colors as mcolors
import colorsys
import hashlib
# -------------------------------
# Style Loading
# -------------------------------
#@st.cache_resource
def load_styles():
    with open("assets/style.css") as f:
        return f"<style>{f.read()}</style>"

st.markdown(load_styles(), unsafe_allow_html=True)

# ------------------------------
# Constants & Configuration
# ------------------------------
COLOR_SCHEMES = {
    "temperature_c": "red",
    "salinity_psu": "blue",
    "oxygen_ml_l": "green",
    "fluorescence_mg_m3": "purple"
}

def stable_hash(station_name):
    """Generate a stable integer hash from a string using MD5."""
    return int(hashlib.md5(station_name.encode()).hexdigest(), 16)
# -------------------------------
# Color Configuration
# -------------------------------
VARIABLE_PALETTES = {
    "temperature_c": {
        "base": "#FF0000",
        "shades": [
            '#C41E3A', '#CD5C5C', '#FF2E2E', '#750000',
            '#FF2400', '#D9381E', '#DC143C', '#DA012D',
            '#D10000', '#A30000', '#9B111E', '#960018',
            '#722F37', '#800020', '#800000', '#750000'
        ]
    },
    "salinity_psu": {
        "base": "#0000FF",
        "shades": [
            '#120A8F', '#00BFFF', '#7B68EE', '#0000FF',
            '#0024FF', '#1E38D9', '#3C14DC', '#2D01DA',
            '#0000D1', '#0000A3', '#111E9B', '#001896',
            '#2F3772', '#002080', '#000080', '#000075'
        ]
    },
    "oxygen_ml_l": {
        "base": "#00FF00",
        "shades": [
            '#8AFF8A', '#006400', '#01796F', '#3CB371',
            '#01796F', '#14DC3C', '#01DA2D',
            '#00D100', '#00A300', '#1E9B11', '#189600',
            '#37722F', '#208000', '#008000', '#007500'
        ]
    },
    "fluorescence_mg_m3": {
        "base": "#800080",
        "shades": [
            '#FF8AFF', '#FF5CFF', '#FF2EFF', '#FF00FF',
            '#FF24FF', '#D938D9', '#DC14DC', '#DA01DA',
            '#D100D1', '#A300A3', '#9B119B', '#960096',
            '#722F72', '#800080', '#800080', '#750075'
        ]
    }
}


def get_year_range(years):
    """Calculate dynamic year range for normalization"""
    if not years:
        return (2000, 2050)  # Default range if no years available
    min_year = min(years)
    max_year = max(years)
    return (min_year, max_year)

def generate_station_color(variable, station_name, year=None, year_range=None):
    """
    Generates colors using predefined palette first, then dynamic variations.
    Incorporates year information if provided.
    """
    palette = VARIABLE_PALETTES[variable]
    num_predefined = len(palette["shades"])
    
    # Get stable index from station name and optionally year
    hash_input = station_name
    if year is not None:
        hash_input += str(year)
    hash_val = stable_hash(hash_input)
    idx = hash_val % (num_predefined + 5)  # Allow 5 extra generated colors
    
    if idx < num_predefined:
        return palette["shades"][idx]
    
    # Generate new color variation
    base_rgb = mcolors.to_rgb(palette["base"])
    h, l, s = colorsys.rgb_to_hls(*base_rgb)
    
    # Create unique variations using multiple hash properties
    hash_ratio = (hash_val % 10000) / 10000
    secondary_hash = stable_hash(station_name + "_secondary") % 10000 / 10000
    
    # Enhanced variation parameters - now optionally influenced by year
    if year is not None and year_range is not None:
        min_year, max_year = year_range
        year_range_span = max(1, max_year - min_year)  # Avoid division by zero
        year_factor = (year - min_year) / year_range_span
        
        lightness_adjust = (hash_ratio - 0.5 + year_factor * 0.2) * 0.8
        hue_adjust = (secondary_hash - 0.5 + year_factor * 0.1) * 0.3
    else:
        lightness_adjust = (hash_ratio - 0.5) * 0.8
        hue_adjust = (secondary_hash - 0.5) * 0.3
    
    saturation_adjust = 1.2 + (hash_ratio - 0.5) * 0.4
    
    new_h = (h + hue_adjust) % 1.0
    new_l = max(0.05, min(0.95, l + lightness_adjust))
    new_s = max(0.4, min(1.0, s * saturation_adjust))
    
    # Convert back to RGB
    new_rgb = colorsys.hls_to_rgb(new_h, new_l, new_s)
    return f"rgb({int(new_rgb[0]*255)}, {int(new_rgb[1]*255)}, {int(new_rgb[2]*255)})"



DASH_PATTERNS = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]

# ------------------------------
# Core Functions
# ------------------------------
def render_header():
    """Render the application header"""
    st.markdown(
        """
        <div class="header" style="display: flex; justify-content: space-between; font-family:Roboto; align-items: center; background-color: #408a93; padding: 0px;">
            <div style="font-size: 1.4rem; font-weight: bold; color: white; margin-left: 0px;">
                 State of the Ocean: Essential Ocean Variables
            </div>
            <div style="font-size: 1rem; font-weight: bold; font-family:Roboto; margin-right: 100px;margin-top: 10px">
                <a href="https://data.ocean.gov.za/" target="_blank" style="color: white; text-decoration: none;">
                    Access Historical Data
                </a>
            </div>
        </div>
        <div style="text-align: left; font-size: 0.9rem; margin-left: 0px; font-family:Roboto; font-weight: bold; color: white; background-color: #408a93; padding: 0px;">
            Department of Forestry, Fisheries, and the Environment
        </div>
        """,
        unsafe_allow_html=True,
    )

def create_map(df):
    """Generate a map visualization for station locations"""
    fig = px.scatter_map(
        df, lat="latitude", lon="longitude", 
        hover_name="grid", zoom=4.7, height=550,map_style="open-street-map",center={"lat": -32.0, "lon": 18.0},
    )
    fig.update_layout(
        margin=dict(l=5, r=5, t=30, b=5),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
    )
    fig.update_traces(marker=dict(size=6, symbol="circle", opacity=0.7, color="black"))
    return fig

def get_unique_options(data):
    """Extract unique filter options from data"""
    return {
        "grids": list(data["grid"].unique()),
        "seasons": list(data["season"].unique()),
        "years": list(data["datetime"].dt.year.dropna().unique())
    }

# ------------------------------
# Tab Components
# ------------------------------
def render_ctd_tab(data):
    """CTD Profiles Tab"""
    st.header("CTD Profiles")
    
    # Get filter options
    options = get_unique_options(data)
    year_range = get_year_range(options["years"])
    with st.expander("Filter Data"):
        cols = st.columns(4)
        with cols[0]:
            sel_grids = st.multiselect("grid", options["grids"], default=[options["grids"][0]], key="ctd_grids")
        with cols[1]:
            sel_seasons = st.multiselect("Season", options["seasons"], default=[options["seasons"][0]], key="ctd_seasons")
        with cols[2]:
            sel_years = st.multiselect("Year", options["years"], default=[options["years"][0]], key="ctd_years")
        with cols[3]:
            selected_vars = st.multiselect(
                "Select up to Two Variables to Plot with Depth",
                options=list(COLOR_SCHEMES.keys()),
                default=["temperature_c"],
                max_selections=2,
                key="ctd_vars"
            )

    # Filter data
    df_ctd = data[
        data["grid"].isin(sel_grids) &
        data["season"].isin(sel_seasons) &
        data["datetime"].dt.year.isin(sel_years)
    ]

    if df_ctd.empty:
        st.warning("No data matches current filters")
        return

    # Create season styles
    unique_seasons = sorted(df_ctd["season"].dropna().unique())
    season_styles = {season: DASH_PATTERNS[i % len(DASH_PATTERNS)] for i, season in enumerate(unique_seasons)}

    # Layout
    col1, col2 = st.columns([0.2, 0.7])
    with col1:
        st.plotly_chart(create_map(df_ctd), use_container_width=True, config=config_figure,key="map_chart_ctd_profiles")

    with col2:
        if not selected_vars:
            st.warning("Please select at least one variable.")
            return

        fig = go.Figure()
        for var_index, var in enumerate(selected_vars):
            xaxis = "x" if var_index == 0 else "x2"
            show_group_title = True
            df_ctd["year"] = df_ctd["datetime"].dt.year
            unique_combos = df_ctd[["grid", "season", "year"]].drop_duplicates()
            color_palette = px.colors.qualitative.Set2 + px.colors.qualitative.Dark24
            color_map = {
                (row.grid, row.season, row.year): color_palette[i % len(color_palette)]
                for i, row in enumerate(unique_combos.itertuples(index=False))
            }
            for season in sel_seasons:
                for station in df_ctd["grid"].unique():
                    for year in sel_years:
                        sub = df_ctd[
                            (df_ctd["grid"] == station) &
                            (df_ctd["season"] == season) &
                            (df_ctd["datetime"].dt.year == year)
                        ].sort_values("depth_m")

                        if sub.empty:
                            continue
                        combo = (station, season, year)
                        trace_color = color_map.get(combo, COLOR_SCHEMES[var])  # fallback to var color
                        fig.add_trace(go.Scatter(
                            x=sub[var], y=sub["depth_m"],
                            mode="lines",
                            name=f"{station} <br> {season} <br> {year}",
                            legendgroup=var,
                            legendgrouptitle=dict(text=var) if show_group_title else None,
                            line=dict(
                                color=generate_station_color(var, station, year, year_range),
                                dash=season_styles.get(season, "solid")
                            ),
                            xaxis=xaxis
                        ))
                        show_group_title = False

        # Configure layout
        layout = {
            "yaxis": dict(title="depth_m", autorange="reversed"),
            "margin": dict(t=60, r=20, b=120, l=60, pad=10),
            "paper_bgcolor": "white",
            "plot_bgcolor": "white",
            "template": "simple_white",
            "height":600,
            "showlegend": True,
            "legend":dict(
                        groupclick="toggleitem",
                    ),
            "xaxis": dict(
                title=selected_vars[0],
                side="bottom",
                title_font=dict(color=VARIABLE_PALETTES[selected_vars[0]]["base"]),
                tickfont=dict(color=VARIABLE_PALETTES[selected_vars[0]]["base"]),
                showgrid=False
            )
        }

        if len(selected_vars) == 2:
            layout["xaxis2"] = dict(
                title=selected_vars[1],
                side="top",
                overlaying="x",
                title_font=dict(color=VARIABLE_PALETTES[selected_vars[1]]["base"]),
                tickfont=dict(color=VARIABLE_PALETTES[selected_vars[1]]["base"]),
                showgrid=False
            )
        fig.update_layout(**layout,)
        st.plotly_chart(fig, use_container_width=True, config=config_figure,key="map_chart_ctd_profiles2")

def render_regression_tab(data):
    """Regression Analysis Tab"""
    st.header("Regression Diagram")
    
    options = get_unique_options(data)
    
    with st.expander("Filter Data"):
        cols = st.columns(3)
        with cols[0]:
            sel_grids = st.multiselect("Grid", options["grids"], default=[options["grids"][0]], key="reg_grids")
        with cols[1]:
            sel_seasons = st.multiselect("Season", options["seasons"], default=[options["seasons"][0]], key="reg_seasons")
        with cols[2]:
            sel_years = st.multiselect("Year", options["years"], default=[options["years"][0]], key="reg_years")
        
        cols = st.columns(3)
        with cols[0]:
            x_var = st.selectbox("X Variable", list(COLOR_SCHEMES.keys()), key="x_reg")
        with cols[1]:
            y_var = st.selectbox("Y Variable", list(COLOR_SCHEMES.keys())[2], key="y_reg")
        with cols[2]:
            add_trend = st.radio("Trend Line?", ("No", "Yes"), key="reg_trend")

    # Filter data
    df_reg = data[
        data["grid"].isin(sel_grids) &
        data["season"].isin(sel_seasons) &
        data["datetime"].dt.year.isin(sel_years)
    ]

    if df_reg.empty:
        st.warning("No data matches current filters")
        return

    # Layout
    col1, col2 = st.columns([0.3, 0.7])
    with col1:
        st.plotly_chart(create_map(df_reg), use_container_width=True, config=config_figure,key="map_chart_regression")

    with col2:
        fig = go.Figure()
        
        # Generate unique colors for each station-season-year combination
        unique_combinations = set((row["grid"], row["season"], row["datetime"].year) 
                              for _, row in df_reg.iterrows())
        color_map = {combo: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] 
                    for i, combo in enumerate(unique_combinations)}

        for (station, season, year), group in df_reg.groupby(["grid", "season", df_reg["datetime"].dt.year]):
            color = color_map.get((station, season, year))
            legendgroup=f"{station}_{season}_{year}"
            # Scatter plot
            fig.add_trace(go.Scatter(
                x=group[x_var],
                y=group[y_var],
                mode="markers",
                showlegend=True,
                name=f"{station} <br> {season} <br> {year}",
                marker=dict(size=6, color=color, opacity=0.2),
                legendgroup=legendgroup,            ))

            if add_trend == "Yes":
                # Regression line
                X = sm.add_constant(group[x_var])
                model = sm.OLS(group[y_var], X).fit()
                x_range = np.linspace(group[x_var].min(), group[x_var].max(), 100)
                y_pred = model.predict(sm.add_constant(x_range))
                
                equation = f"y = {model.params[1]:.2f}x + {model.params[0]:.2f}<br>R² = {model.rsquared:.2f}"
                
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=y_pred,
                    mode="lines",
                    line=dict(color=color, dash="dash"),
                    showlegend=False,
                    hovertemplate=equation,
                    legendgroup=legendgroup,
                ))
                # Smart equation positioning
                x_pos = group[x_var].min() + 0.2 * (group[x_var].max() - group[x_var].min())
                y_pos = group[y_var].min() + 0.9 * (group[y_var].max() - group[y_var].min())
                
                # Equation annotation
                fig.add_trace(go.Scatter(
                    x=[x_pos],
                    y=[y_pos],
                    mode="text",
                    text=[equation],
                    textposition="top right",
                    textfont=dict(
                        color=color,
                        size=15,
                        family="Courier New, monospace"
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                    legendgroup=legendgroup
                ))

        fig.update_layout(
            title=f"{x_var} vs {y_var}",
            xaxis_title=x_var,
            yaxis_title=y_var,
            margin=dict(l=100, r=100, t=50, b=50),
            paper_bgcolor="white",
            plot_bgcolor="white",
            template="simple_white",
            legend=dict(groupclick="togglegroup"),
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True, config=config_figure,key="map_chart_regression2")

def render_correlation_tab(data):
    """Correlation Heatmap Tab"""
    st.header("Correlation Heatmap")
    
    options = get_unique_options(data)
    
    with st.expander("Filters", expanded=False):
        cols = st.columns(3)
        with cols[0]:
            sel_grid = st.selectbox("grid", options["grids"], index=0, key="heat_grid")
        with cols[1]:
            sel_season = st.selectbox("Season", options["seasons"], index=0, key="heat_season")
        with cols[2]:
            sel_year = st.selectbox("Year", options["years"], index=0, key="heat_year")

    # Filter data
    df_heat = data[
        (data["grid"] == sel_grid) &
        (data["season"] == sel_season) &
        (data["datetime"].dt.year == sel_year)
    ]

    if df_heat.empty:
        st.warning("No data matches current filters")
        return

    # Layout
    col1, col2 = st.columns([0.3, 0.7])
    with col1:
        st.plotly_chart(create_map(df_heat), use_container_width=True, config=config_figure,key="map_chart_correlation")

    with col2:
        fig = generate_correlation_heatmap(sel_grid, sel_season, sel_year, df_heat)
        fig.update_layout(
            margin=dict(l=70, r=200, t=50, b=50),
            paper_bgcolor="white", 
            plot_bgcolor="white"
        )
        st.plotly_chart(fig, use_container_width=True, config=config_figure,key="map_chart_correlation2")

def render_timeseries_tab(data):
    """Time Series Analysis Tab"""
    st.header("Time Series Analysis")
    
    if "datetime" not in data.columns:
        st.info("Skipping time series plot because no 'datetime' column is present.")
        return

    options = get_unique_options(data)
    
    with st.expander("Filters", expanded=False):
        cols = st.columns(3)
        with cols[0]:
            sel_stations = st.multiselect(
                "Select Station(s)",
                options=options["grids"],
                default=[options["grids"][0]],
            )
        with cols[1]:
            start_date = st.date_input("Start Date", value=data["datetime"].min().date())
        with cols[2]:
            end_date = st.date_input("End Date", value=data["datetime"].max().date())
        
        selected_vars = st.multiselect(
            "Select Variables",
            options=list(COLOR_SCHEMES.keys()),
            default=["temperature_c"],
            max_selections=2
        )

    # Filter data
    df_ts = data[
        (data["grid"].isin(sel_stations)) &
        (data["datetime"] >= pd.to_datetime(start_date)) &
        (data["datetime"] <= pd.to_datetime(end_date))
    ]

    if df_ts.empty:
        st.warning("No data matches current filters")
        return

    # Layout
    col1, col2 = st.columns([0.2, 0.7])
    with col1:
        st.plotly_chart(create_map(df_ts), use_container_width=True, config=config_figure, key="map_chart_timeseries")

    with col2:
        if not selected_vars:
            st.info("Please select at least one variable to display.")
            return

        # Calculate daily averages
        ts_data = df_ts.groupby(["datetime", "grid"])[selected_vars].mean().reset_index()

        fig = go.Figure()
        
        # Generate colors for stations
        station_colors = {station: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] 
                         for i, station in enumerate(ts_data["grid"].unique())}

        for i, var in enumerate(selected_vars):
            for station in ts_data["grid"].unique():
                station_data = ts_data[ts_data["grid"] == station]
                fig.add_trace(go.Scatter(
                    x=station_data["datetime"],
                    y=station_data[var],
                    mode="lines+markers",
                    name=f"{station} <br> {var}",
                    showlegend=True,
                    line=dict(
                        color=station_colors[station],
                        dash=DASH_PATTERNS[i % len(DASH_PATTERNS)]
                    ),
                    yaxis="y" if i == 0 else "y2"
                ))

        fig.update_layout(
            title="Time Series of Selected Variables",
            xaxis_title="Date",
            yaxis_title=selected_vars[0],
            yaxis2=dict(
                title=selected_vars[1] if len(selected_vars) > 1 else "",
                overlaying="y",
                side="right"
            ) if len(selected_vars) > 1 else None,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.3,
                xanchor="center",
                x=0.5
            ),
            margin=dict(t=60, r=10, b=120, l=40)
        )
        st.plotly_chart(fig, use_container_width=True, config=config_figure,key="map_chart_timeseries2")

# ------------------------------
# Main Application
# ------------------------------
def main():
    # Load and validate data
    if 'data' not in st.session_state:
        st.session_state.data = load_data()
    
    data = st.session_state.data

    if data is None:
        st.error("Data not found. Please load data in the main application.")
        st.stop()
    # Preprocess data
    data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
    data["latitude"] = -data["latitude"].abs()

    # Render UI
    render_header()

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "CTD Profiles",
        "Regression Diagram", 
        "Correlation Heatmap",
        "Timeseries"
    ])

    with tab1:
        render_ctd_tab(data)
    with tab2:
        render_regression_tab(data)
    with tab3:
        render_correlation_tab(data)
    with tab4:
        render_timeseries_tab(data)

if __name__ == "__main__":
    main()
