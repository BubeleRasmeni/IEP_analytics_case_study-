import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import gsw
from functions import generate_correlation_heatmap, config_figure

# ------------------------------
# Configuration & Constants
# ------------------------------
MLD_METHODS = {
    "Temperature Threshold (ΔT)": "temp",
    "Density Threshold (Δσₜ)": "density",
    "Hybrid (Temp + Density)": "hybrid"
}

COLOR_SCALES = {
    "Temperature": px.colors.sequential.thermal,
    "Density": px.colors.sequential.dense,
    "Viridis": px.colors.sequential.Plasma
}

DEFAULT_THRESHOLDS = {
    "temp": 0.5,      # °C
    "density": 0.125, # kg/m³
    "hybrid": 0.5     # °C
}

# -------------------------------
# Style Loading
# -------------------------------
#@st.cache_resource
def load_styles():
    with open("assets/style.css") as f:
        return f"<style>{f.read()}</style>"

st.markdown(load_styles(), unsafe_allow_html=True)

# ------------------------------
# Custom Header Implementation
# ------------------------------
def render_header():
    st.markdown("""
    <div class="header" style="display: flex; justify-content: space-between; font-family:Roboto; align-items: center; background-color: #408a93; padding: 0px;">
        <div style="font-size: 1.6rem; font-weight: bold; color: white; margin-left: 0px;">
             State of the Ocean: Mixed Layer Depth (MLD) Analysis
        </div>
        <div style="font-size: 1rem; font-weight: bold; font-family:Roboto; margin-right: 100px;margin-top: 10px">
            <a href="https://data.ocean.gov.za/" target="_blank" style="color: white; text-decoration: none;">
                Access Historical Data
            </a>
        </div>
    </div>
    <div style="text-align: left; font-size: 0.9rem; margin-left: 0px;margin-bottom:10px; font-family:Roboto; font-weight: bold; color: white; background-color: #408a93; padding: 0px;">
        Department of Forestry, Fisheries, and the Environment
    </div>
    """, unsafe_allow_html=True)

# ------------------------------
# Data Loading & Preparation
# ------------------------------
@st.cache_data
def load_data():
    data = st.session_state.get("data")
    if data is None:
        st.error("Please load data from the main page")
        st.stop()
    
    data["latitude"] = -data["latitude"].abs()
    data["datetime"] = pd.to_datetime(data["datetime"])
    return data.dropna(subset=["temperature_c", "salinity_psu", "depth_m"])

# ------------------------------
# Core MLD Calculations
# ------------------------------
#@st.cache_data
def calculate_mld(df, method="temp", threshold=0.5):
    if df.empty or len(df) < 10:
        return np.nan
        
    try:
        surface = df.iloc[0]
        df = df.sort_values("depth_m")
        
        if method == "temp":
            t_diff = surface["temperature_c"] - threshold
            mld_row = df[df["temperature_c"] <= t_diff].first_valid_index()
            
        elif method in ["density", "hybrid"]:
            if "salinity_psu" not in df.columns:
                raise ValueError("Salinity data required for density calculations")
                
            df["density"] = gsw.sigma0(df["salinity_psu"], df["temperature_c"])
            surface_density = df["density"].iloc[0]
            
            if method == "density":
                mld_row = df[df["density"] >= surface_density + threshold].first_valid_index()
            else:  # hybrid
                t_diff = surface["temperature_c"] - threshold
                temp_mask = df["temperature_c"] <= t_diff
                density_mask = df["density"] >= (surface_density + 0.125)
                mld_row = df[temp_mask & density_mask].first_valid_index()
            
        return df.loc[mld_row, "depth_m"] if mld_row else df["depth_m"].max()
    
    except Exception as e:
        st.error(f"MLD calculation error: {str(e)}")
        return np.nan

# ------------------------------
# Visualization Components
# ------------------------------

def create_map(df, color_scale="Viridis"):
    """Create interactive map with mean MLD color coding"""
    # Calculate mean MLD per station if multiple seasons selected
    if 'Season' in df.columns:
        df = df.groupby(['Station', 'Latitude', 'Longitude'], as_index=False).agg({
            'MLD (m)': 'mean',
            'Season': lambda x: ', '.join(sorted(set(x)))
        })
    
    fig = px.scatter_map(
        df, 
        lat="Latitude", 
        lon="Longitude",
        color="MLD (m)",
        hover_name="Station",
        hover_data={
            "Station": True,
            "MLD (m)": ":.1f",
            "Latitude": ":.2f",
            "Longitude": ":.2f",
        },
        color_continuous_scale=COLOR_SCALES[color_scale],
        zoom=5,
        height=600,
        map_style="open-street-map",
        center={"lat": -32.0, "lon": 17.0},
    )
        # Update marker style
    fig.update_traces(
        marker=dict(
            size=8,  # Increase size for better visibility
            symbol="circle", 
            opacity=0.9,
        ),
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        coloraxis_colorbar=dict(title="Mean MLD (m)")
    )
    return fig
def create_profile_plot(profiles, method,annotate_mld):
    fig = go.Figure()
    
    for idx, (station, season, year, mld, df) in enumerate(profiles):
        color = px.colors.qualitative.Plotly[idx % len(px.colors.qualitative.Plotly)]
        
        # Determine x-axis values based on method
        if method in ["density", "hybrid"]:
            if "density" not in df.columns:
                df["density"] = gsw.sigma0(df["salinity_psu"], df["temperature_c"])
            x_values = df["density"]
            x_title = "Density (kg/m³)"
        else:
            x_values = df["temperature_c"]
            x_title = "Temperature (°C)"
        
        # Main profile
        fig.add_trace(go.Scatter(
            x=x_values,
            y=df["depth_m"],
            mode="lines",
            name=f"{station} <br> {season} <br> {year}",
            line=dict(color=color, width=2),
            hovertemplate=f"{x_title.split(' ')[0]}: %{{x:.2f}}<br>Depth: %{{y}}m",
            legendgroup=station,
            
        ))
        
        # MLD line
        fig.add_trace(go.Scatter(
            x=[x_values.min(), x_values.max()],
            y=[mld, mld],
            mode="lines",
            line=dict(color=color, dash="dash", width=1.5),
            hovertemplate=f"MLD: {mld:.1f}m",
            showlegend=False,
            legendgroup=station
        ))
        # ➡️ Add annotation for MLD value
        if annotate_mld == "Yes":
            fig.add_trace(go.Scatter(
                x=[x_values.mean()],
                y=[mld],
                text=[f"{mld:.1f}m"],
                mode="text",
                textposition="top center",
                showlegend=False,
                textfont=dict(
                    color=color,
                    size=12,
                    family="Arial",
                ),
                hoverinfo='skip',
                legendgroup=station,
                zorder=5  # Bring text to the front
            ))

    # Set title based on method
    method_titles = {
        "temp": "Temperature Threshold Method (ΔT)",
        "density": "Density Threshold Method (Δσₜ)",
        "hybrid": "Hybrid Method (ΔT + Δσₜ)"
    }        
    fig.update_layout(
        yaxis=dict(autorange="reversed", title="Depth (m)"),
        title=f"MLD: {method_titles[method]}",
        xaxis=dict(title=x_title),
        hovermode="x unified",
        height=600,
        legend=dict(groupclick="togglegroup"),
    )
    return fig


def render_mld_timeseries(data):
    """MLD Timeseries Analysis with Filters"""    
    if "datetime" not in data.columns:
        st.info("Skipping timeseries plot - missing datetime column")
        return

    # Get default values
    min_date = data["datetime"].min().date()
    max_date = data["datetime"].max().date()
    all_stations = data["grid"].unique().tolist()
    
    # Filters
    with st.expander("Analysis Filters", expanded=False):
        cols = st.columns(3)
        with cols[0]:
            selected_stations = st.multiselect(
                "Select Stations",
                options=all_stations,
                default=all_stations[:2],
                key="ts_stations"
            )
        with cols[1]:
            start_date = st.date_input(
                "Start Date",
                value=min_date,
                min_value=min_date,
                max_value=max_date
            )
        with cols[2]:
            end_date = st.date_input(
                "End Date",
                value=max_date,
                min_value=min_date,
                max_value=max_date
            )
        
        method = st.selectbox(
            "MLD Calculation Method",
            ["Temperature (ΔT=0.5°C)", "Density (Δσ=0.125 kg/m³)"],
            index=0
        )

    # Convert method to parameters
    method_params = {
        "Temperature (ΔT=0.5°C)": ("temp", 0.5),
        "Density (Δσ=0.125 kg/m³)": ("density", 0.125)
    }[method]

    # Filter data
    filtered = data[
        (data["grid"].isin(selected_stations)) &
        (data["datetime"] >= pd.to_datetime(start_date)) &
        (data["datetime"] <= pd.to_datetime(end_date))
    ]

    if filtered.empty:
        st.warning("No data matches selected filters")
        return

    # Calculate MLD for all station-date combinations
    mld_data = []
    for (station, date), group in filtered.groupby(["grid", "datetime"]):
        try:
            profile = group.sort_values("depth_m")
            mld = calculate_mld(profile, method_params[0], method_params[1])
            if not np.isnan(mld):
                mld_data.append({
                    "Station": station,
                    "Date": date,
                    "MLD (m)": mld
                })
        except Exception as e:
            continue
    
    ts_df = pd.DataFrame(mld_data)
    
    if ts_df.empty:
        st.warning("No MLD values calculated for selected filters")
        return

    # Create plot
    fig = px.scatter(
        ts_df,
        x="Date",
        y="MLD (m)",
        color="Station",
        title=f"MLD Timeseries ({method})",
        labels={"MLD (m)": "Mixed Layer Depth (m)"},
        hover_data={"Station": True, "Date": "|%B %d, %Y", "MLD (m)": ":.1f"}
    ).update_traces(mode='lines+markers')
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="MLD (m)",
        hovermode="x unified",
        legend_title_text="Station",
        height=500
    )
    
    # Add range slider
    fig.update_xaxes(
        rangeslider_visible=False,
        rangeslider_thickness=0.05,
        rangeselector=dict(
            buttons=list([
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    st.plotly_chart(fig, use_container_width=True, config=config_figure)

    # Summary statistics
    with st.expander("Summary Statistics", expanded=False):
        cols = st.columns([2,2])
        # cols[0].metric("Mean MLD", f"{ts_df['MLD (m)'].mean():.1f}m")
        
        min_mld = ts_df.loc[ts_df['MLD (m)'].idxmin()]
        cols[0].metric(
            "Minimum MLD", 
            f"{min_mld['MLD (m)']:.1f}m",
            f"{min_mld['Station']} {min_mld['Date'].strftime('%Y-%m-%d')}"
        )
        
        max_mld = ts_df.loc[ts_df['MLD (m)'].idxmax()]
        cols[1].metric(
            "Maximum MLD", 
            f"{max_mld['MLD (m)']:.1f}m",
            f"{max_mld['Station']} {max_mld['Date'].strftime('%Y-%m-%d')}"
        )
# ------------------------------
# Main Application
# ------------------------------
def main():
    data = load_data()
    render_header()
    # Create tabs
    tab1, tab2 = st.tabs(["CTD Profiles MLD", "Station MLD Timeseries"])
    
    with tab1:
        # Set default selections
        default_grid = data["grid"].unique()[0] if len(data["grid"].unique()) > 0 else ""
        default_season = data["season"].unique()[0] if len(data["season"].unique()) > 0 else ""
        default_year = data["datetime"].dt.year.unique()[0] if len(data["datetime"].dt.year.unique()) > 0 else ""

        with st.expander("Analysis Controls", expanded=False):
            cols = st.columns(3)
            
            with cols[0]:
                method = st.selectbox("MLD Method", list(MLD_METHODS.keys()))
                threshold = st.slider("Threshold Value", 0.1, 2.0, 
                                    DEFAULT_THRESHOLDS[MLD_METHODS[method]], 0.05)
                
            with cols[1]:
                grids = st.multiselect("Stations", data["grid"].unique(), default=[default_grid])
                seasons = st.multiselect("Seasons", data["season"].unique(), default=[default_season])
                
            with cols[2]:
                years = st.multiselect("Years", sorted(data["datetime"].dt.year.unique()), default=[default_year])
                color_scale = st.selectbox("Color Scale", ["Temperature"] if MLD_METHODS[method] == "temp" else ["Density", "Viridis"])
            annotate_mld = st.selectbox("Annotate MLD Values?", ["Yes", "No"], index=1)
            show_table = st.checkbox("Show MLD Data Table", False)

        # Process data
        filtered = data[
            data["grid"].isin(grids) &
            data["season"].isin(seasons) &
            data["datetime"].dt.year.isin(years)
        ]
        
        if filtered.empty:
            st.warning("No data matching selected filters")
            return
        
        # Calculate MLD
        profiles = []
        mld_data = []
        calculation_errors = 0
        
        for (station, season, year), group in filtered.groupby(["grid", "season", filtered["datetime"].dt.year]):
            try:
                profile = group.sort_values("depth_m")
                mld = calculate_mld(profile, MLD_METHODS[method], threshold)
                
                if not np.isnan(mld):
                    profiles.append((station, season, year, mld, profile))
                    mld_data.append({
                        "Station": station,
                        "Season": season,
                        "Year": year,
                        "MLD (m)": mld,
                        "Latitude": profile["latitude"].mean(),
                        "Longitude": profile["longitude"].mean()
                    })
                else:
                    calculation_errors += 1
            except:
                calculation_errors += 1
        
        mld_df = pd.DataFrame(mld_data)
        
        if calculation_errors > 0:
            st.warning(f"Could not calculate MLD for {calculation_errors} profiles")

        # Main layout
        if not mld_df.empty:
            col1, col2 = st.columns([4, 6])
            with col1:
                st.plotly_chart(create_map(mld_df, color_scale), use_container_width=True, config=config_figure)
            
            with col2:
                if profiles:
                    st.plotly_chart(create_profile_plot(profiles, MLD_METHODS[method],annotate_mld), use_container_width=True, config=config_figure)
            if show_table:
                st.dataframe(mld_df.sort_values(["Year", "Season"]), height=300, use_container_width=True)
        else:
            st.warning("No valid MLD calculations for selected filters")

    with tab2:
        render_mld_timeseries(data)
            
if __name__ == "__main__":
    main()