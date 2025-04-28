import streamlit as st
import pandas as pd
from datetime import datetime
from utils import load_data

# Page setup
apptitle = "IEP Analysis 🌊"
st.set_page_config(page_title=apptitle, page_icon="🌊", layout='wide')

# Load data into session state if not already there
if 'data' not in st.session_state:
    st.session_state.data = load_data()
#st.dataframe(st.session_state.data, use_container_width=True)
# Define the available pages
pages = {
    "Information": [
        st.Page("about.py", title="📄 About IEP"),
    ],
    "Data Exploration": [
        st.Page("data_exploration.py", title="📊 Essential Ocean Variables"),
    ],
    "Advanced Ocean Analysis": [
        st.Page("ts_diagrams.py", title="🌊 Temperature–Salinity Diagram"),
        st.Page("MLD_Analysis.py", title="📏 Mixed Layer Depth (m) Analysis"),
    ],
}

# Create the navigation
pg = st.navigation(pages)

# Run the selected page
pg.run()

# Dynamic values
year = datetime.now().year
app_name = "IEP Analysis"
powered_by = "Streamlit 🌊"

# Sidebar content
st.sidebar.header("Resources")
st.sidebar.markdown(
    """
- [Marine Information Management System](https://data.ocean.gov.za/)
- [DFFE Oceans and Coasts](https://www.dffe.gov.za/OceansandCoastsManagement)
- [Python](https://www.python.org/) (Getting Started with Python)
- [Streamlit](https://docs.streamlit.io/)
"""
)

# Add a dynamic footer
st.markdown(
    f"""
    <footer style='text-align: center; padding-top: 2rem; color: grey; font-size: 0.9em;'>
        Developed for {app_name} © {year} | Powered by {powered_by}
    </footer>
    """,
    unsafe_allow_html=True,
)
