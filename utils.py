import pandas as pd
import streamlit as st

@st.cache_data
def load_data(data_path):
    df = pd.read_csv(data_path)
    return df
