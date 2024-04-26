import pandas as pd
import streamlit as st

'''
This script is designed to consist the data ingestion stage
As of now we are loading from csv but later on more ingesting methods can be implemented
'''
def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        st.write("## Data loaded successfully.")
        return data
    except FileNotFoundError:
        st.error("Data CSV file not found.")
        return None