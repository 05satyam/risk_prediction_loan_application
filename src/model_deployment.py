import shutil
from utils import *
import streamlit as st

'''
This script is designed for deployment stage
I have not implemened proper deployment 
I am just mentioning the stage to complete the process
We can deploy on inhouse servers/cloud servers as per requirements

'''
def deploy_model():
    source = get_path_to_save_and_load_model()
    #write the deployment code after loading the trained model

    st.write("Model deployed to production - DEMO ONLY")
