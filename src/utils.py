import json

import os
from dotenv import load_dotenv

'''
A helper scipts to load any utility used across whole project
'''

# Load environment variables from .env file
load_dotenv()
DATA_CSV_PATH = os.getenv('DATA_CSV_PATH')
MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH')
CONFIG_JSON_PATH=os.getenv('CONFIG_JSON_PATH')

def get_default_data_file_from_env():
    return DATA_CSV_PATH



def load_config():
    base_path = os.path.dirname(os.path.realpath(__file__))
    abs_path = os.path.join(base_path, CONFIG_JSON_PATH)

    print("abs_path ", abs_path)
    """Load the configuration file."""
    with open(abs_path, 'r') as file:
        config = json.load(file)
    return config

def get_path_to_save_and_load_model():
    base_path = os.path.dirname(os.path.realpath(__file__))
    abs_path = os.path.join(base_path, MODEL_SAVE_PATH)
    return abs_path
