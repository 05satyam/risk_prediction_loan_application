'''
This script is designed to have a basic implementation as a base model
Later on the approach was moved to model_training itself, where we are first searching for any saved model to load if not found then loading a new one

'''
import data_ingestion
import pandas as pd 
import numpy as np 
import lightgbm as lgb 
from sklearn.model_selection import train_test_split 
import os
from dotenv import load_dotenv
import joblib

#loading environment variables
load_dotenv()
MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH')
DATA_CSV_PATH = os.getenv('LOAN_DATA_CSV_PATH')
TARGET_COLUMN_NAME = os.getenv('TARGET_COLUMN_NAME')

df=data_ingestion.load_data(DATA_CSV_PATH)

y = df[TARGET_COLUMN_NAME] 
X = df.drop(TARGET_COLUMN_NAME, axis=1) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = lgb.LGBMClassifier() 
model.fit(X_train, y_train) 
# Predict and evaluate 
predictions = model.predict(X_test) 
joblib.dump(model, MODEL_SAVE_PATH)