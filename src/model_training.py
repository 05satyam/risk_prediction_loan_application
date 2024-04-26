import lightgbm as lgb
import joblib
import streamlit as st
import os

''''
This script is key part as it consist of model trainin phase on the preprocessed data

Point to note: I have not considering multiple sets of hyperparams, I just used very basic from documentations of Lighbgm
'''
def train_model(X_train, y_train, params=None, save_path=None, base_model_path=None):
    """
    Train a LightGBM model using the provided training data, with an option to load and continue training a pre-existing model.

    Args:
        X_train (DataFrame): The feature set used for training the model.
        y_train (Series): The target variable for the model.
        params (dict, optional): Parameters for the LightGBM model. See default parameters in the function.
        save_path (str, optional): Path where the trained model should be saved.
        base_model_path (str, optional): Path to a pre-trained model which should be loaded and used as the base for further training.

    Returns:
        model: A trained LightGBM classifier.
    """
    # Default parameters for the LightGBM model if none provided
    if params is None:
        params = {
            'boosting_type': 'gbdt',       # Gradient Boosted Decision Trees, 
            'objective': 'binary',         # objective function the model is trying to optimize, here binary classification task 
            'metric': 'binary_logloss',    # binary logarithmic loss, measures the performance of a classification model 
            'num_leaves': 31,              # maximum number of leaves in one tree.
            'learning_rate': 0.05,         # rate at which the model learns
            'feature_fraction': 0.9,       # 90% of features are used for each tree.(tells the fraction of features to be randomly selected during training)
            'bagging_fraction': 0.8,       # fraction of the data to be used for each iteration
            'bagging_freq': 5,             # frequency of bagging 
            'verbose': 0                   # silent mode (no output).
        }

    if base_model_path:
        # Check if the model file exists before loading
        if os.path.exists(base_model_path):
            try:
                model = joblib.load(base_model_path)
                st.write(f" - Loading pre-trained model saved at: {base_model_path}")
            except Exception as e:
                st.error(f"Failed to load the model: {str(e)}")
                st.write(" - Loading new lightgbm model due to failure in loading pre-trained model")
                model = lgb.LGBMClassifier(**params)
        else:
            st.warning(f"Model file not found at {base_model_path}. Loading a new model instead.")
            model = lgb.LGBMClassifier(**params)
    else:
        st.write(" - Loading new lightgbm model")
        model = lgb.LGBMClassifier(**params)

    # Train the model
    model.fit(X_train, y_train)

    # Save the model if a save path is provided
    if save_path:
        joblib.dump(model, save_path)
        st.write(f" - Model is saved at :  {save_path}")

    return model
