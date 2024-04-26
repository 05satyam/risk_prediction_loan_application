import streamlit as st
import pandas as pd
import os
from utils import *  
from data_preprocessing import preprocess_data, split_data
from model_training import train_model
from model_evaluation import evaluate_model, plot_confusion_matrix, plot_roc_curve
from model_deployment import deploy_model

'''
It the point which consolidates all the pipeline stages and call them one by one.
'''
def main():
    """
        1. Below implementation is the entry point of pipeline
        2. First we are loading the data from csv
        3. Next, preprocess the data
        4. Split the preprocess data
        5. Train the model
        6. Evaluate the model
        7. Save the model
        8. Display the plots(confusion and roc metric)
        
        NOTE: If you need to run this file uncomment the __main__ below
    """
    try:
        
        st.title('Loan Risk Prediction Model')
        
        config = load_config()
        
        # Option to choose data source 
        data_source = st.radio("Choose your data source:", ('Upload CSV File', 'Use Default Path'))

        if data_source == 'Upload CSV File':
            uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file)
            else:
                st.warning("Please upload a CSV file.")
                return  
        else:
            # Try to load data from the path specified in the environment
            if os.path.exists(get_default_data_file_from_env()):
                data = pd.read_csv(get_default_data_file_from_env())
                st.success("Data loaded successfully from the default path.")
            else:
                st.error(f"Failed to load data. Please check the default data path or upload a CSV file.")
                return

        if data is not None:
            # Display raw data
            st.write("## Raw Data")
            st.write(data.head())

            # Data preprocessing
            data_preprocessed = preprocess_data(data, config)
            st.write("## Preprocessed Data")
            st.dataframe(data_preprocessed.head())  # Display the first few rows of preprocessed data

            st.write("## Evaluation on new dataset loaded: ")
            st.write(" - Loading the saved model (if any).");
            st.write(" - Training the model to get the accuracy score.")
            # Data splitting
            X_train, X_test, y_train, y_test = split_data(data_preprocessed, config)

            # Train the model
            model = train_model(X_train, y_train, base_model_path=get_path_to_save_and_load_model())

            # Evaluate the model
            accuracy, roc_auc = evaluate_model(X_test, y_test, model)

            # Display evaluation metrics
            st.write("### Evaluation Metrics of new data")
            st.write(f" - **Accuracy:** {accuracy:.2f}")
            st.write(f" - **ROC-AUC:** {roc_auc:.2f}")

            # Conditionally save the model based on the accuracy threshold
            if accuracy >= 0.95:
                st.write("## Model meets the requirement on new dataset...")
                model_save_path = get_path_to_save_and_load_model()
                base_model_path = get_path_to_save_and_load_model()
                train_model(X_train, y_train, save_path=model_save_path, base_model_path=base_model_path)  # Retrain & save model
                st.success('### Model meets the performance threshold')
                st.write(f' Model is saved to location:  {model_save_path}.')
                deploy_model()
            else:
                st.error("Model does not meet the performance criteria (95% accuracy required).")

            # Plotting metrics
            fig_cm = plot_confusion_matrix(y_test, model.predict(X_test))
            fig_roc = plot_roc_curve(y_test, model.predict_proba(X_test)[:, 1])
            st.pyplot(fig_cm)
            st.pyplot(fig_roc)
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        
if __name__ == '__main__':
    main()
