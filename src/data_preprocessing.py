import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from scipy.stats import boxcox
from sklearn.model_selection import train_test_split
import streamlit as st

'''
This is a preprocessing stage script
Data is sent here and preprocessd as per the analysis done by data team
As of now the processing done here are mentioned in detail inside src/ead_notebook/exploratory_data_analysis_initial_phase.ipynb
'''
def preprocess_data(df, config):
    """
    Parameters:
    df (DataFrame): The pandas DataFrame containing the preprocessed data to be saved.
    config : config.json data
    Returns: preprocessed dataframe {df}
  
    """


    # Select relevant columns
    df = df[['loanId', 'anon_ssn', 'payFrequency', 'apr', 'applicationDate',
             'originatedDate', 'nPaidOff', 'loanStatus', 'loanAmount',
             'originallyScheduledPaymentAmount', 'leadType']]

    # Handling missing values
    df.loc[df['nPaidOff'].isna(), 'nPaidOff'] = df['nPaidOff'].mode()[0]
    def loan_status_feature_mapping(application_status):
        labels = {0: ['Settled Bankruptcy', 'Charged Off'],
                1: ['Paid Off Loan', 'Settlement Paid Off']}
        
        for label, status in labels.items():
            if application_status in status:
                return label
            
    df['loan_application_status'] = df['loanStatus'].map(loan_status_feature_mapping)
    # Drop rows with no loan_application_status label (implying their status is in process)
    df = df[df['loan_application_status'].notna()].reset_index(drop=True)

    # Drop unused columns
    df.drop(columns=['loanStatus'], inplace=True)

    # Mapping ordinal encoding for payFrequency
    frequency_mapping = {'B': 0, 'I': 1, 'M': 2, 'S': 3, 'W': 4}
    df['payFrequency'] = df['payFrequency'].map(frequency_mapping)

    # One-hot encoding for categorical variables
    df = pd.get_dummies(df, columns=['leadType'])

    # Convert dates and calculate derived features
    df['applicationDate'] = pd.to_datetime(df['applicationDate'], format='ISO8601')
    df['originatedDate'] = pd.to_datetime(df['originatedDate'], format='ISO8601')
    df['time_since_last'] = df.sort_values(by=['anon_ssn', 'applicationDate']).groupby('anon_ssn')['applicationDate'].diff().dt.total_seconds() / (24 * 3600)
    df['time_since_last'] = df['time_since_last'].fillna(-1)
    df['time_to_get_loan_originated_status'] = (df['originatedDate'] - df['applicationDate']).dt.total_seconds() / 3600
    df.drop(columns=['applicationDate', 'originatedDate'], inplace=True)

    # Apply transformations to numeric features
    numeric_features = ['apr', 'nPaidOff', 'loanAmount', 'originallyScheduledPaymentAmount', 'time_since_last', 'time_to_get_loan_originated_status']
    numeric_transformer = Pipeline(steps=[
        ('scaler', MinMaxScaler())])

    # Configure preprocessing for categorical features if any remain
    categorical_features = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Combine transformations in a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # Apply preprocessing
    df_preprocessed = preprocessor.fit_transform(df)
    columns_to_drop = config['columns_to_drop']
    df = df.drop(columns=columns_to_drop)
    st.write("## Data pre-processing complete.")
    return df

def split_data(df, config):
    target_column = config['target_column']
    if isinstance(df, pd.DataFrame):
        X = df.drop(target_column, axis=1)
        y = df[target_column]
    else:
        # Handle the case where df might be a sparse matrix
        df = pd.DataFrame.sparse.from_spmatrix(df)
        X = df.drop(target_column, axis=1)
        y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)
