# Risk Prediction for customer loan application

## Project Structure
project_root/
│
├── src/
│   ├── pickel
│   │       └── model.pkl
│   ├── eda_notebooks
│   │         └── exploratory_data_analysis_initial_phase.ipynb
│   ├── base_model.py
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── model_deployment.py
│   ├── model_evaluation.py
│   ├── model_training.py
│   ├── pipeline.py
│   ├── utils.py
│   ├── pipeline.ipynb
├── .env
├── .gitignore
├── config.json
├── app.py
├── requirements.txt
├── index.html
├── .streamlit
│       └── config.toml
├── pipeline_stages.png
├── data/
│     ├── loan.csv
│     ├── plots
│           ├── corelation_btwn_principal_apr_repayment_amt.png
│           ├── correlation_heatmap.png
│           ├── loan_application_status_with_repay_trends_bargraph.png
│           ├── relation_between_cleint_paying_freq_and_fraud_score.png
│           ├── turnover_clients.png
│       
└── project_output_images/
            ├── 1.png, 2.png, 3.png, 4.png


# Steps to run the project
 - brew install libomp {install as per your operating system- i was using macos}
 -  To run the application 
     - Go to root folder {project_root}
     - **run:** pip install -r requirements.txt
     - **run:** streamlit run app.py

 - Now If you want to run the piepline script, Please open pipeline.py and uncomment the __main__ and run the script

# Deatuils on Project structure:
 - **src/** : Contains all the Python scripts for data ingestion, preprocessing, model training, evaluation, deployment,
             pipeline(main class) utils, pickel file for model, EDA notebook.
     - **pickel/**: Directory to contains the saved (trained model)
     - **notebooks/** : Directory to contains the Exploratory Data Analysis(EDA) notebook to evaluate and understand data
     - **app.py** - The Streamlit application script for the web interface.
     - **utils.py** - Helper script to load env variables and get file paths for csv and model path
     - **pipeline,ipynb** - a notebook to run the and test pipeline block by block

 - **.env** : Set the environemnt variables for different paths
 - **config.py** : Helper script to store 
  - Which columns to remove during training {columns which are not at all needed}
  - Store the target column name

 - **requirements.txt** - Lists all the Python packages that need to be installed.
 - **data/** - Directory for storing input data and any processed datasets and plots generated during .ipynb notebook evaluations.
 - **project_output_images/** - This folder contains my output images of the web-ui and display outputs of running model 
                                following pipeline stages 
 - **index.html** : documentation html for intermal working of the pipeline

# Details on Pipeline:
 - Read index.html 

#### Possible error while setting up 
 - Streamlit axios error
     - Resolve:
         - Link: https://discuss.streamlit.io/t/axioserror-request-failed-with-status-code-403/38112
         - Created a new folder called .streamlit in root path and create a “config.toml” file inside it.
         - Inside config.toml
             ``` 
             [server]
              enableXsrfProtection = false
              enableCORS = false
              ```

