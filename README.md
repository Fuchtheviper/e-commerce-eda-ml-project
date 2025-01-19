# e-commerce-eda-ml
This repository implements an end-to-end machine learning pipeline for revenue prediction using e-commerce transaction data. It includes data preprocessing, exploratory data analysis (EDA), hyperparameter tuning, model training, and evaluation.

---

## Table of Contents
    1. [Requirements](#requirements)
    2. [Setup](#setup)
    3. [How to Run](#how-to-run)
    4. [File Descriptions](#file-descriptions)
    5. [Outputs](#outputs)

---

## Requirements

    - Python 3.9 or higher
    - Pipenv (for managing Python dependencies)

---

## Setup

1. Clone this repository:
   ```bash
    git clone https://github.com/your-repo/e-commerce-eda-ml.git
    cd e-commerce-eda-ml

2. Install dependencies using pipenv:
    pipenv install
    pipenv shell

---

## How to run
1. Run EDA notebook to explore and get example data
    for EDA to explore the data

        jupyter notebook eda/eda_analysis.ipynb
    ** This process will get the example data 'data\processed_data\exported_data.csv'

2. Run ML pipeline
    for ML pipeline
        python main.py
    ** This process will get the model and predict value in return

    Remark : The ml notebook is just to show how this model was built

---

## File Descriptions
    - Data Folder
    data/processed_data/exported_data.csv: Input data for the pipeline.
    - EDA
    eda/eda_analysis.ipynb: Jupyter Notebook for analyzing the input dataset.
    - ML Scripts
    ml/scripts/data_preprocessing.py: Prepares the dataset for training, including missing value handling, scaling, and creating lagged features.
    ml/scripts/evaluate_model.py: Evaluates the trained model's performance using metrics such as MAE and RMSE.
    ml/scripts/hyperparameter_tuning.py: Automates hyperparameter tuning with Keras Tuner.
    ml/scripts/predict.py: Generates predictions using the trained model.
    ml/scripts/train_model.py: Trains the LSTM model with the given data.
    Main Script
    main.py: Combines all steps of the pipeline.

---

## Outputs
    Models:
        Trained LSTM models are saved in ml/models/.
        Hyperparameter tuning results are saved in ml/tuning_results/.

    Evaluation:
        Prints metrics like MAE and RMSE after model evaluation.

    Graphs:
        Training vs. validation loss.
        Actual vs. predicted revenue graph.

---

## Notes
    Make sure the input dataset is located at data/processed_data/exported_data.csv.
    Customize hyperparameter tuning settings in ml/scripts/hyperparameter_tuning.py.
    Check the ml_notebook.ipynb for detailed ML experimentation and insights.