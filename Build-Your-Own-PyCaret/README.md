# Capstone Project: Automated Machine Learning Package and Web App

## Overview

In this capstone project, participants engage with PyCaret, an open-source, low-code machine learning library in Python designed to automate machine learning workflows. The primary aim is to create a general package that simplifies data handling and streamlines the machine learning process.

## Objectives

1. **Exploration of PyCaret:** Understand PyCaret's functionality fully, including loading data, conducting exploratory data analysis (EDA), training various machine learning models, evaluating their performance, and utilizing PyCaret's AutoML feature.
  
2. **Development of a General ML Package:** Develop a machine learning package that simplifies data handling, EDA, and model training. The package should automatically decide whether to use regressors or classifiers based on the data and allow users to select the models they wish to train.
  
3. **Creation of a User-Friendly Web App:** Utilize Streamlit to create a user-friendly web app. The web app should enable users to upload their data, select the target variable, and choose the machine learning models they want to use.

## Criteria

- The package must effectively load data, perform EDA, and train machine learning models, automatically selecting the correct type of model (regressor or classifier) for the data.
  
- The Streamlit web app should facilitate an easy and interactive way for users to upload their data, select a target variable, and choose which machine learning models to use.
  
- Thorough testing across various datasets to ensure consistent identification of the best machine learning model for each dataset and freedom from errors.
  
- Demonstration of a comprehensive understanding of PyCaret by effectively applying its features, especially the AutoML functionality for optimal model selection.
  
- Clear documentation and guidelines for using the package and web app should be provided to ensure users can fully utilize all available functionalities.

## Code Overview

The provided code includes a Streamlit web app for facilitating the machine learning process. The steps to get the algorithms' prediction accuracy are as follows:

1. **Upload Dataset:** Users can upload CSV or Excel files containing their data.
2. **Choose Target Feature:** Select the target variable for the machine learning models.
3. **Remove Unimportant Features:** Optionally remove features that are not significant for modeling.
4. **Preprocess Data:** Handle missing values and encode categorical variables.
5. **Compare and Create Model:** Select the type of machine learning task (Regression or Classification), compare different models, and create the best-performing model.

## Usage

1. Clone this repository to your local machine.
2. Install the required Python packages by running `pip install -r requirements.txt`.
3. Run the Streamlit app by executing `streamlit run app.py` in your terminal.
4. Follow the steps outlined in the Streamlit app interface to upload your dataset and select models.
