import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from pycaret.regression import RegressionExperiment
from pycaret.classification import ClassificationExperiment

# Function to perform EDA
def perform_eda(data):
    st.header("Exploratory Data Analysis (EDA)")
    analyze_data = st.checkbox("Perform EDA?")
    if analyze_data:
        columns_to_analyze = st.multiselect("Select columns for analysis:", options=data.columns)
        if columns_to_analyze:
            # Display histograms
            st.subheader("Histograms")
            for col in columns_to_analyze:
                plt.figure(figsize=(8, 6))
                sns.histplot(data[col], kde=True)
                plt.title(f"Histogram for {col}")
                plt.xlabel(col)
                plt.ylabel("Frequency")
                st.pyplot()
            # Display correlation matrix
            st.subheader("Correlation Matrix")
            corr = data[columns_to_analyze].corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
            plt.title("Correlation Matrix")
            st.pyplot()

# Function to encode categorical data
def encode_categorical(data):
    categorical_features = data.select_dtypes(include=['object']).columns
    encoding_method = st.radio("Select encoding method for categorical data:", ("Label Encoding", "One-Hot Encoding"))
    if encoding_method == "Label Encoding":
        for col in categorical_features:
            data[col] = LabelEncoder().fit_transform(data[col])
    

# Function to choose X and Y variables
def choose_variables(data):
    st.header("Choose X and Y variables")
    X_variables = st.multiselect("Select independent variables (X):", options=data.columns)
    Y_variable = st.selectbox("Select dependent variable (Y):", options=data.columns)
    return X_variables, Y_variable

# Main function to run the app
def main():
    st.sidebar.header("Steps to get the algorithms prediction accuracy")
    st.sidebar.text("1- Upload CSV or Excel file")
    st.sidebar.text("2- Choose target feature")
    st.sidebar.text("3- Remove unimportant features")

    data = pd.DataFrame()
    target = ""

    # Step 1: Upload dataset
    dataset = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])
    if dataset is not None:
        if "csv" in dataset.name:
            data = pd.read_csv(dataset)
        elif "xlsx" in dataset.name:
            data = pd.read_excel(dataset)
        st.write(data.head())
        st.write(data.shape)

        # Step 2: Choose target variable
        target = st.selectbox("Choose the target variable:", options=data.columns)

        # Step 4: Perform EDA
        perform_eda(data)

        # Step 5: Encode categorical data
        encode_categorical(data)

        # Step 3: Remove unimportant features
        select_columns = st.multiselect("Select features to remove from the dataframe:", options=data.columns)
        if select_columns:
            data.drop(select_columns, axis=1, inplace=True)

        # Step 6: Choose X and Y variables
        X_variables, Y_variable = choose_variables(data)

        # Step 7: Perform preprocessing if needed
        numerical_features = data.select_dtypes(['int64', 'float64']).columns
        categorical_feature = data.select_dtypes(['object']).columns
        missing_value_num = st.radio("Set missing value for numerical value 👇", ["mean", "median"])
        missing_value_cat = st.radio("Set missing value for categorical value 👇", ['most frequent', "put additional class"])

        for col in numerical_features:
            data[col] = SimpleImputer(strategy=missing_value_num, missing_values=np.nan).fit_transform(
                data[col].values.reshape(-1, 1))
        for col in categorical_feature:
            if data[col].nunique() > 7:
                data[col] = SimpleImputer(strategy='most_frequent', missing_values=np.nan).fit_transform(
                    data[col].values.reshape(-1, 1))
            else:
                data[col] = LabelEncoder().fit_transform(data[col])
        if (len(numerical_features) != 0):
            st.header("Numerical Columns")
            st.write(numerical_features)
        if (len(categorical_feature) != 0):
            st.header("Categorical columns")
            st.write(categorical_feature)
        if (len(categorical_feature) != 0 or len(numerical_features) != 0):
            st.header("Number of null values")
            st.write(data.isna().sum())

        # Step 8: Perform model comparison and prediction
        if target and X_variables and Y_variable:
            option = "Regression" if data[Y_variable].dtype in ['int64', 'float64'] else "Classification"
            st.header(f"Detected Task Type: {option}")

            if option == 'Regression':
                s = RegressionExperiment()
            elif option == 'Classification':
                s = ClassificationExperiment()

            s.setup(data, target=Y_variable, session_id=123)
            best = s.compare_models()
            st.header("Best Algorithm")
            st.write(best)
            st.write(s.evaluate_model(best))
            st.header("30 rows of Prediction")
            predictions = s.predict_model(best, data=data, raw_score=True)
            st.write(predictions.head(30))

if __name__ == "__main__":
    main()
