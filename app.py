from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import setup as regression_setup, compare_models as regression_compare_models, pull as regression_pull, save_model as regression_save_model
from pycaret.classification import setup as classification_setup, compare_models as classification_compare_models, pull as classification_pull, save_model as classification_save_model
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os 

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoML : Do EDA and build ML model without coding ")
    choice = st.radio("Navigation", ["Upload","EDA","Modelling", "Download"])
    st.info("This project application helps you explore your data and build ML models.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "EDA": 
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choice == "Modelling":
    # Choose between regression and classification
    problem_type = st.radio("Select Problem Type", ["Regression", "Classification"])

    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'): 
        if problem_type == "Regression":
            regression_setup(df, target=chosen_target)
            setup_df = regression_pull()
            st.dataframe(setup_df)
            best_model = regression_compare_models()
            compare_df = regression_pull()
            st.dataframe(compare_df)
            regression_save_model(best_model, 'best_model')

        elif problem_type == "Classification":
            classification_setup(df, target=chosen_target)
            setup_df = classification_pull()
            st.dataframe(setup_df)
            best_model = classification_compare_models()
            compare_df = classification_pull()
            st.dataframe(compare_df)
            classification_save_model(best_model, 'best_model')

            
if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")
