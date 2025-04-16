import streamlit as st
import pandas as pd
from pycaret.classification import setup as setup_clf
from pycaret.classification import compare_models as compare_clfs
from pycaret.classification import evaluate_model as evaluate_clfs
from pycaret.classification import pull as pull_clf
from pycaret.regression import setup as setup_reg
from pycaret.regression import compare_models as compare_reg
from pycaret.regression import evaluate_model as evaluate_reg
from pycaret.regression import pull as pull_reg
import pickle as pkl
from pathlib import Path
st.set_page_config("SwiftML")

st.title("SwiftML")
task=st.sidebar.radio("Select task",["Data Ingestion","Model Selection"])
if task=="Data Ingestion":
    file=st.file_uploader("Upload your file")
    if file is not None:
        df=pd.read_csv(file)
        st.dataframe(df)
        df.to_csv("sourcefile.csv",index=False)
elif task=="Model Selection":
    file=pd.read_csv("sourcefile.csv")
    task=st.selectbox("Task",options=['Regression','Classification'],label_visibility="hidden")
    target=st.selectbox("Select your target",file.columns)
    if st.button("Run"):
        if task=="Regression":
            setup_df=setup_reg(file,target=target)
            st.write("**SETUP**")
            st.dataframe(pull_reg())
            best=compare_reg()
            st.write("Performance of Models")
            st.dataframe(pull_reg())
            best=evaluate_reg(best)
            if st.button("Download model"):
                pkl.dumps(Path("D:\model.pkl"))
