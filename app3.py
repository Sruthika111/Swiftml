import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import io

def convert_to_supervised(data, n_clusters):
    """
    Convert unsupervised dataset to supervised by assigning cluster labels.
    """
    data_array = data.values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data_array)

    supervised_data = data.copy()
    supervised_data['cluster_label'] = cluster_labels
    return supervised_data

st.title("Unsupervised to Supervised Converter using KMeans Clustering")

st.markdown("""
Upload your CSV dataset and choose the number of clusters. The app will perform KMeans clustering 
and append cluster labels to your data.
""")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"Dataset loaded successfully! Shape: {df.shape}")
        st.write("Preview of uploaded data:", df.head())
        
        n_clusters = st.number_input("Enter the number of clusters", min_value=1, max_value=20, value=3, step=1)

        if st.button("Convert to Supervised"):
            with st.spinner("Clustering in progress..."):
                result_df = convert_to_supervised(df, n_clusters)
                st.success("Clustering complete!")
                st.write("Preview of data with cluster labels:")
                st.dataframe(result_df.head())

                # Download button
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download clustered data as CSV",
                    data=csv,
                    file_name="clustered_data.csv",
                    mime="text/csv"
                )
    except Exception as e:
        st.error(f"Error processing file: {e}")
