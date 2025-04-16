import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DatasetPreprocessor:
    def __init__(self, output_dir="preprocessed_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def load_dataset(self, path: str) -> pd.DataFrame:
        if path.endswith('.csv'):
            return pd.read_csv(path)
        elif path.endswith('.json'):
            return pd.read_json(path)
        elif path.endswith(('.xlsx', '.xls')):
            return pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported format: {path}")

    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        strategies = {
            'mean': lambda x: x.fillna(x.mean(numeric_only=True)),
            'median': lambda x: x.fillna(x.median(numeric_only=True)),
            'mode': lambda x: x.fillna(x.mode().iloc[0]),
            'interpolate': lambda x: x.interpolate()
        }
        df[numeric_cols] = strategies[strategy](df[numeric_cols])

        string_cols = df.select_dtypes(include=['object', 'string']).columns
        for col in string_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'unknown'
                df[col] = df[col].fillna(mode_val)

        for col in df.columns:
            if 'gender' in col.lower():
                df[col] = df[col].fillna('unknown')

        return df

    def clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                df[col] = df[col].clip(lower=0).round().astype(int)
            elif pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].astype(str).str.strip().str.lower().replace({'nan': 'unknown', '': 'unknown'}).fillna('unknown')
                if 'gender' in col.lower():
                    df[col] = df[col].replace({
                        'm': 'male', 'male': 'male',
                        'f': 'female', 'female': 'female'
                    }).fillna('unknown')
        return df

    def scale_features(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if not numerical_cols.empty:
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        return df

    def split_dataset(self, df: pd.DataFrame, test_size: float = 0.2):
        return train_test_split(df, test_size=test_size, random_state=42)


st.title("📊 Dataset Preprocessing App")

uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "json", "xlsx", "xls"])

missing_strategy = st.selectbox("Select missing value strategy", ['mean', 'median', 'mode', 'interpolate'])
scaling_method = st.selectbox("Select feature scaling method", ['standard', 'minmax'])
test_size = st.slider("Test size (fraction)", 0.1, 0.5, 0.2, 0.05)
output_prefix = st.text_input("Output filename prefix", value="preprocessed")
output_dir = st.text_input("Output directory", value="preprocessed_data")

if uploaded_file:
    with st.spinner("Processing dataset..."):
        df = None
        try:
            suffix = uploaded_file.name.split('.')[-1]
            temp_path = f"temp_uploaded.{suffix}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            preprocessor = DatasetPreprocessor(output_dir=output_dir)
            df = preprocessor.load_dataset(temp_path)
            df = preprocessor.handle_missing_values(df, missing_strategy)
            df = preprocessor.clean_columns(df)
            df = preprocessor.scale_features(df, scaling_method)
            train_df, test_df = preprocessor.split_dataset(df, test_size)

            os.makedirs(output_dir, exist_ok=True)
            train_path = os.path.join(output_dir, f"{output_prefix}_train.csv")
            test_path = os.path.join(output_dir, f"{output_prefix}_test.csv")

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            st.success("✅ Preprocessing complete!")
            st.download_button("⬇️ Download Training Set", data=train_df.to_csv(index=False), file_name=f"{output_prefix}_train.csv", mime='text/csv')
            st.download_button("⬇️ Download Testing Set", data=test_df.to_csv(index=False), file_name=f"{output_prefix}_test.csv", mime='text/csv')
            st.dataframe(df.head())

            os.remove(temp_path)
        except Exception as e:
            st.error(f"❌ Error during preprocessing: {e}")
