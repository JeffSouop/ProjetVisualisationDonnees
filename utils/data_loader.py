import pandas as pd
import numpy as np
import streamlit as st


@st.cache_data
def load_data(path="adult.csv"):
    df = pd.read_csv(path)
    # Nettoyer les espaces dans les colonnes texte
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    # Encoder la cible en binaire
    df["income_binary"] = (df["income"] == ">50K").astype(int)
    return df


def get_kpis(df):
    n_rows = len(df)
    n_cols = len(df.columns)
    missing_rate = df.isnull().mean().mean() * 100
    high_income_rate = df["income_binary"].mean() * 100
    return n_rows, n_cols, missing_rate, high_income_rate
