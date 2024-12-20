import pandas as pd
import numpy as np
import streamlit as st

def preprocess_data(df):
    """
    Preprocess stock data for machine learning
    
    Args:
        df (pd.DataFrame): Input stock data DataFrame
    
    Returns:
        pd.DataFrame: Preprocessed stock data
    """
    # Check if data is empty
    if df.empty:
        st.warning("No data returned for the ticker. Skipping.")
        return df
    
    ticker_value = df.columns[1][1] #Get the ticker value from the dataframe

    # Add 'Prev Close' column by shifting 'Close' by 1 to the next row but ensuring to preserve the ticker value
    df[('Prev Close', ticker_value)] = df[('Close', ticker_value)].shift(1)
    df.dropna(inplace=True) #removes the first record in the data set (since there is no close for the previous row)

    return df