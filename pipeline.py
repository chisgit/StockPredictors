import streamlit as st
import numpy as np
import time as time_module
from datetime import timedelta
from utils import get_nyse_date, market_status, get_nyse_datetime
from data_handler import fetch_data, fetch_features
from data_processor import preprocess_data
from model import train_model

def execute_pipeline(tickers):
    results = []
    status = market_status()
    print(f"Market status: {status}")
    
    for each_ticker in tickers:
        st.markdown(f"Processing {each_ticker}...")

        try:
            # Get training data using NYSE timezone
            nyse_date = get_nyse_date()
            stock_data = fetch_data(each_ticker, nyse_date + timedelta(days=1))
            train_ready_data = preprocess_data(stock_data)
            
            # Train model
            model_type = "linear_regression"
            model = train_model(train_ready_data, model_type)
            
            # Get prediction features and make prediction
            prediction_features = fetch_features(each_ticker)
            prediction = model.predict(prediction_features)
            
            # Convert prediction to scalar if needed
            if isinstance(prediction, (list, np.ndarray)):
                prediction = prediction.flatten()[0] if isinstance(prediction, np.ndarray) else prediction[0]
            
            results.append((each_ticker, prediction))

        except Exception as e:
            print(f"Error processing {each_ticker}: {e}")
            results.append((each_ticker, None))
            continue

    return results
