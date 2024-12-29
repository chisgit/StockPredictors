import streamlit as st
import numpy as np
import time as time_module
from datetime import timedelta
from utils import get_nyse_date, market_status, get_nyse_datetime
from data_handler import fetch_data, fetch_features
from data_processor import preprocess_data, preprocess_non_linear_data
from model import train_model

def execute_pipeline(tickers):
    # Initialize variables before try block
    close_predictions = []
    next_day_close_predictions = []
    train_ready_data = None
    predictions = [close_predictions, next_day_close_predictions]
    
    status = market_status()
    print(f"Market status: {status}")
    
    for each_ticker in tickers:
        st.markdown(f"Processing {each_ticker}...")
        print(f"\nDEBUG - execute_pipeline() - Starting processing for {each_ticker}")

        try:
            # Get training data using NYSE timezone
            nyse_date = get_nyse_date()
            stock_data = fetch_data(each_ticker, nyse_date + timedelta(days=1))
            
            train_ready_data = preprocess_data(stock_data)
            
            # Today's close prediction
            # Train model using linear regression
            model_type = "linear_regression"
            if model_type == "linear_regression":
                model = train_model(train_ready_data, model_type)
                print("DEBUG - execute_pipeline() - Model trained successfully")
                
                # Get prediction features and make prediction
                feature_cols = fetch_features(each_ticker, model_type)
                print("DEBUG - execute_pipeline() - Feature columns:", feature_cols)
                
                #prediction_data = train_ready_data[feature_cols].iloc[-1:].values
                prediction_data = fetch_features(each_ticker, model_type)
                
                print("DEBUG - execute_pipeline() - Prediction data shape:", prediction_data.shape)
                print("DEBUG - execute_pipeline() - Prediction data:", prediction_data)

                prediction = model.predict(prediction_data)
                print("DEBUG - execute_pipeline() - Raw prediction:", prediction)
                print("DEBUG - execute_pipeline() - Raw prediction type:", type(prediction))
                
                # Convert prediction to scalar if needed
                if isinstance(prediction, (list, np.ndarray)):
                    prediction = prediction.flatten()[0] if isinstance(prediction, np.ndarray) else prediction[0]
                print("DEBUG - execute_pipeline() - Final prediction value:", prediction)
                print("DEBUG - execute_pipeline() - Final prediction type:", type(prediction))
                
                close_predictions.append((each_ticker, prediction))

                # Next day close prediction
                # Train model using XGBoost
                model_type = "xgboost"
                
                # if model_type == "xgboost":
                #     train_ready_data = preprocess_non_linear_data(train_ready_data)
                # model = train_model(train_ready_data, model_type)

                # # Get prediction features and make prediction
                # prediction_features = fetch_features(each_ticker, model_type)
                # prediction = model.predict(prediction_features)

                # # Convert prediction to scalar if needed
                # if isinstance(prediction, (list, np.ndarray)):
                #     prediction = prediction.flatten()[0] if isinstance(prediction, np.ndarray) else prediction[0]
                #     next_day_close_predictions.append((each_ticker, prediction))
    
        except Exception as e:
            print(f"Error processing {each_ticker}: {e}")
            import traceback
            print("Full traceback:", traceback.format_exc())
            close_predictions.append((each_ticker, None))
            continue
        
    return predictions
