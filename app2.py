import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from stock_predictors import market_status, execute_pipeline, display_market_status, display_results

def main():
    st.set_page_config(
        page_title="Stock Price Predictor",
        page_icon="📈",
        layout="wide"
    )

    # Initialize session state for selected tickers if not exists
    if 'selected_tickers' not in st.session_state:
        st.session_state.selected_tickers = []

    # Title and description
    st.title("📈 Stock Price Predictor")
    st.markdown("""
    This app predicts stock prices using machine learning. Select stocks from the S&P 500 to analyze.
    The model considers various technical indicators and historical data to make predictions.
    """)

    # Sidebar for stock selection
    st.sidebar.header("Stock Selection")
    
    # Get S&P 500 tickers (you might want to cache this)
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500_table = pd.read_html(sp500_url)[0]
    all_tickers = sp500_table['Symbol'].tolist()

    # Multi-select for stocks
    selected = st.sidebar.multiselect(
        "Select stocks to analyze (max 5):",
        all_tickers,
        max_selections=5,
        key="stock_multiselect",
        default=st.session_state.selected_tickers
    )

    # Display market status
    display_market_status()

    if selected:
        try:
            # Execute prediction pipeline
            results = execute_pipeline(selected)
            
            # Display results
            display_results(results)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please select stocks from the sidebar to begin analysis.")

if __name__ == "__main__":
    main()
