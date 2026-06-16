import streamlit as st
import time as time_module
from render_ui import render_ui, display_results
from pipeline import execute_pipeline
from session_state import initialize_session_state
from data_handler import debug_yfinance_cache_location

def main():
    # Initialize session state
    initialize_session_state()
    debug_yfinance_cache_location()

    # Render UI immediately after initialization
    render_ui()

    # Check if prediction should be run
    if hasattr(st.session_state, 'run_prediction') and st.session_state.run_prediction:
        # Reset the flag
        st.session_state.run_prediction = False

        # Increment the key to force new containers on next render
        st.session_state.results_key += 1

        # Wait for 2 seconds
        time_module.sleep(2)

        # Get selected tickers
        tickers = st.session_state.selected_tickers

        # Execute pipeline
        predictions, skipped_tickers = execute_pipeline(tickers)

        # Cache predictions and skipped tickers
        st.session_state.last_predictions = predictions
        st.session_state.last_skipped_tickers = skipped_tickers

        # Display results
        display_results(predictions, skipped_tickers)
    elif st.session_state.get('last_predictions') is not None:
        # Re-render cached predictions
        skipped_tickers = st.session_state.get('last_skipped_tickers', [])
        display_results(st.session_state.last_predictions, skipped_tickers)

# Run the main function
if __name__ == "__main__":
    main()
