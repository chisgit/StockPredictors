import streamlit as st
import time as time_module
from app import render_ui, display_results
from pipeline import execute_pipeline
from session_state import initialize_session_state

def main():
    # Initialize session state
    initialize_session_state()

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
        close_price_prediction = execute_pipeline(tickers)
        
        # Display results
        display_results(close_price_prediction)

# Run the main function
if __name__ == "__main__":
    main()