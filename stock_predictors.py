import streamlit as st
from render_ui import render_ui, display_results
from pipeline import execute_pipeline
from session_state import initialize_session_state
from data_handler import debug_yfinance_cache_location
from trace_utils import trace_event

def main():
    # Initialize session state
    initialize_session_state()
    st.session_state.trace_pass = st.session_state.get("trace_pass", 0) + 1
    trace_event(
        "main.enter",
        pass_no=st.session_state.trace_pass,
        run_prediction=st.session_state.get("run_prediction"),
        selected=st.session_state.get("selected_tickers"),
        tickers=st.session_state.get("tickers"),
        new_ticker=st.session_state.get("new_ticker"),
        has_last_predictions=st.session_state.get("last_predictions") is not None,
    )
    debug_yfinance_cache_location()

    # Render UI immediately after initialization
    trace_event("main.before_render_ui", pass_no=st.session_state.trace_pass)
    render_ui()
    trace_event(
        "main.after_render_ui",
        pass_no=st.session_state.trace_pass,
        run_prediction=st.session_state.get("run_prediction"),
        selected=st.session_state.get("selected_tickers"),
        new_ticker=st.session_state.get("new_ticker"),
        has_last_predictions=st.session_state.get("last_predictions") is not None,
    )

    # Check if prediction should be run
    if hasattr(st.session_state, 'run_prediction') and st.session_state.run_prediction:
        trace_event("main.prediction_requested", pass_no=st.session_state.trace_pass)
        # Reset the flag
        st.session_state.run_prediction = False

        # Increment the key to force new containers on next render
        st.session_state.results_key += 1
        trace_event(
            "main.prediction_state_reset",
            pass_no=st.session_state.trace_pass,
            results_key=st.session_state.results_key,
        )

        # Get selected tickers
        tickers = st.session_state.selected_tickers
        trace_event("main.before_execute_pipeline", tickers=tickers)

        # Execute pipeline with per-ticker loading status
        with st.status("Running predictions...", expanded=True) as status:
            predictions, skipped_tickers = execute_pipeline(tickers)
            status.update(
                label=f"Predictions ready for {', '.join(tickers)}",
                state="complete",
                expanded=False,
            )
        trace_event(
            "main.after_execute_pipeline",
            close_count=len(predictions[0]) if predictions else None,
            next_count=len(predictions[1]) if predictions else None,
            skipped_count=len(skipped_tickers),
        )

        # Cache predictions and skipped tickers
        st.session_state.last_predictions = predictions
        st.session_state.last_skipped_tickers = skipped_tickers

        # Display results
        trace_event("main.before_display_results", source="fresh_prediction")
        display_results(predictions, skipped_tickers)
    elif st.session_state.get('last_predictions') is not None:
        # Re-render cached predictions
        skipped_tickers = st.session_state.get('last_skipped_tickers', [])
        trace_event("main.before_display_results", source="cached_prediction")
        display_results(st.session_state.last_predictions, skipped_tickers)
    else:
        trace_event("main.idle", pass_no=st.session_state.trace_pass)

# Run the main function
if __name__ == "__main__":
    main()
