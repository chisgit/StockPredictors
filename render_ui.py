import streamlit as st
from render_helpers import (
    get_recent_data,
    group_predictions_by_ticker,
    extract_and_format_ticker_data,
    display_predictions,
    preds_sameline,
    create_grid_display,
    search_and_add_ticker,
)
from rules import UI_RULES
from display_market_status import display_market_status
from utils import market_status


def enforce_max_tickers():
    if len(st.session_state.selected_tickers) > UI_RULES["max_tickers"]:
        st.warning(
            f"Maximum {UI_RULES['max_tickers']} tickers can be selected at once."
        )
        st.session_state.selected_tickers = st.session_state.selected_tickers[
            : UI_RULES["max_tickers"]
        ]


def display_results(predictions):
    """Display latest market data and predictions for each ticker."""
    enforce_max_tickers()

    last_available_date = None
    todays_close_predictions = predictions[0]
    next_day_close_predictions = predictions[1]

    print(f"Today's Close and Next_Day Predictions: {predictions}")

    # Group predictions by ticker while preserving order
    grouped_predictions, grouped_next_day_predictions = group_predictions_by_ticker(
        todays_close_predictions, next_day_close_predictions
    )
    ticker_data = {}  # Cache for storing downloaded data per ticker

    for ticker, predictions in grouped_predictions.items():
        try:
            # Use the predictions for display
            # Download data once per ticker
            recent_data = get_recent_data(ticker)
            if recent_data is None:
                continue

            ticker_data[ticker] = recent_data

            # Checks if there's enough data AND if it's the first time
            # since last_available_date is set to None once
            if len(ticker_data[ticker]) >= 2 and last_available_date is None:
                last_available_date = ticker_data[ticker].index[-1].date()
                display_market_status(last_available_date)
                st.subheader("Today's Close Predictions")

        except Exception as e:
            st.error(f"Error in the downloading of current data for {ticker}: {str(e)}")
            continue

    # Process each ticker once for today's close
    for ticker, predictions in grouped_predictions.items():
        try:
            latest_data = ticker_data.get(ticker)

            if latest_data is None or len(latest_data) < 2:
                st.warning(
                    f"Insufficient data available for {ticker}. Need at least 2 days of data to show previous close."
                )
                continue

            # Extract and format ticker data
            formatted_data = extract_and_format_ticker_data(latest_data)
            if formatted_data is None:
                continue

            # Combine ticker header with predictions on same line
            predictions_html = preds_sameline(
                grouped_predictions[ticker], formatted_data["Close"]
            )

            # Replace the existing display logic with a call to display_predictions
            display_predictions(ticker, predictions_html)

            # Create grid display with appropriate close value based on market status
            grid_html = create_grid_display(
                formatted_data["Open"],
                formatted_data["High"],
                formatted_data["Low"],
                formatted_data["Prev Close"],
                formatted_data["Close"],
                formatted_data["Volume"],
            )
            st.markdown(grid_html, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error processing {ticker}: {str(e)}")
            continue

    # Display next day predictions if available
    if next_day_close_predictions:
        st.markdown("---")
        st.subheader("Next Day's Close Predictions")

        # Process each ticker for next day's predictions
        for ticker, predictions in grouped_next_day_predictions.items():

            if ticker in ticker_data:
                try:
                    latest_data = ticker_data[ticker]  # Use cached data

                    if not latest_data.empty and len(latest_data) >= 1:
                        current_close = (
                            latest_data["Close"].iloc[-1].item()
                        )  # Added .item()

                        # Use preds_sameline to format the predictions
                        predictions_html = preds_sameline(predictions, current_close)

                        # Display the predictions
                        display_predictions(ticker, predictions_html)

                except Exception as e:

                    st.error(
                        f"Error processing next day predictions for {ticker}: {str(e)}"
                    )
                    continue


def update_selected_tickers(change):
    print(f"[UPDATE] Change: {change}")  # This is the key "stock_multiselect"

    # Get the current multiselect state
    updated_sel_tickers = st.session_state.stock_multiselect

    # Update selected_tickers to match the multiselect state
    st.session_state.selected_tickers = updated_sel_tickers

    print(f"[AFTER MULTISELECT] Multiselect value: {st.session_state.selected_tickers}")


def render_ui():
    st.title("Stock Price Predictor")

    # Initialize default tickers if not already in session state
    if "tickers" not in st.session_state:
        st.session_state.tickers = UI_RULES["default_tickers"]

    if "selected_tickers" not in st.session_state:
        st.session_state.selected_tickers = UI_RULES["default_tickers"].copy()

    # Add warning if max tickers limit is reached
    if len(st.session_state.selected_tickers) >= UI_RULES["max_tickers"]:
        st.warning(f"Maximum {UI_RULES['max_tickers']} tickers can be selected")

    tickers = st.multiselect(
        "Select stocks to predict:",
        st.session_state.tickers,
        default=st.session_state.selected_tickers,
        key="stock_multiselect",
        on_change=update_selected_tickers,
        args=["stock_multiselect"],
        max_selections=UI_RULES["max_tickers"],
    )

    # Update selected tickers based on multiselect
    st.session_state.selected_tickers = tickers

    # Search bar for new tickers
    new_ticker = st.text_input(
        "Search for a stock ticker:",
        value=st.session_state.new_ticker,
        key="new_ticker_input",
    )

    # This clears out the ticker if there's a change in removing the ticker
    # this way it doesn't keep appearing in the search bar after it's been removed
    if new_ticker != st.session_state.new_ticker:
        st.session_state.new_ticker = new_ticker

    if new_ticker:
        search_and_add_ticker(new_ticker)
    st.session_state.new_ticker = ""

    # Create two distinct containers
    processing_container = st.container(
        key=f"processing_container_{st.session_state.results_key}"
    )
    results_container = st.container(
        key=f"results_container_{st.session_state.results_key}"
    )

    # Predict button and results area
    predict_button = st.button("Predict")

    if predict_button:
        # Set a flag in session state to indicate prediction is requested
        st.session_state.run_prediction = True
        # Trigger a rerun to allow the controller to execute the pipeline
        st.rerun()
