def group_predictions_by_ticker(todays_close_predictions, next_day_close_predictions):
    grouped_predictions = {}
    grouped_next_day_predictions = {}
    
    for ticker, prediction in todays_close_predictions:
        if ticker not in grouped_predictions:
            grouped_predictions[ticker] = []
        grouped_predictions[ticker].append(prediction)

    for ticker, prediction in next_day_close_predictions:
        if ticker not in grouped_next_day_predictions:
            grouped_next_day_predictions[ticker] = []
        grouped_next_day_predictions[ticker].append(prediction)

    return grouped_predictions, grouped_next_day_predictions