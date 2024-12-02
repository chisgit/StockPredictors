import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def create_candlestick_chart(df, ticker):
    """
    Create an interactive candlestick chart using Plotly.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLC data
        ticker (str): Stock ticker symbol
    
    Returns:
        go.Figure: Plotly figure object
    """
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                        open=df['Open'],
                                        high=df['High'],
                                        low=df['Low'],
                                        close=df['Close'])])
    
    fig.update_layout(
        title=f'{ticker} Stock Price',
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        template='plotly_dark'
    )
    
    return fig

def plot_prediction_with_interval(dates, actual, predictions, lower_bound, upper_bound):
    """
    Plot actual values, predictions, and prediction intervals.
    
    Args:
        dates (array-like): Dates for x-axis
        actual (array-like): Actual stock prices
        predictions (array-like): Predicted stock prices
        lower_bound (array-like): Lower bound of prediction interval
        upper_bound (array-like): Upper bound of prediction interval
    
    Returns:
        go.Figure: Plotly figure object
    """
    fig = go.Figure()
    
    # Add actual values
    fig.add_trace(go.Scatter(
        x=dates,
        y=actual,
        name='Actual',
        line=dict(color='blue')
    ))
    
    # Add predictions
    fig.add_trace(go.Scatter(
        x=dates,
        y=predictions,
        name='Predicted',
        line=dict(color='red')
    ))
    
    # Add prediction interval
    fig.add_trace(go.Scatter(
        x=dates.tolist() + dates.tolist()[::-1],
        y=upper_bound.tolist() + lower_bound.tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255,0,0,0.2)',
        line=dict(color='rgba(255,0,0,0)'),
        name='Prediction Interval'
    ))
    
    fig.update_layout(
        title='Stock Price Prediction with Confidence Interval',
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        template='plotly_dark',
        showlegend=True
    )
    
    return fig

def plot_feature_importance(feature_names, importance_scores):
    """
    Create a bar plot of feature importance scores.
    
    Args:
        feature_names (list): Names of features
        importance_scores (array-like): Importance scores for each feature
    
    Returns:
        go.Figure: Plotly figure object
    """
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    })
    df = df.sort_values('Importance', ascending=True)
    
    fig = px.bar(
        df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance'
    )
    
    fig.update_layout(
        template='plotly_dark',
        showlegend=False
    )
    
    return fig

def plot_metrics_comparison(metrics_dict, ticker):
    """
    Create a bar plot comparing different metrics for a stock.
    
    Args:
        metrics_dict (dict): Dictionary of metric names and values
        ticker (str): Stock ticker symbol
    
    Returns:
        go.Figure: Plotly figure object
    """
    fig = go.Figure(data=[
        go.Bar(
            x=list(metrics_dict.keys()),
            y=list(metrics_dict.values()),
            text=[f'{value:.4f}' for value in metrics_dict.values()],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=f'Model Performance Metrics for {ticker}',
        yaxis_title='Value',
        xaxis_title='Metric',
        template='plotly_dark'
    )
    
    return fig