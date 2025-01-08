import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

def walk_forward_validation(df, models, target_column, test_size=0.2, window_size=30):
    """
    Perform Walk-Forward Validation for time series data.
    
    Args:
        df (pd.DataFrame): Input DataFrame with multi-index columns
        models (dict): Dictionary of models to evaluate
        target_column (str): Column to predict
        test_size (float): Proportion of data used for testing
        window_size (int): Size of the training window
    
    Returns:
        dict: Performance metrics for each model
    """
    # Ensure DataFrame is sorted by index
    df = df.sort_index()
    
    # Extract ticker value
    ticker_value = df.columns[1][1]
    
    # Prepare data
    X = df.drop(columns=[(target_column, ticker_value)])
    y = df[(target_column, ticker_value)]
    
    # Compute split points
    total_samples = len(df)
    train_samples = int(total_samples * (1 - test_size))
    
    # Initialize results storage
    model_results = {name: {'mse': [], 'mae': [], 'r2': []} for name in models.keys()}
    
    # Walk-forward validation
    for start in range(train_samples, total_samples - window_size):
        # Training window
        train_end = start
        train_start = max(0, train_end - window_size)
        
        # Test window
        test_start = train_end
        test_end = min(test_start + window_size, total_samples)
        
        # Split data
        X_train = X.iloc[train_start:train_end]
        y_train = y.iloc[train_start:train_end]
        X_test = X.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end]
        
        # Train and evaluate each model
        for name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Compute metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Calculate R-squared only for linear regression
            r2 = r2_score(y_test, y_pred) if isinstance(model, LinearRegression) else None
            
            # Store results
            model_results[name]['mse'].append(mse)
            model_results[name]['mae'].append(mae)
            model_results[name]['r2'].append(r2)
    
    # Compute average metrics
    for name in models.keys():
        model_results[name]['avg_mse'] = np.mean(model_results[name]['mse'])
        model_results[name]['avg_mae'] = np.mean(model_results[name]['mae'])
        
        # Only compute average R-squared for linear regression
        if isinstance(list(models.values())[0], LinearRegression):
            model_results[name]['avg_r2'] = np.mean([r for r in model_results[name]['r2'] if r is not None])
    
    return model_results

def print_model_performance(model_results):
    """
    Print detailed performance metrics for each model.
    
    Args:
        model_results (dict): Results from walk_forward_validation
    """
    print("\nModel Performance Metrics:")
    for name, metrics in model_results.items():
        print(f"\n{name}:")
        print(f"  Average MSE:  {metrics['avg_mse']:.4f}")
        print(f"  Average MAE:  {metrics['avg_mae']:.4f}")
        
        # Only print R-squared if it exists
        if 'avg_r2' in metrics and metrics['avg_r2'] is not None:
            print(f"  Average R²:   {metrics['avg_r2']:.4f}")