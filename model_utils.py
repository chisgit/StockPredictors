import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_data(df, target_column, feature_columns, test_size=0.2, random_state=42):
    """
    Prepare data for model training by splitting into train/test sets and scaling features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Name of the target column
        feature_columns (list): List of feature column names
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    X = df[feature_columns]
    y = df[target_column]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance using MSE and R2 score.
    
    Args:
        model: Trained model object
        X_test: Test features
        y_test: True test values
        
    Returns:
        dict: Dictionary containing MSE and R2 metrics
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'r2': r2
    }

def calculate_prediction_interval(model, X, confidence=0.95):
    """
    Calculate prediction interval for model predictions.
    
    Args:
        model: Trained model object
        X: Input features
        confidence (float): Confidence level (default: 0.95)
        
    Returns:
        tuple: (predictions, lower_bound, upper_bound)
    """
    predictions = model.predict(X)
    
    # Calculate prediction standard error (simplified approach)
    residuals = model.predict(X) - predictions
    std_error = np.std(residuals)
    
    # Calculate z-score for the given confidence level
    z_score = abs(np.percentile(np.random.standard_normal(10000), (1 - confidence) * 100))
    
    # Calculate bounds
    lower_bound = predictions - z_score * std_error
    upper_bound = predictions + z_score * std_error
    
    return predictions, lower_bound, upper_bound