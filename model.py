import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from feature_engineering import get_feature_columns

def train_model(train_ready_data, model_type="linear_regression", target_column='Close'):
    """Trains the model on the provided data."""
    print(f"Current model type: {model_type}\n")
    
    ticker = train_ready_data.columns[0][1]
    feature_cols = get_feature_columns(model_type)

    # Prepare features and target
    X = train_ready_data[[(col, ticker) for col in feature_cols]]
    y = train_ready_data[(target_column, ticker)]

    # Split data into train, validation, and test sets (70%, 20%, 10%)
    # Use the most recent data for testing and validation
    test_size = int(len(X) * 0.1)
    val_size = int(len(X) * 0.2)
    
    # Split chronologically: past -> future
    # Training: oldest data
    # Validation: more recent data
    # Test: most recent data (excluding the very last row for prediction)
    X_train = X[:-test_size-val_size]
    y_train = y[:-test_size-val_size]
    
    X_val = X[-test_size-val_size:-test_size]
    y_val = y[-test_size-val_size:-test_size]
    
    X_test = X[-test_size:]
    y_test = y[-test_size:]
    
    print(f"\nTime series split sizes (oldest to newest):")
    print(f"Train: {len(X_train)} samples (oldest data)")
    print(f"Validation: {len(X_val)} samples (more recent data)")
    print(f"Test: {len(X_test)} samples (most recent data)")
    print(f"\nDate ranges:")
    print(f"Train: {X_train.index[0]} to {X_train.index[-1]}")
    print(f"Validation: {X_val.index[0]} to {X_val.index[-1]}")
    print(f"Test: {X_test.index[0]} to {X_test.index[-1]}")

    # Scale features and target using only training data
    X_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train.values.reshape(-1, 1))
    
    # Transform all sets
    X_train_scaled = X_scaler.transform(X_train)
    X_val_scaled = X_scaler.transform(X_val)
    X_test_scaled = X_scaler.transform(X_test)
    
    y_train_scaled = y_scaler.transform(y_train.values.reshape(-1, 1)).ravel()
    y_val_scaled = y_scaler.transform(y_val.values.reshape(-1, 1)).ravel()
    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

    if model_type == "linear_regression":
        model = LinearRegression()
        model.fit(X_train_scaled, y_train_scaled)
        
        # Evaluate on all sets
        train_score = model.score(X_train_scaled, y_train_scaled)
        val_score = model.score(X_val_scaled, y_val_scaled)
        test_score = model.score(X_test_scaled, y_test_scaled)
        
        print(f"\nLinear Regression Scores (R²):")
        print(f"Train: {train_score:.4f}")
        print(f"Validation: {val_score:.4f}")
        print(f"Test: {test_score:.4f}")

    else:  # XGBoost
        model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=5,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42
        )
        
        # Train with validation set for monitoring
        eval_set = [(X_train_scaled, y_train_scaled),
                   (X_val_scaled, y_val_scaled)]
        
        model.fit(X_train_scaled, y_train_scaled,
                 eval_set=eval_set,
                 eval_metric='rmse',
                 verbose=False)
        
        # Get predictions for all sets
        train_pred = model.predict(X_train_scaled)
        val_pred = model.predict(X_val_scaled)
        test_pred = model.predict(X_test_scaled)
        
        # Calculate RMSE for all sets
        train_rmse = np.sqrt(mean_squared_error(y_train_scaled, train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val_scaled, val_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test_scaled, test_pred))
        
        print(f"\nXGBoost RMSE Scores:")
        print(f"Train: {train_rmse:.4f}")
        print(f"Validation: {val_rmse:.4f}")
        print(f"Test: {test_rmse:.4f}")

    # Store evaluation metrics in model object for access in pipeline
    model.train_score_ = train_score if model_type == "linear_regression" else train_rmse
    model.val_score_ = val_score if model_type == "linear_regression" else val_rmse
    model.test_score_ = test_score if model_type == "linear_regression" else test_rmse

    return model, (X_scaler, y_scaler)

def feature_importance(model, X):
    """
    Extract and print feature importances for interpretable models.
    
    Args:
        model: Trained model
        X (pd.DataFrame): Feature matrix
    
    Returns:
        dict: Feature importance dictionary
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = X.columns
        feature_importance_dict = dict(zip(feature_names, importances))
        
        # Sort features by importance
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        print("\nFeature Importances:")
        for feature, importance in sorted_features:
            print(f"{feature}: {importance:.4f}")
        
        return dict(sorted_features)
    else:
        print("Model does not support feature importance.")
        return {}