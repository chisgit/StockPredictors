import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from feature_engineering import get_feature_columns

def train_model(train_ready_data, model_type="linear_regression", X_val=None, y_val=None, target="today"):
    """
    Trains the model on the provided data.
    
    Args:
        train_ready_data (pd.DataFrame): Preprocessed training data with MultiIndex columns (feature, ticker)
        model_type (str): Type of model to train
        X_val (pd.DataFrame): Validation features (optional)
        y_val (pd.Series): Validation target (optional)
        target (str): 'today' or 'next_day' - determines which close price to predict
    """
    print(f"\nTraining {model_type} model for {target}'s close")
    
    # Get ticker from the data
    ticker = train_ready_data.columns[0][1]
    print(f"Training model for ticker: {ticker}")
    
    # Get features with target parameter
    feature_cols = get_feature_columns(model_type=model_type, target=target)
    X = train_ready_data[[(col, ticker) for col in feature_cols]]
    
    # Select target based on prediction goal
    if target == "today":
        y = train_ready_data[('Close', ticker)]
        print(f"Using Close price as target for {ticker}")
    else:  # next_day
        y = train_ready_data[('Next_Day_Close', ticker)]
        print(f"Using Next_Day_Close as target for {ticker}")
    
    # Scale features and target
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_scaled = X_scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))
    y_scaled = pd.Series(y_scaled.flatten(), index=y.index)

    # Use TimeSeriesSplit for validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []

    if model_type == "linear_regression":
        model = LinearRegression()
        
        # Cross validation
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train = X_scaled.iloc[train_idx]
            X_test = X_scaled.iloc[val_idx]
            y_train = y_scaled.iloc[train_idx]
            y_test = y_scaled.iloc[val_idx]
            
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            cv_scores.append(score)
            
        print(f"\nLinear Regression CV Scores: {cv_scores}")
        print(f"Mean CV Score: {np.mean(cv_scores)}")
            
    elif model_type == "xgboost":
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42,
            eval_metric='rmse',
            early_stopping_rounds=10
        )
        
        # Split data for final training
        train_size = int(len(X_scaled) * 0.8)
        X_train = X_scaled.iloc[:train_size]
        X_val = X_scaled.iloc[train_size:]
        y_train = y_scaled.iloc[:train_size]
        y_val = y_scaled.iloc[train_size:]
        
        # Train with validation
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Print feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(f"\nFeature importance for {ticker}:")
        print(importance)

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