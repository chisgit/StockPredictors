import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from feature_engineering import get_feature_columns
def train_model(train_ready_data, model_type):
    """
    Trains the model on the provided data.
    
    Args:
        train_ready_data (pd.DataFrame): Preprocessed training data
        model_type (str): Type of model to train
    
    Returns:
        Trained machine learning model
    """
    print("\nDEBUG - train_model() - First few rows:\n", train_ready_data.head())
    print("\nDEBUG - train_model() - Last few rows:\n", train_ready_data.tail())
    
    if model_type == "linear_regression":
        # Get the ticker value from the DataFrame
        ticker_value = train_ready_data.columns[1][1]
        
        # Separate features and target using correct column structure with MultiIndex
        X = train_ready_data[[('Open', ticker_value), 
                            ('High', ticker_value), 
                            ('Low', ticker_value), 
                            ('Volume', ticker_value), 
                            ('Prev Close', ticker_value)]]
        y = train_ready_data[('Close', ticker_value)]
        
        # Define feature columns once
        feature_cols = ['Open', 'High', 'Low', 'Volume', 'Prev Close']


        
        # Use them for both X and y
        X = train_ready_data[feature_cols]
        y = train_ready_data['Close']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Print model coefficients
        print("\nDEBUG - train_model() - Model coefficients:")
        print("Raw coefficients:", model.coef_)
        for feature, coef in zip(X.columns, model.coef_[0]):
            print(f"{feature}: {coef:.2f}")
        print(f"Intercept: {float(model.intercept_):.2f}")
        
        r2_score = model.score(X_test, y_test)
        print(f"R2 Score: {r2_score}")
        
        return model
    
    if model_type !=  "linear_regression":
        
        """
        Trains the model with all engineered features including today's close.
        """
        if model_type == 'xgboost':
            model = xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=100, 
                random_state=42, 
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(
                n_estimators=100, 
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Get the ticker value from the DataFrame
        ticker_value = train_ready_data.columns[1][1]
        
        # Get feature columns using the feature engineering utility
        feature_columns = get_feature_columns(train_ready_data, model_type)
        
        # Separate features and target
        X = train_ready_data[feature_columns].iloc[:-1]  
        y = train_ready_data[('Next_Day_Close', ticker_value)].iloc[:-1]  # Target is next day's close
        
        print(f"Training with {len(feature_columns)} features:")
        print("\nBase features:")
        print([col for col in feature_columns if col[0] in ['Open', 'High', 'Low', 'Close', 'Prev Close', 'Volume']])
        print("\nTechnical indicators:")
        print([col for col in feature_columns if col[0] not in ['Open', 'High', 'Low', 'Close', 'Prev Close', 'Volume']])
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Rest of the function remains the same as before
        
        
        # Fit the model
        model.fit(X_train, y_train)
        
        # Optional: Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        print(f"{model_type.capitalize()} Cross-Validation Scores:")
        print("MSE Scores:", -cv_scores)
        print("Mean CV Score:", -cv_scores.mean())
        
        return model

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