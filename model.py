import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
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
    print("\nDEBUG - train_model() - Train Ready DataLast few rows:\n", train_ready_data.tail())
    
    if model_type == "linear_regression":
        
        # Define feature columns once
        feature_cols = ['Open', 'High', 'Low', 'Prev Close', 'Volume']
 
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
        # if model_type == 'xgboost':
        #     model = xgb.XGBRegressor(
        #         n_estimators=200,
        #         learning_rate=0.05,
        #         max_depth=6,
        #         min_child_weight=1,
        #         subsample=0.8,
        #         colsample_bytree=0.8,
        #         random_state=42,
        #         n_jobs=-1
        #     )

        if model_type == 'xgboost':
            # model = xgb.XGBRegressor(
            #     n_estimators=200,
            #     learning_rate=0.01,  # Reduced from 0.05
            #     max_depth=4,         # Reduced from 6
            #     min_child_weight=3,  # Increased from 1
            #     subsample=0.7,       # Slightly reduced
            #     colsample_bytree=0.7,# Slightly reduced
            #     gamma=0.1,           # Added regularization
            #     reg_alpha=0.1,       # L1 regularization
            #     reg_lambda=0.1,      # L2 regularization
            #     random_state=42,
            #     n_jobs=-1
            #     )
            model = xgb.XGBRegressor(
                n_estimators=300,  # Increased from 200
                learning_rate=0.005,  # Further reduced
                max_depth=3,  # Reduced complexity
                min_child_weight=5,  # Increased to prevent overfitting
                subsample=0.6,  # More aggressive sampling
                colsample_bytree=0.6,  # Reduce feature sampling
                gamma=0.3,  # Increased regularization
                reg_alpha=0.3,  # L1 regularization
                reg_lambda=0.3,  # L2 regularization
                random_state=42,
                n_jobs=-1,
                early_stopping_rounds=10,  # Add early stopping
                eval_metric='rmse'  # Use RMSE as the evaluation metric
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
        
        # Get feature columns for non-linear models using the feature engineering utility
        feature_columns = get_feature_columns(train_ready_data, model_type)
        
        print(f"DEBUG - train_model() - Feature columns non-linear:\n{feature_columns}")

        # Separate features and target
        X = train_ready_data[feature_columns].iloc[:-1]  
        print(f"DEBUG - train_model() - X:\n{X}")
        y = train_ready_data[('Next_Day_Close', ticker_value)].iloc[:-1]  # Target is next day's close
        
        print(f"Training with {len(feature_columns)} features:")
        
        print("\nTechnical indicators:")
        print([col for col in feature_columns if col[0] not in ['Open', 'High', 'Low', 'Close', 'Prev Close', 'Volume']])
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
           
        # Fit the model
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        # Optional: Perform cross-validation
        # cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        # print(f"{model_type.capitalize()} Cross-Validation Scores:")
        # print("MSE Scores:", -cv_scores)
        # print("Mean CV Score:", -cv_scores.mean())

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []

        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            model_cv = xgb.XGBRegressor(
                n_estimators=300,
                learning_rate=0.005,
                max_depth=3,
                min_child_weight=5,
                subsample=0.6,
                colsample_bytree=0.6,
                gamma=0.3,
                reg_alpha=0.3,
                reg_lambda=0.3,
                random_state=42,
                n_jobs=-1,
                eval_metric='rmse'
            )
            
            model_cv.fit(X_train, y_train)
            y_pred = model_cv.predict(X_test)
            
            # Calculate MSE
            mse = np.mean((y_test - y_pred)**2)
            cv_scores.append(mse)

        print("Cross-validation MSE scores:", cv_scores)
        print("Mean CV Score:", np.mean(cv_scores))
        
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