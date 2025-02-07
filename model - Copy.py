import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from feature_engineering import get_feature_columns

def train_model(train_ready_data, model_type="linear_regression", X_val=None, y_val=None):
    """
    Trains the model on the provided data.
    
    Args:
        train_ready_data (pd.DataFrame): Preprocessed training data
        model_type (str): Type of model to train
        X_val (pd.DataFrame): Validation features (optional)
        y_val (pd.Series): Validation target (optional)
    
    Returns:
        Trained machine learning model and scaler
    """
    print(f"Current model type: {model_type}\n")

    # Split the data into features and target
    feature_cols = get_feature_columns(train_ready_data.iloc[-1:], model_type)
    X = train_ready_data[feature_cols]
    y = train_ready_data[('Close')]  # Predict today's close
    
    # Ensure predictions are consistent with today's close
    if model_type == "linear_regression":
        y = train_ready_data[('Close')]  # Ensure correct target for today's close
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == "linear_regression":
        print(f"Train Model ()LINEAR REGRESSION- Shape of train_ready_data: \n {train_ready_data.shape}\n")
        print(f"Train Model ()LINEAR REGRESSION- Columns in train_ready_data: \n{train_ready_data.columns.tolist()}\n")
        
        # Select the last row
        last_row = train_ready_data.iloc[-1:]
        #print(f"Last row data:\n{last_row}")
        
        # Get feature columns
        feature_cols = get_feature_columns(last_row, "linear_regression")
        #print(f"LINEAR REGRESSION- Feature columns: {feature_cols}")
                
        # Prepare features and target
        X = train_ready_data[feature_cols]
        y = train_ready_data[('Close')]  # Predict today's close

        # Ensure X only includes the selected features
        #print(f"Shape of X for Linear Regression: {X.shape}")  # Debugging line
        
        # Scale features and target
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()
        
        print(f"\nDEBUG: Fitting the X_scaler in train_model function\n")
        X_scaled = X_scaler.fit_transform(X)
        print(f"Shape of X_scaled: {X_scaled.shape}")  # Debugging line
        print(f"X_scaled size: {X_scaled.size}")  # Debugging line
        print("X_scaled contents:")
        print(X_scaled[:5])  # Display the first 5 rows
        print(f"Shape of X_scaled: {X_scaled.shape}\n")  # Debugging line

        print(f"\nDEBUG: Fitting the y_scaler in train_model function\n")
        y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))
        
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        print(f"Shape of X_scaled for model training: {X_scaled.shape}")  # Debugging line

        y_scaled = pd.Series(y_scaled.flatten(), index=y.index)
        
        # Use TimeSeriesSplit instead of random split
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
            y_train, y_test = y_scaled.iloc[train_idx], y_scaled.iloc[test_idx]
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            cv_scores.append(score)
            
        print(f"\nLinear Regression CV Scores: {cv_scores}")
        print(f"Mean CV Score: {np.mean(cv_scores)}")
        
        # Train final model on all data
        model = LinearRegression()
        print(f"Shape of X_scaled just before model.fit training: {X_scaled.shape}")  # Debugging line
        model.fit(X_scaled, y_scaled)
        
        # Print model coefficients with feature names
        print("\nLinear Regression Coefficients:")
        coefficients = model.coef_.flatten() if len(model.coef_.shape) > 1 else model.coef_
        coef_dict = dict(zip(feature_cols, coefficients))
        for feature, coef in coef_dict.items():
            print(f"{feature}: {coef:0.4f}")
            
        return model, (X_scaler, y_scaler)
    
    elif model_type == "xgboost":
        print("Entering XGBoost model training section")
        print(f"XGBOOST- Shape of train_ready_data: {train_ready_data.shape}")
        print(f"XGBOOST- Columns in train_ready_data: {train_ready_data.columns.tolist()}")

        # Select the last row
        last_row = train_ready_data.iloc[-1:]
        #print(f"XGBOOST- Last row data:\n{last_row}")

        # Get features for XGBoost
        feature_cols = get_feature_columns(train_ready_data.iloc[-1:], "xgboost")
        #print("\nXGBoost Features:")
        #print(feature_cols)

        # Prepare features and target
        X = train_ready_data[feature_cols]
        y = train_ready_data[('Next_Day_Close')]  # Predict next day's close

        # Scale features
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()

        print(f"\nDEBUG: Fitting the X_scaler in train_model function\n")
        X_scaled = X_scaler.fit_transform(X)
        print(f"Shape of X_scaled: {X_scaled.shape}")
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        print(f"\nDEBUG: Fitting the y_scaler in train_model function\n")
        y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))
        y_scaled = pd.Series(y_scaled.flatten(), index=y.index)

        # Use TimeSeriesSplit instead of random split
        tscv = TimeSeriesSplit(n_splits=5)
        val_scores = []
        
        for train_idx, test_idx in tscv.split(X_scaled):
            # Split the data into training and validation sets
            X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
            y_train, y_val = y_scaled.iloc[train_idx], y_scaled.iloc[test_idx]
            
            # Configure the XGBoost model
            model = xgb.XGBRegressor(
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=5,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                random_state=42,
                eval_metric='rmse'
            )
            model.set_params(early_stopping_rounds=None) 

        # Debug: Print shapes of validation data
        print(f"\nDEBUG: Shape of X_val: {X_val.shape if X_val is not None else 'None'}")
        print(f"DEBUG: Shape of y_val: {y_val.shape if y_val is not None else 'None'}")

        # # Configure XGBoost model
        # model = xgb.XGBRegressor(
        #     n_estimators=1000,
        #     learning_rate=0.01,
        #     max_depth=5,
        #     min_child_weight=1,
        #     subsample=0.8,
        #     colsample_bytree=0.8,
        #     objective='reg:squarederror',
        #     random_state=42,
        #     eval_metric='rmse',
        #     verbose=True
        # )
        model.set_params(early_stopping_rounds=10)  # Set early_stopping_rounds

        if len(X_val) > 0 and len(y_val) > 0:
            model.fit(
                X_train, 
                y_train, 
                eval_set=[(X_val, y_val)], 
                verbose=False  # Disable progress logs for brevity
            )
            # Evaluate model performance on validation set
            val_score = model.evals_result()['validation_0']['rmse'][-1]
            val_scores.append(val_score)
        else:
            print("Skipping fold with no validation data.")
        # Fit the model using training data and validation data
        # if X_val is not None and y_val is not None and len(X_val) > 0 and len(y_val) > 0:
        #     print(f"\nDEBUG: Fitting the model with validation set\n")
        #     model.fit(
        #         X_train, 
        #         y_train, 
        #         eval_set=[(X_val, y_val)],  # Provide validation data
        #         verbose=True                # Optional: Print progress
        #     )
        # else:
        #     print("\nDEBUG: No validation set provided. Disabling early stopping.\n")
        #     model.fit(X_train, y_train)
        # Print the average validation RMSE

        # Initialize list to store validation scores
        

        if val_scores:
            print(f"Average Validation RMSE: {sum(val_scores) / len(val_scores):.4f}")
        else:
            print("No validation scores to report.")
        # Get feature importance
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        print("\nXGBoost Feature Importance:")
        print(feature_importance)

        # Make prediction for next day close
        next_day_close_prediction = model.predict(X_scaled.iloc[-1:])
        print(f"\nXGBoost Prediction for next day close: ${next_day_close_prediction[0]:.2f}\n")

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