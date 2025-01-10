import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from feature_engineering import get_feature_columns

def train_model(train_ready_data, model_type="linear_regression"):
    """
    Trains the model on the provided data.
    
    Args:
        train_ready_data (pd.DataFrame): Preprocessed training data
        model_type (str): Type of model to train
    
    Returns:
        Trained machine learning model and scaler
    """
    print(f"Current model type: {model_type}\n")

    if model_type == "linear_regression":
        print(f"Train Model ()LINEAR REGRESSION- Shape of train_ready_data: \n {train_ready_data.shape}\n")
        print(f"Train Model ()LINEAR REGRESSION- Columns in train_ready_data: \n{train_ready_data.columns.tolist()}\n")
        
        # Select the last row
        last_row = train_ready_data.iloc[-1:]
        print(f"Last row data:\n{last_row}")
        
        # Get feature columns
        feature_cols = get_feature_columns(last_row, "linear_regression")
        print(f"LINEAR REGRESSION- Feature columns: {feature_cols}")
                
        # Prepare features and target
        X = train_ready_data[feature_cols]
        y = train_ready_data[('Close')]  # Predict today's close

        # Ensure X only includes the selected features
        print(f"Shape of X for Linear Regression: {X.shape}")  # Debugging line
        
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
        tscv = TimeSeriesSplit(n_splits=6)
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
        print(f"XGBOOST- Last row data:\n{last_row}")
        
        # Get features for XGBoost
        feature_cols = get_feature_columns(train_ready_data.iloc[-1:], "xgboost")
        print("\nXGBoost Features:")
        print(feature_cols)
        
        # Prepare features and target
        X = train_ready_data[feature_cols]
        y = train_ready_data[('Next_Day_Close')]  # Predict next day's close
        
        # Scale features
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()
        
        print(f"\nDEBUG: Fitting the X_scaler in train_model function\n")
        X_scaled = X_scaler.fit_transform(X)
        print(f"Shape of X_scaled: {X_scaled.shape}")  # Debugging line
        print(f"X_scaled size: {X_scaled.size}")  # Debugging line
        print("X_scaled contents:")
        print(X_scaled[:5])  # Display the first 5 rows
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        print(f"\nDEBUG: Fitting the y_scaler in train_model function\n")
        y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))
        
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        y_scaled = pd.Series(y_scaled.flatten(), index=y.index)

        # Use TimeSeriesSplit for validation
        tscv = TimeSeriesSplit(n_splits=5)
        train_splits = []
        test_splits = []
        for train_idx, test_idx in tscv.split(X_scaled):
            train_splits.append(train_idx)
            test_splits.append(test_idx)

        # Use the last split for final training
        X_train = X_scaled.iloc[train_splits[-1]]
        X_val = X_scaled.iloc[test_splits[-1]]
        y_train = y_scaled.iloc[train_splits[-1]]
        y_val = y_scaled.iloc[test_splits[-1]]

        # Check shapes and types before fitting the model
        print(f"\nDEBUG: Shape of X_train: {X_train.shape}, Type: {type(X_train)}\n")
        print(f"\nDEBUG: Shape of y_train: {y_train.shape}, Type: {type(y_train)}\n")
        print(f"\nDEBUG: Shape of X_val: {X_val.shape}, Type: {type(X_val)}\n")
        print(f"\nDEBUG: Shape of y_val: {y_val.shape}, Type: {type(y_val)}\n")

        # Configure XGBoost model
        model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=5,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42,
            eval_metric='rmse',
            early_stopping_rounds=50
        )

        # Train with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )

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
    
    elif model_type != "linear_regression" and model_type != "xgboost":
        
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
        ticker_value = train_ready_data.columns[0][1]
        
        print(f"Shape of train_ready_data: {train_ready_data.shape}")
        print(f"Columns in train_ready_data: {train_ready_data.columns.tolist()}")
        
        # Select the last row
        last_row = train_ready_data.iloc[-1:]
        print(f"Last row data:\n{last_row}")
        
        # Get feature columns for non-linear models using the feature engineering utility
        feature_columns = get_feature_columns(last_row, model_type)
        
        print(f"DEBUG - Non-Linear Model Features: {feature_columns}")

        # Separate features and target
        X = train_ready_data[feature_columns]  
        y = train_ready_data[('Next_Day_Close')]  # Ensure y is a Series
        
        print(f"Training with {len(feature_columns)} features:")
        
        print("\nTechnical indicators:")
        print([col for col in feature_columns if col[0] not in ['Open', 'High', 'Low', 'Close', 'Prev Close', 'Volume']])
        
        # Scale the features
        scaler = StandardScaler()
        print(f"\nDEBUG: Fitting the scaler in train_model function\n")
        X_scaled = scaler.fit_transform(X)
        print(f"Shape of X_scaled: {X_scaled.shape}")  # Debugging line
        print(f"X_scaled size: {X_scaled.size}")  # Debugging line
        print("X_scaled contents:")
        print(X_scaled[:5])  # Display the first 5 rows
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Use TimeSeriesSplit instead of random split
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []

        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate MSE
            mse = np.mean((y_test.values - y_pred)**2)  # Ensure y_test is a Series
            
            cv_scores.append(mse)

        print("Non-Linear Model CV Scores:", cv_scores)
        print("Mean CV Score:", np.mean(cv_scores))
        
        # Train final model on all data
        model.fit(X_scaled, y)
        
        return model, scaler

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