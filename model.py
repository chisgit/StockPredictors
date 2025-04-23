import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from feature_engineering import get_feature_columns


def train_model(
    train_ready_data,
    model_type="linear_regression",
    X_val=None,
    y_val=None,
    target="today",
):
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

    # Get ticker from data columns
    ticker_level = 1  # Index of ticker in MultiIndex
    current_ticker = train_ready_data.columns[0][ticker_level]
    print(f"Training model for ticker: {current_ticker}")

    # Get features with target parameter
    feature_cols = get_feature_columns(model_type=model_type, target=target)
    X = train_ready_data[[(col, current_ticker) for col in feature_cols]]

    # Select target based on prediction goal
    if target == "today":
        y = train_ready_data[("Close", current_ticker)]
        print(f"Using Close price as target for {current_ticker}")
    else:  # next_day
        y = train_ready_data[("Next_Day_Close", current_ticker)]
        print(f"Using Next_Day_Close as target for {current_ticker}")

    # Scale features and target
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = X_scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))
    y_scaled = pd.Series(y_scaled.flatten(), index=y.index)

    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Use TimeSeriesSplit for validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []

    if model_type == "linear_regression":
        model = LinearRegression()

        # Cross validation
        for train_idx, val_idx in tscv.split(X_train):
            X_train_fold = X_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_train_fold = y_train.iloc[train_idx]
            y_val_fold = y_train.iloc[val_idx]

            model.fit(X_train_fold, y_train_fold)
            score = model.score(X_val_fold, y_val_fold)
            cv_scores.append(score)

        print(f"\nLinear Regression CV Scores: {cv_scores}")
        print(f"Mean CV Score: {np.mean(cv_scores)}")

        # Train on full train set
        model.fit(X_train, y_train)

    elif model_type == "xgboost":
        model = xgb.XGBRegressor(
            n_estimators=2000,  # Doubled for even finer convergence
            learning_rate=0.01,  # Further reduced for precision
            max_depth=3,  # Reduced complexity
            min_child_weight=5,  # More aggressive pattern filtering
            subsample=0.95,  # Increased sampling
            colsample_bytree=0.95,  # Increased feature sampling
            objective="reg:squarederror",
            random_state=42,
            eval_metric="rmse",
            early_stopping_rounds=50,  # Increased patience significantly
            gamma=0.01,  # Minimal regularization
            reg_alpha=0.01,  # Minimal L1 regularization
            reg_lambda=0.05,  # Reduced L2 regularization
            max_leaves=16,  # Reduced tree complexity
            tree_method='hist'  # Faster, more accurate method
        )

        # Train with validation
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # Print feature importance with threshold
        importance = pd.DataFrame(
            {"feature": feature_cols, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)
        
        # Filter features by importance threshold
        importance_threshold = 0.00001
        significant_features = importance[importance["importance"] > importance_threshold]
        
        print(f"\nFeature importance for {current_ticker} (threshold > {importance_threshold}):")
        if not significant_features.empty:
            print(significant_features.head())
        else:
            print("No features above importance threshold")
            
        # Print lowest importance features for debugging
        print("\nLowest importance features:")
        print(importance.tail())

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
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_names = X.columns
        feature_importance_dict = dict(zip(feature_names, importances))

        # Sort features by importance
        sorted_features = sorted(
            feature_importance_dict.items(), key=lambda x: x[1], reverse=True
        )

        print("\nFeature Importances:")
        for feature, importance in sorted_features:
            print(f"{feature}: {importance:.4f}")

        return dict(sorted_features)
    else:
        print("Model does not support feature importance.")
        return {}
