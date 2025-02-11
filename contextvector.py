
class ContextVector:
    """
    Maps function relationships and dependencies in the stock predictor project.
    """
    TEST 
    def __init__(self):
        self.function_map = {
            # Pipeline Entry Point
            'pipeline.py': {
                'execute_pipeline': {
                    'description': 'Main controller function that orchestrates the prediction process',
                    'dependencies': [
                        'fetch_data',
                        'preprocess_data',
                        'add_technical_indicators',
                        'train_model'
                    ]
                }
            },

            # Data Handling
            'data_handler.py': {
                'fetch_data': {
                    'description': 'Fetches historical stock data using yfinance',
                    'dependencies': [
                        'get_nyse_date',
                        'get_nyse_datetime'
                    ]
                }
            },

            # Model Training and Evaluation
            'model.py': {
                'train_model': {
                    'description': 'Trains either linear regression or XGBoost models',
                    'dependencies': [
                        'get_feature_columns',
                        'StandardScaler',
                        'LinearRegression',
                        'XGBRegressor'
                    ]
                },
                'feature_importance': {
                    'description': 'Calculates and displays feature importance for models',
                    'dependencies': []
                }
            },

            # Feature Engineering
            'feature_engineering.py': {
                'add_technical_indicators': {
                    'description': 'Adds all technical indicators to the dataset',
                    'dependencies': [
                        'add_moving_averages',
                        'add_momentum_indicators',
                        'add_volatility_indicators',
                        'add_trend_indicators',
                        'add_price_derivatives',
                        'add_lagged_features'
                    ]
                },
                'add_moving_averages': {
                    'description': 'Calculates SMA and EMA',
                    'dependencies': []
                },
                'add_momentum_indicators': {
                    'description': 'Calculates RSI, ROC, and Momentum',
                    'dependencies': ['RSIIndicator']
                },
                'add_volatility_indicators': {
                    'description': 'Calculates Bollinger Bands and ATR',
                    'dependencies': ['BollingerBands']
                },
                'add_trend_indicators': {
                    'description': 'Calculates MACD',
                    'dependencies': ['MACD']
                },
                'add_price_derivatives': {
                    'description': 'Calculates price changes and ranges',
                    'dependencies': []
                },
                'add_lagged_features': {
                    'description': 'Creates lagged versions of features',
                    'dependencies': []
                },
                'get_feature_columns': {
                    'description': 'Returns feature sets for different model types',
                    'dependencies': []
                }
            },

            # Model Utilities
            'model_utils.py': {
                'walk_forward_validation': {
                    'description': 'Performs walk-forward validation for time series',
                    'dependencies': [
                        'mean_squared_error',
                        'mean_absolute_error',
                        'r2_score'
                    ]
                },
                'print_model_performance': {
                    'description': 'Prints model performance metrics',
                    'dependencies': []
                }
            }
        }

    def get_function_dependencies(self, function_name):
        """Returns dependencies for a specific function"""
        for module in self.function_map.values():
            if function_name in module:
                return module[function_name]['dependencies']
        return []

    def get_module_functions(self, module_name):
        """Returns all functions in a specific module"""
        return self.function_map.get(module_name, {})

    def print_function_tree(self, function_name, indent=0):
        """Prints dependency tree for a function"""
        print('  ' * indent + function_name)
        deps = self.get_function_dependencies(function_name)
        for dep in deps:
            self.print_function_tree(dep, indent + 1)

