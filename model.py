from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def train_model(train_ready_data, model_type):
    # """
    # Trains the model on the provided data.
    
    # Args:
    #     train_ready_data (pd.DataFrame): Preprocessed training data
    #     model_type (str): Type of model to train
    
    # Returns:
    #     Trained machine learning model
    # """
    
    # Separate features and target
    X = train_ready_data[['Open', 'High', 'Low', 'Volume', 'Prev Close']]
    y = train_ready_data['Close']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model based on type
    if model_type == 'linear':
        model = LinearRegression()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Fit the model
    model.fit(X, y)
    return model