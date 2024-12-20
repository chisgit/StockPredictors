import gen_utils
import app
from models.base_model import BaseModel
from models.linear_reg_model import LinearModel  # Add other models as needed

def initialize_model(model_type="Linear"):
    """
    Initializes and returns a model based on the provided model type.
    Parameters:
        model_type (str): The type of model to initialize (e.g., 'Linear').
    Returns:
        BaseModel: An instance of the requested model type.
    """
    if model_type == "Linear":
        return LinearModel()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def initialize_session():
    # Initialize session state variables
    if 'tickers' not in st.session_state:
        st.session_state.tickers = ['TSLA', 'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']

    if 'selected_tickers' not in st.session_state:
        st.session_state.selected_tickers = []
        
# Additional initializations can be added here
def initialize_app():
    """
    Placeholder for app-wide initialization logic.
    """
    app.render_ui()
    app.multiselector()
    app.searchbar()
    print("App initialized successfully.")

# initializer.py