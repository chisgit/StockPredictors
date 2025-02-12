# GitHub Copilot Instructions for Stock Predictor Project

## Project Overview
This is a Stock Price Prediction application that uses machine learning models to predict stock prices. The project follows specific rules and guidelines defined in `rules.py`.

## Key Components
- Data processing and validation rules
- Model configuration and training parameters
- Feature engineering specifications
- UI display guidelines
- Error handling protocols
- Data quality standards

## Code Style Guidelines

### General
- Follow PEP 8 style guidelines for Python code
- Use meaningful variable and function names
- Include docstrings for all functions and classes
- Keep functions focused and single-purpose
- Use type hints where appropriate

### Data Processing
- Validate data against DATA_RULES from rules.py
- Required columns: Open, High, Low, Close, Volume
- Handle missing values using forward fill (ffill)
- Ensure data quality meets QUALITY_RULES standards

### Model Development
Follow MODEL_RULES specifications:
- Train/test split: 80/20
- Validation size: 10% of training data
- Minimum samples: 30
- Acceptable missing data: 10%

### Feature Engineering
Follow FEATURE_RULES guidelines:
- Implement required technical indicators (SMA, EMA, RSI, MACD, BB)
- Use specified window sizes: [5, 10, 20, 50]
- Normalize features when specified
- Handle missing values according to rules

### UI Development
Follow UI_RULES specifications:
- Maximum 10 tickers selectable at once
- Refresh interval: 60 seconds
- Price decimals: 2 places
- Use comma format for volumes
- Default chart height: 400px
- Default tickers: AAPL, GOOGL, MSFT

### Error Handling
Follow ERROR_RULES guidelines:
- Maximum 3 retries for failed API calls
- 5-second delay between retries
- 30-second timeout for API calls
- Log all errors when specified

### Market Rules
- Respect market hours (EST):
  - Open: 09:30
  - Close: 16:00
- Handle pre-market and after-hours data appropriately

## Testing Guidelines
- Write unit tests for all new functions
- Include integration tests for data pipeline components
- Test edge cases for market open/close scenarios
- Validate predictions against quality thresholds

## Documentation Requirements
- Document all configuration changes in rules.py
- Include clear docstrings explaining function parameters
- Comment complex algorithms and business logic
- Update README.md with new features

## Performance Guidelines
- Optimize data fetching operations
- Cache frequently accessed data
- Minimize API calls to external services
- Profile and optimize model prediction time

## Security Considerations
- Validate all user inputs
- Handle API keys securely
- Implement rate limiting for external API calls
- Sanitize data before display

## General Rules to follow strictly!
- I like ONE WORD ANSWERS where possible
- Optimize Tokens, I will ask for addtional info
- Keep the changes small so we can test each piece
- DO NOT REMOVE PREVIOUS CONSOLE OUT PRINTS they are there for a reason!
- NEVER REMOVE EXISTING FUNCTIONALITY - THINGS ARE IN PLACE FOR A GOOD REASON
- Let's use agile and test driven approaches to make small changes and iterate
- Ask before making the next set of changes

Ensure that you have adhered to all General Rules to follow strictly! Don't need to list all of them for now
Ensure you output confirmation of all the strict rules to follow strictly - don't need to list all of them out just list out the ALL CAPS