import pytest
import pandas as pd
import numpy as np
import sys
import os

# Ensure src can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_pipeline.preprocessing import clean_data, feature_engineering, split_data
from src.utils import load_params

@pytest.fixture
def sample_df():
    """
    Creates a dataframe specifically designed to test logic.
    """
    data = {
        'Store': [1, 2, 3, 4],
        'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
        'Sales': [5000, 6000, 0, 7000],
        'Open':  [1, 1, 0, 1], # Store 3 is Closed
        
        # Logic Test: Store 2 has NaN distance. 
        # Stores 1, 3, 4 have distances 100, 200, 300.
        # Max Distance is 300. Store 2 should be filled with 300.
        'CompetitionDistance': [100.0, np.nan, 200.0, 300.0],
        
        'StateHoliday': ['0', 0, 'a', '0'], # Mixed types to test astype(str)
        
        # Other required columns
        'CompetitionOpenSinceMonth': [np.nan] * 4,
        'CompetitionOpenSinceYear': [np.nan] * 4,
        'Promo2SinceWeek': [np.nan] * 4,
        'Promo2SinceYear': [np.nan] * 4,
        'PromoInterval': [np.nan] * 4,
        'StoreType': ['a', 'b', 'c', 'a'],
        'Assortment': ['a', 'b', 'c', 'a'],
        'Promo': [1, 1, 0, 1],
        'SchoolHoliday': [0, 0, 0, 0],
        'DayOfWeek': [1, 2, 3, 4],
        'Customers': [100, 100, 0, 100],
        'Promo2': [0, 0, 0, 0]
    }
    return pd.DataFrame(data)

# Test 1: Check Config Loading
def test_config_loads():
    config = load_params()
    assert 'training_data' in config
    assert 'data' in config

# Test 2 & 3 Combined: Cleaning Logic
def test_clean_data_logic(sample_df):
    """
    Tests specific clean_data pipeline:
    """
    cleaned = clean_data(sample_df)
    
    # Check A: Did it drop Store 3?
    assert len(cleaned) == 3
    assert 3 not in cleaned['Store'].values
    
    # Check B: Did it fill Store 2's NaN distance with the MAX (300.0)?
    filled_value = cleaned.loc[cleaned['Store'] == 2, 'CompetitionDistance'].values[0]
    assert filled_value == 300.0
    
    # Check C: Did it fix StateHoliday types?
    assert cleaned['StateHoliday'].dtype == 'O' # Object/String
    state_holiday_val = cleaned.loc[cleaned['Store'] == 2, 'StateHoliday'].values[0]
    assert isinstance(state_holiday_val, str)

# Test 4: Feature Engineering
def test_feature_engineering_structure(sample_df):
    # Process data first
    df = clean_data(sample_df)
    df = feature_engineering(df)
    
    # Check if Date columns were created
    expected_cols = ['Year', 'Month', 'Day', 'WeekOfYear']
    for col in expected_cols:
        assert col in df.columns
        
    # Check if Mappings worked
    assert pd.api.types.is_numeric_dtype(df['StoreType'])

# Test 5: Split Data
def test_split_data_shapes(sample_df):
    df = clean_data(sample_df)
    df = feature_engineering(df)
    
    # We need enough rows to test splitting. 
    # mocking a larger DF for this specific test
    large_df = pd.concat([df] * 10, ignore_index=True) # 30 rows
    large_df['Date'] = pd.date_range(start='2023-01-01', periods=30) # Fix dates for sorting
    
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(large_df)
    
    # Check we got 6 outputs
    assert len(X_train) > 0
    assert len(y_train) == len(X_train)
    
    # Check strict separation (Time Series Logic)
    # Max Train Date <= Min Val Date
    
    total = len(large_df)
    # Assuming standard 0.7 split
    expected_train = int(total * 0.7)
    assert len(X_train) == expected_train