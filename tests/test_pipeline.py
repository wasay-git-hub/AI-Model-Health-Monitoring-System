import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_pipeline.preprocessing import process_data, split_raw_data, extract_X_y
from src.utils import load_params

@pytest.fixture
def sample_df():
    data = {
        'Store': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Date': pd.date_range(start='2023-01-01', periods=10).astype(str),
        'Sales': [5000, 6000, 0, 7000, 8000, 9000, 4000, 3000, 2000, 1000],
        'Open':  [1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
        'CompetitionDistance': [100.0, np.nan, 200.0, 300.0, 150.0, 250.0, 180.0, 220.0, 310.0, 90.0],
        'StateHoliday': ['0', 0, 'a', '0', '0', '0', '0', '0', '0', '0'],
        'CompetitionOpenSinceMonth': [np.nan] * 10,
        'CompetitionOpenSinceYear': [np.nan] * 10,
        'Promo2SinceWeek': [np.nan] * 10,
        'Promo2SinceYear': [np.nan] * 10,
        'PromoInterval': [np.nan] * 10,
        'StoreType': ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a'],
        'Assortment': ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a'],
        'Promo': [1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
        'SchoolHoliday': [0] * 10,
        'DayOfWeek': [1, 2, 3, 4, 5, 6, 7, 1, 2, 3],
        'Customers': [100] * 10,
        'Promo2': [0] * 10
    }
    return pd.DataFrame(data)


# Test 1: Config loads correctly
def test_config_loads():
    config = load_params()
    assert 'training_data' in config
    assert 'data' in config


# Test 2: process_data drops closed stores and fills NaN distance
def test_process_data_cleaning(sample_df):
    processed, _ = process_data(sample_df)

    # Store 3 was Open=0, must be dropped
    assert 3 not in processed['Store'].values

    # NaN CompetitionDistance must be filled with max
    assert processed['CompetitionDistance'].isna().sum() == 0


# Test 3: process_data fills all NaN columns
def test_process_data_no_nulls(sample_df):
    processed, _ = process_data(sample_df)

    for col in ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear',
                'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']:
        assert processed[col].isna().sum() == 0


# Test 4: process_data creates date feature columns
def test_process_data_feature_engineering(sample_df):
    processed, _ = process_data(sample_df)

    for col in ['Year', 'Month', 'Day', 'WeekOfYear']:
        assert col in processed.columns


# Test 5: Categorical columns are mapped to numeric
def test_categorical_mapping(sample_df):
    processed, _ = process_data(sample_df)

    assert pd.api.types.is_numeric_dtype(processed['StoreType'])
    assert pd.api.types.is_numeric_dtype(processed['Assortment'])


# Test 6: split_raw_data returns 3 splits with correct sizes
def test_split_raw_data_shapes(sample_df):
    processed, _ = process_data(sample_df)
    train_df, val_df, test_df = split_raw_data(processed)

    total = len(train_df) + len(val_df) + len(test_df)
    assert total == len(processed)
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0


# Test 7: extract_X_y returns correct shapes
def test_extract_X_y(sample_df):
    processed, _ = process_data(sample_df)
    X, y = extract_X_y(processed)

    assert len(X) == len(y)
    assert 'Sales' not in X.columns