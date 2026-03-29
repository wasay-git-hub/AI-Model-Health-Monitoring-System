import pandas as pd
import numpy as np
from src.utils import get_mappings, load_params

config = load_params()

def load_and_merge(train_path, store_path):
    train = pd.read_csv(train_path, low_memory=False)
    store = pd.read_csv(store_path)
    df = pd.merge(train, store, how='left', on='Store')
    return df

def split_raw_data(df):
    """
    Splits the raw dataframe into Train, CV, Test based on time.
    Returns DataFrames.
    """
    # Sort by Date and reset index
    df = df.sort_values('Date').reset_index(drop=True)

    # splitting ratio of 70:15:15
    total_rows = len(df)
    train_ratio = config['data']['train_ratio']
    val_ratio = config['data']['cv_ratio']

    train_end = int(total_rows * train_ratio)
    val_end = int(total_rows * (train_ratio + val_ratio))

    # Split into 3 chunks
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df

def process_data(df, train_stats=None):
    """
    Cleans and Engineers features.
    If train_stats is None, we calculate them (this is Training Data).
    If train_stats is provided, we use them (this is CV/Test Data).
    """
    # cleaning
    df['StateHoliday'] = df['StateHoliday'].astype(str)

    # Calculate Max only if we are in Training mode (prevents leakage)
    if train_stats is None:
        max_dist = df['CompetitionDistance'].max()
        train_stats = {'max_dist': max_dist}
    else:
        max_dist = train_stats['max_dist']

    # Apply the fill
    df['CompetitionDistance'] = df['CompetitionDistance'].fillna(max_dist)
    
    # Constant fills
    df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(0)
    df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].fillna(0)
    df['Promo2SinceWeek'] = df['Promo2SinceWeek'].fillna(0)
    df['Promo2SinceYear'] = df['Promo2SinceYear'].fillna(0)
    df['PromoInterval'] = df['PromoInterval'].fillna("None")

    # Drop rows where store is closed (Sales=0)
    # Note: We do this for all sets to avoid dividing by zero in RMSPE
    df = df[df['Open'] != 0].copy()

    # feature engineering
    # Support mixed date formats across train/test/input files (e.g., YYYY-MM-DD and DD/MM/YYYY)
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True, errors='coerce')
    df = df[df['Date'].notna()].copy()
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)

    mappings = get_mappings()
    for col in config['categorical_features']:
        df[col] = df[col].map(lambda x: mappings.get(x, 0))

    # Return the processed df and the stats (so we can pass stats to CV/Test)
    return df, train_stats

def extract_X_y(df):
    """
    Helper to separate Features and Target after processing
    """
    X = df[config['training_data']['features']]
    y = df[config['training_data']['target']]
    return X, y