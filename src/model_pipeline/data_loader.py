import pandas as pd

def load_and_merge(train_path, store_path):
    train = pd.read_csv(train_path, low_memory=False)
    store = pd.read_csv(store_path)
    df = pd.merge(train, store, how='left', on='Store')
    return df