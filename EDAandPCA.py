import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def get_pca_features(file_path: str) -> pd.DataFrame:
    """
    Load data from csv file and return a pandas dataframe.
    """
    df = pd.read_csv(file_path)

    assert 'RanMei' in df.columns, "RanMei column not found in the data"
    assert 'YiChangYiErChun' in df.columns, "YiChangYiErChun column not found in the data"
    
    df = df.drop(columns=['RanMei', 'YiChangYiErChun'])
    df = df.fillna(0)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    pca = PCA(n_components=4)
    df_pca = pca.fit_transform(df_scaled)

    scaler_whitened = StandardScaler()
    df_pca_whitened = scaler_whitened.fit_transform(df_pca)
    return df_pca_whitened


def get_pca_features_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Load data features from pandas dataframe and return a pandas dataframe.
    """
    df = df.fillna(0)

    if 'YiChangYiErChun' in df.columns:
        df = df.drop(columns=['YiChangYiErChun'])

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    pca = PCA(n_components=4)
    df_pca = pca.fit_transform(df_scaled)

    scaler_whitened = StandardScaler()
    df_pca_whitened = scaler_whitened.fit_transform(df_pca)

    return pd.DataFrame(df_pca_whitened, columns=['PC1', 'PC2', 'PC3', 'PC4'])
