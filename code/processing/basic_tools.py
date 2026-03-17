from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def iqr_anomaly_correction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and smooth outliers with the IQR (1.5x) rule.

    Suitable when data includes pulse-like spikes and the median trend should be preserved.
    """
    out = df.copy()
    num_cols = _numeric_columns(out)
    if not num_cols:
        return out

    for col in num_cols:
        series = out[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        out[col] = series.clip(lower=lower, upper=upper)
    return out


def knn_imputation(df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    """
    Fill missing values with KNN imputation.

    Recommended when missing ratio is moderate to low (for example <10%) and numeric columns
    are correlated.
    """
    out = df.copy()
    num_cols = _numeric_columns(out)
    if not num_cols:
        return out

    imputer = KNNImputer(n_neighbors=n_neighbors)
    out[num_cols] = imputer.fit_transform(out[num_cols])
    return out


def pca_dimensionality_reduction(df: pd.DataFrame, variance_ratio: float = 0.95) -> pd.DataFrame:
    """
    Reduce feature redundancy with PCA while keeping target explained variance.
    """
    num_cols = _numeric_columns(df)
    if not num_cols:
        return df.copy()

    numeric = df[num_cols].copy()
    numeric = numeric.fillna(numeric.median(numeric_only=True))

    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric.values)

    pca = PCA(n_components=variance_ratio, svd_solver="full")
    transformed = pca.fit_transform(scaled)
    pca_cols = [f"pca_{i + 1}" for i in range(transformed.shape[1])]
    pca_df = pd.DataFrame(transformed, columns=pca_cols, index=df.index)

    non_num_cols = [c for c in df.columns if c not in num_cols]
    if non_num_cols:
        return pd.concat([pca_df, df[non_num_cols]], axis=1)
    return pca_df


def minmax_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Min-Max normalization to numeric columns."""
    out = df.copy()
    num_cols = _numeric_columns(out)
    if not num_cols:
        return out

    scaler = MinMaxScaler()
    out[num_cols] = scaler.fit_transform(out[num_cols])
    return out


def basic_fallback_processing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust fallback flow for industrial runtime:
    1) time/linear interpolation
    2) boundary fill
    3) min-max normalization
    """
    out = df.copy()
    num_cols = _numeric_columns(out)
    if num_cols:
        out[num_cols] = out[num_cols].interpolate(method="linear", limit_direction="both")
        out[num_cols] = out[num_cols].fillna(out[num_cols].median(numeric_only=True))
    out = minmax_normalize(out)
    return out

