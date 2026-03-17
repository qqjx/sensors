from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from processing.advanced_tools import lstm_time_series_correction, mmd_distribution_normalization
from processing.basic_tools import iqr_anomaly_correction, knn_imputation, pca_dimensionality_reduction

try:
    from langchain.tools import tool

    HAS_LANGCHAIN = True
except Exception:
    HAS_LANGCHAIN = False

    def tool(*args, **kwargs):  # type: ignore[misc]
        def _decorator(fn):
            return fn

        return _decorator


@dataclass
class _WorkingMemory:
    source_df: Optional[pd.DataFrame] = None
    target_df: Optional[pd.DataFrame] = None


WORKING_MEMORY = _WorkingMemory()


def set_working_frames(source_df: pd.DataFrame, target_df: Optional[pd.DataFrame] = None) -> None:
    WORKING_MEMORY.source_df = source_df.copy()
    WORKING_MEMORY.target_df = target_df.copy() if target_df is not None else None


def get_working_df() -> pd.DataFrame:
    if WORKING_MEMORY.source_df is None:
        return pd.DataFrame()
    return WORKING_MEMORY.source_df.copy()


@tool
def iqr_anomaly_correction_tool() -> str:
    """
    Use IQR (1.5x) to detect and correct outliers.
    Best for noisy data with pulse-like spikes or obvious extreme values.
    """
    if WORKING_MEMORY.source_df is None:
        return "No working DataFrame found."
    WORKING_MEMORY.source_df = iqr_anomaly_correction(WORKING_MEMORY.source_df)
    return "Applied iqr_anomaly_correction."


@tool
def knn_imputation_tool() -> str:
    """
    Perform KNN missing-value imputation with k=5.
    Best when missing ratio is relatively low (for example <10%).
    """
    if WORKING_MEMORY.source_df is None:
        return "No working DataFrame found."
    WORKING_MEMORY.source_df = knn_imputation(WORKING_MEMORY.source_df, n_neighbors=5)
    return "Applied knn_imputation."


@tool
def pca_dimensionality_reduction_tool() -> str:
    """
    Run PCA to reduce redundancy while retaining ~95% explained variance.
    Best for wide numeric tables with correlated features.
    """
    if WORKING_MEMORY.source_df is None:
        return "No working DataFrame found."
    WORKING_MEMORY.source_df = pca_dimensionality_reduction(WORKING_MEMORY.source_df, variance_ratio=0.95)
    return "Applied pca_dimensionality_reduction."


@tool
def mmd_distribution_normalization_tool() -> str:
    """
    Align source distribution to target distribution using MMD-guided scaling.
    Best for cross-sensor or cross-domain distribution mismatch.
    """
    if WORKING_MEMORY.source_df is None:
        return "No working DataFrame found."
    if WORKING_MEMORY.target_df is None:
        return "No target DataFrame found; skipped MMD normalization."
    WORKING_MEMORY.source_df = mmd_distribution_normalization(WORKING_MEMORY.source_df, WORKING_MEMORY.target_df)
    return "Applied mmd_distribution_normalization."


@tool
def lstm_time_series_correction_tool() -> str:
    """
    Apply LSTM autoencoder based correction for strong temporal dependencies.
    Best for sequence sensors where temporal context improves anomaly repair.
    """
    if WORKING_MEMORY.source_df is None:
        return "No working DataFrame found."
    WORKING_MEMORY.source_df = lstm_time_series_correction(WORKING_MEMORY.source_df)
    return "Applied lstm_time_series_correction."


def build_langchain_tools() -> List[object]:
    if not HAS_LANGCHAIN:
        return []
    return [
        iqr_anomaly_correction_tool,
        knn_imputation_tool,
        pca_dimensionality_reduction_tool,
        mmd_distribution_normalization_tool,
        lstm_time_series_correction_tool,
    ]

