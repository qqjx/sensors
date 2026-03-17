from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def perceive_data_state(df: pd.DataFrame) -> Dict[str, object]:
    """
    Lightweight data-state perception for agent planning.

    Returns:
    - missing_ratio: overall NaN ratio in [0, 1]
    - noise_level: low/medium/high estimated from first-difference volatility
    - has_outliers: whether IQR outliers exist in numeric columns
    """
    if df.empty:
        return {"missing_ratio": 0.0, "noise_level": "low", "has_outliers": False}

    missing_ratio = float(df.isna().sum().sum() / max(df.size, 1))

    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return {"missing_ratio": round(missing_ratio, 4), "noise_level": "low", "has_outliers": False}

    diff = numeric.diff().dropna(how="all")
    base_std = numeric.std(numeric_only=True).replace(0.0, np.nan)
    diff_std = diff.std(numeric_only=True) if not diff.empty else base_std
    ratio = (diff_std / base_std).replace([np.inf, -np.inf], np.nan).dropna()
    score = float(ratio.mean()) if not ratio.empty else 0.0

    if score < 0.5:
        noise_level = "low"
    elif score < 1.2:
        noise_level = "medium"
    else:
        noise_level = "high"

    has_outliers = False
    for col in numeric.columns:
        s = numeric[col].dropna()
        if s.empty:
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        if ((s < lower) | (s > upper)).any():
            has_outliers = True
            break

    return {
        "missing_ratio": round(missing_ratio, 4),
        "noise_level": noise_level,
        "has_outliers": has_outliers,
    }

