from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler


LOGGER = logging.getLogger(__name__)


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    x_norm = np.sum(x * x, axis=1).reshape(-1, 1)
    y_norm = np.sum(y * y, axis=1).reshape(1, -1)
    dist2 = x_norm + y_norm - 2.0 * np.dot(x, y.T)
    return np.exp(-gamma * np.maximum(dist2, 0.0))


def _compute_mmd_rbf(x: np.ndarray, y: np.ndarray, gamma: float | None = None) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0

    if gamma is None:
        joined = np.vstack([x, y])
        if len(joined) < 2:
            gamma = 1.0
        else:
            sample_idx = np.random.default_rng(42).choice(len(joined), size=min(256, len(joined)), replace=False)
            sub = joined[sample_idx]
            d2 = np.sum((sub[:, None, :] - sub[None, :, :]) ** 2, axis=2)
            med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
            gamma = 1.0 / (2.0 * med)

    k_xx = _rbf_kernel(x, x, gamma)
    k_yy = _rbf_kernel(y, y, gamma)
    k_xy = _rbf_kernel(x, y, gamma)
    mmd2 = k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
    return float(np.sqrt(max(mmd2, 0.0)))


def mmd_distribution_normalization(source_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """
    Distribution-aware normalization using MMD-guided alignment.

    The function applies per-feature linear scaling of source data toward target statistics,
    and uses RBF-MMD as quality metric.
    """
    src = source_df.copy()
    src_num = _numeric_columns(src)
    tgt_num = _numeric_columns(target_df)
    shared = [c for c in src_num if c in tgt_num]
    if not shared:
        return src

    src_values = src[shared].copy().fillna(src[shared].median(numeric_only=True))
    tgt_values = target_df[shared].copy().fillna(target_df[shared].median(numeric_only=True))

    before = _compute_mmd_rbf(src_values.values, tgt_values.values)

    src_mean = src_values.mean(axis=0)
    src_std = src_values.std(axis=0).replace(0.0, 1.0)
    tgt_mean = tgt_values.mean(axis=0)
    tgt_std = tgt_values.std(axis=0).replace(0.0, 1.0)

    aligned = (src_values - src_mean) / src_std * tgt_std + tgt_mean
    after = _compute_mmd_rbf(aligned.values, tgt_values.values)
    LOGGER.info("MMD before=%.6f, after=%.6f", before, after)

    src[shared] = aligned
    return src


def kpca_feature_extraction(df: pd.DataFrame, n_components: int | None = None) -> pd.DataFrame:
    """Extract nonlinear components with RBF-KPCA."""
    num_cols = _numeric_columns(df)
    if not num_cols:
        return df.copy()

    numeric = df[num_cols].copy().fillna(df[num_cols].median(numeric_only=True))
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric.values)

    if n_components is None:
        n_components = min(8, len(num_cols), len(df))
        n_components = max(n_components, 1)

    kpca = KernelPCA(n_components=n_components, kernel="rbf", fit_inverse_transform=False, random_state=42)
    transformed = kpca.fit_transform(scaled)
    cols = [f"kpca_{i + 1}" for i in range(transformed.shape[1])]
    return pd.DataFrame(transformed, columns=cols, index=df.index)


def _build_sequences(values: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    centers = []
    for i in range(len(values) - window_size + 1):
        window = values[i : i + window_size]
        xs.append(window)
        centers.append(i + window_size // 2)
    if not xs:
        return np.empty((0, window_size, values.shape[1])), np.empty((0,), dtype=int)
    return np.array(xs, dtype=np.float32), np.array(centers, dtype=int)


def lstm_time_series_correction(
    df: pd.DataFrame, window_size: int = 10, epochs: int = 100, learning_rate: float = 0.001
) -> pd.DataFrame:
    """
    LSTM autoencoder based time-series anomaly correction.

    - hidden units: 64
    - epochs: default 100
    - optimizer lr: default 0.001
    """
    num_cols = _numeric_columns(df)
    if not num_cols or len(df) < window_size + 2:
        return df.copy()

    try:
        import tensorflow as tf
    except Exception:
        LOGGER.warning("TensorFlow is unavailable; skip LSTM correction.")
        return df.copy()

    tf.random.set_seed(42)
    np.random.seed(42)

    out = df.copy()
    numeric = out[num_cols].copy().interpolate(method="linear", limit_direction="both")
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    values = numeric.values.astype(np.float32)

    x_train, center_idx = _build_sequences(values, window_size=window_size)
    if len(x_train) == 0:
        return out

    n_features = values.shape[1]
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(window_size, n_features)),
            tf.keras.layers.LSTM(64, return_sequences=False),
            tf.keras.layers.RepeatVector(window_size),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features)),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")

    class _EpochLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):  # type: ignore[override]
            if (epoch + 1) % 10 == 0 or epoch == 0:
                LOGGER.info("LSTM epoch %d/%d, loss=%.6f", epoch + 1, epochs, float(logs.get("loss", 0.0)))

    LOGGER.info("Start LSTM training: samples=%d, window=%d, features=%d", len(x_train), window_size, n_features)
    model.fit(x_train, x_train, epochs=epochs, batch_size=32, verbose=0, callbacks=[_EpochLogger()])

    recon = model.predict(x_train, verbose=0)
    recon_err = np.mean((recon - x_train) ** 2, axis=(1, 2))
    threshold = float(recon_err.mean() + 3.0 * recon_err.std())
    anomaly_mask = recon_err > threshold

    corrected_values = values.copy()
    for i, is_anomaly in enumerate(anomaly_mask):
        if not is_anomaly:
            continue
        center = center_idx[i]
        recon_center = recon[i, window_size // 2]
        corrected_values[center] = recon_center

    out[num_cols] = corrected_values
    LOGGER.info("LSTM correction finished: anomalies corrected=%d", int(anomaly_mask.sum()))
    return out

