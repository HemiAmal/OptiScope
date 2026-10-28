import numpy as np
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class SensorDataset:
    features: np.ndarray
    feature_names: List[str]
    labels: np.ndarray
    raw_features: np.ndarray
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    ground_truth_weights: List[float]


def generate_industrial_sensor_data(
    num_samples: int = 1200,
    seed: int = 42,
    failure_rate_target: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[float]]:
    np.random.seed(seed)

    feature_names = [
        "Vibration_RMS_mms",
        "Temperature_C",
        "Operating_Hours",
        "Pressure_psi",
        "Mechanical_Load_pct",
    ]

    vibration = np.random.gamma(shape=3.0, scale=1.0, size=num_samples)
    temperature = np.random.normal(loc=65.0, scale=8.0, size=num_samples)
    hours = np.random.uniform(low=200.0, high=4500.0, size=num_samples)
    pressure = np.random.normal(loc=32.0, scale=4.0, size=num_samples)
    load = np.random.beta(a=4.0, b=2.0, size=num_samples) * 100.0

    X_raw = np.column_stack([vibration, temperature, hours, pressure, load])

    w_true = [1.25, 0.95, 0.85, 0.50, 0.90]
    intercept = -2.40

    x_standardized = (X_raw - np.mean(X_raw, axis=0)) / np.std(X_raw, axis=0)
    pressure_dev = np.abs(x_standardized[:, 3])

    logits = (
        intercept
        + w_true[0] * x_standardized[:, 0]
        + w_true[1] * x_standardized[:, 1]
        + w_true[2] * x_standardized[:, 2]
        + w_true[3] * pressure_dev
        + w_true[4] * x_standardized[:, 4]
    )

    probabilities = 1.0 / (1.0 + np.exp(-logits))
    labels = (np.random.uniform(0.0, 1.0, size=num_samples) < probabilities).astype(float)

    return X_raw, labels, feature_names, w_true


def create_predictive_maintenance_dataset(
    num_samples: int = 1200,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[SensorDataset, SensorDataset]:
    X_raw, labels, feature_names, w_true = generate_industrial_sensor_data(
        num_samples=num_samples, seed=seed
    )

    n_train = int(num_samples * train_ratio)

    X_train_raw = X_raw[:n_train]
    y_train = labels[:n_train]

    X_val_raw = X_raw[n_train:]
    y_val = labels[n_train:]

    scaler_mean = np.mean(X_train_raw, axis=0)
    scaler_std = np.std(X_train_raw, axis=0)
    scaler_std[scaler_std < 1e-8] = 1.0

    X_train_norm = (X_train_raw - scaler_mean) / scaler_std
    X_val_norm = (X_val_raw - scaler_mean) / scaler_std

    train_ds = SensorDataset(
        features=X_train_norm,
        feature_names=feature_names,
        labels=y_train,
        raw_features=X_train_raw,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        ground_truth_weights=w_true,
    )

    val_ds = SensorDataset(
        features=X_val_norm,
        feature_names=feature_names,
        labels=y_val,
        raw_features=X_val_raw,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        ground_truth_weights=w_true,
    )

    return train_ds, val_ds
