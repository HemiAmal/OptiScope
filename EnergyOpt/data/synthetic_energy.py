import numpy as np
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class EnergyDataset:
    features: np.ndarray
    feature_names: List[str]
    targets: np.ndarray
    timestamps: np.ndarray
    raw_temperatures: np.ndarray
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    target_mean: float
    target_std: float


def generate_hourly_electricity_demand(
    num_days: int = 60,
    seed: int = 42,
    base_load: float = 250.0,
    trend_rate: float = 0.05,
    noise_std: float = 8.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    total_hours = num_days * 24
    hours = np.arange(total_hours)

    hour_of_day = hours % 24
    daily_pattern = (
        35.0 * np.sin(2.0 * np.pi * (hour_of_day - 6.0) / 24.0)
        + 20.0 * np.sin(4.0 * np.pi * (hour_of_day - 4.0) / 24.0)
    )

    day_of_week = (hours // 24) % 7
    is_weekend = (day_of_week >= 5).astype(float)
    weekly_effect = -25.0 * is_weekend

    temp_seasonal = 22.0 + 6.0 * np.sin(2.0 * np.pi * hours / (30.0 * 24.0))
    temp_daily = 5.0 * np.sin(2.0 * np.pi * (hour_of_day - 9.0) / 24.0)
    temp_noise = np.random.normal(0.0, 1.5, size=total_hours)
    temperatures = temp_seasonal + temp_daily + temp_noise

    temp_comfort = 20.0
    temp_effect = (
        2.2 * np.maximum(temperatures - temp_comfort, 0.0) ** 1.5
        + 1.8 * np.maximum(temp_comfort - temperatures, 0.0) ** 1.3
    )

    trend = trend_rate * hours
    noise = np.random.normal(0.0, noise_std, size=total_hours)

    demands = base_load + daily_pattern + weekly_effect + temp_effect + trend + noise
    demands = np.maximum(demands, 50.0)

    return hours, temperatures, demands


def create_energy_regression_dataset(
    num_days: int = 60,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[EnergyDataset, EnergyDataset]:
    hours, temps, demands = generate_hourly_electricity_demand(num_days=num_days, seed=seed)
    n_samples = len(demands) - 1

    hour_of_day = (hours[1:] % 24).astype(float)
    day_of_week = ((hours[1:] // 24) % 7).astype(float)
    is_weekend = (day_of_week >= 5).astype(float)
    current_temps = temps[1:]
    temp_sq = current_temps ** 2
    lag_demands = demands[:-1]
    y = demands[1:]

    X_raw = np.column_stack([
        hour_of_day,
        current_temps,
        temp_sq,
        is_weekend,
        lag_demands
    ])

    feature_names = [
        "HourOfDay",
        "Temperature_C",
        "Temp_Squared",
        "IsWeekend",
        "Lag1_Demand_MW"
    ]

    n_train = int(n_samples * train_ratio)

    train_mean = np.mean(X_raw[:n_train], axis=0)
    train_std = np.std(X_raw[:n_train], axis=0)
    train_std[train_std < 1e-8] = 1.0

    X_train_norm = (X_raw[:n_train] - train_mean) / train_std
    X_val_norm = (X_raw[n_train:] - train_mean) / train_std

    y_train = y[:n_train]
    y_val = y[n_train:]

    train_ds = EnergyDataset(
        features=X_train_norm,
        feature_names=feature_names,
        targets=y_train,
        timestamps=hours[1:n_train+1],
        raw_temperatures=current_temps[:n_train],
        scaler_mean=train_mean,
        scaler_std=train_std,
        target_mean=float(np.mean(y_train)),
        target_std=float(np.std(y_train)),
    )

    val_ds = EnergyDataset(
        features=X_val_norm,
        feature_names=feature_names,
        targets=y_val,
        timestamps=hours[n_train+1:],
        raw_temperatures=current_temps[n_train:],
        scaler_mean=train_mean,
        scaler_std=train_std,
        target_mean=float(np.mean(y_train)),
        target_std=float(np.std(y_train)),
    )

    return train_ds, val_ds
