"""
Synthetic Sparse Resource Usage Dataset Generator for SparseModel.

Generates regression datasets with multiple candidate features where only a
sparse subset of features contributes meaningfully to resource consumption.
Features represent potential drivers (e.g., server load, memory requests,
concurrent users, batch jobs, cache misses, disk writes, background daemons, network packets).
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class SparseDataset:
    features: np.ndarray       # Normalized features (N, p)
    feature_names: List[str]
    targets: np.ndarray        # Target resource usage (N,)
    true_weights: np.ndarray   # True sparse weight vector
    active_indices: List[int]
    noise_std: float
    scaler_mean: np.ndarray
    scaler_std: np.ndarray


def generate_sparse_resource_data(
    num_samples: int = 500,
    seed: int = 42,
    noise_std: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, List[int]]:
    """
    Constructs synthetic data with 8 features, of which only 3 are active.
    """
    np.random.seed(seed)
    
    feature_names = [
        "CPU_Load_pct",            # Active (w = 3.5)
        "Aux_Sensor_A",            # Irrelevant (w = 0.0)
        "Memory_Allocation_GB",    # Active (w = -2.5)
        "Aux_Sensor_B",            # Irrelevant (w = 0.0)
        "Concurrent_Requests_k",   # Active (w = 1.8)
        "Redundant_Counter_1",     # Irrelevant (w = 0.0)
        "Redundant_Counter_2",     # Irrelevant (w = 0.0)
        "Noise_Channel",           # Irrelevant (w = 0.0)
    ]
    
    p = len(feature_names)
    true_weights = np.array([3.5, 0.0, -2.5, 0.0, 1.8, 0.0, 0.0, 0.0])
    active_indices = [0, 2, 4]
    
    # Generate random features with mild inter-correlations
    mean_vec = np.zeros(p)
    cov_mat = np.eye(p)
    for i in range(p):
        for j in range(p):
            if i != j:
                cov_mat[i, j] = 0.15 ** abs(i - j)
                
    X = np.random.multivariate_normal(mean_vec, cov_mat, size=num_samples)
    noise = np.random.normal(0.0, noise_std, size=num_samples)
    y = X @ true_weights + noise
    
    return X, y, feature_names, true_weights, active_indices


def create_sparse_regression_dataset(
    num_samples: int = 500,
    train_ratio: float = 0.8,
    seed: int = 42,
    noise_std: float = 0.5,
) -> Tuple[SparseDataset, SparseDataset]:
    """
    Constructs normalized train and validation sets with zero data leakage.
    """
    X_raw, y, feature_names, true_weights, active_indices = generate_sparse_resource_data(
        num_samples=num_samples, seed=seed, noise_std=noise_std
    )
    
    n_train = int(num_samples * train_ratio)
    
    X_train_raw = X_raw[:n_train]
    y_train = y[:n_train]
    X_val_raw = X_raw[n_train:]
    y_val = y[n_train:]
    
    scaler_mean = np.mean(X_train_raw, axis=0)
    scaler_std = np.std(X_train_raw, axis=0)
    scaler_std[scaler_std < 1e-8] = 1.0
    
    X_train_norm = (X_train_raw - scaler_mean) / scaler_std
    X_val_norm = (X_val_raw - scaler_mean) / scaler_std
    
    train_ds = SparseDataset(
        features=X_train_norm,
        feature_names=feature_names,
        targets=y_train,
        true_weights=true_weights,
        active_indices=active_indices,
        noise_std=noise_std,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
    )
    
    val_ds = SparseDataset(
        features=X_val_norm,
        feature_names=feature_names,
        targets=y_val,
        true_weights=true_weights,
        active_indices=active_indices,
        noise_std=noise_std,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
    )
    
    return train_ds, val_ds
