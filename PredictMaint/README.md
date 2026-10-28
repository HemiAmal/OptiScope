# PredictMaint: Industrial Predictive Maintenance Risk Classification

PredictMaint is an optimization platform identifying factory machinery at risk of breakdown from multivariate industrial sensor telemetry.

---

## 1. Problem Formulation & Methodology

### Equipment Failure Risk Classification
- **Objective:** Minimize Binary Cross-Entropy (BCE) / Log-Loss over telemetry measurements:
  $$\min_{w, b} L(w, b) = -\frac{1}{n} \sum_{i=1}^n \left[y_i \ln \sigma(w^T x_i + b) + (1 - y_i) \ln(1 - \sigma(w^T x_i + b))\right]$$
  where $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the logistic sigmoid activation.
- **Sensor Features:**
  1. Vibration (mm/s RMS)
  2. Operating Temperature (°C)
  3. Operating Hours since overhaul
  4. Differential Pressure (psi)
  5. Mechanical Load percentage (%)
- **Optimization Algorithms:**
  - Logistic Regression with analytical gradient descent
  - AdaGrad
  - RMSProp
  - Adam
- **Evaluation:** BCE loss, Accuracy, Precision, Recall, F1 score, Confusion Matrix, 2D decision boundary visualization on physical units, and decision threshold sensitivity sweep ($t \in [0.1, 0.9]$).

---

## 2. Directory Structure

```
PredictMaint/
├── main.py                         # Interactive CLI & runner
├── README.md                       # Project documentation
├── data/
│   └── synthetic_sensors.py        # Telemetry simulator & zero-leakage standardizer
├── problems/
│   └── failure_classification.py   # BCE loss & gradient optimization formulations
├── experiments/
│   └── run_experiments.py         # End-to-end experiment pipeline
├── visualization/
│   └── plots.py                    # Loss curves, 2D boundary, confusion matrix, threshold curves
├── results/
│   ├── plots/                      # Generated figures (.png)
│   ├── histories/                  # Epoch histories (.csv)
│   └── reports/                    # Benchmark metrics (.txt)
└── tests/
    └── test_maintenance.py        # Automated unit tests
```

---

## 3. Running PredictMaint

```bash
# Run experiments and generate plots/reports:
python main.py --all

# Run automated unit tests:
python main.py --test
```
