# NonlinearLab: Nonlinear Function Calibration & Smooth Optimization

NonlinearLab benchmarks first-order, accelerated, second-order, and quasi-Newton optimization methods across ill-conditioned quadratic landscapes and non-convex Rosenbrock valleys.

---

## 1. Problem Formulations & Methodology

### Ill-Conditioned Quadratic Benchmark
- **Objective:** Strictly convex quadratic with condition number $\kappa(Q) = \lambda_{\max}/\lambda_{\min} = 10.0$:
  $$f(x) = \frac{1}{2} x^T Q x - b^T x, \quad x^* = Q^{-1} b$$
- **Optimization Algorithms:**
  - Fixed-Step Gradient Descent
  - Exact Line Search GD: $\alpha_k^* = \frac{g_k^T g_k}{g_k^T Q g_k}$
  - Backtracking Line Search with Armijo condition
  - Polyak Heavy-Ball Momentum GD
  - Nesterov Accelerated GD
  - Newton's Method ($p_k = -H_k^{-1} \nabla f(x_k)$ via partial pivoting Gaussian elimination)
  - Damped Newton's Method (Armijo step length)
  - BFGS Quasi-Newton (rank-2 inverse Hessian approximation updates)

### Newton One-Step Convergence Theorem Demonstration
- Evaluates the theoretical property that on any strictly convex quadratic objective, pure Newton's method reaches the exact minimizer in **exactly 1 iteration**:
  $$p = -H^{-1} \nabla f(x_0) = -Q^{-1}(Q x_0 - b) = x^* - x_0 \implies x_1 = x_0 + p = x^*$$
- Demonstrates why this property does not generalize to non-quadratic nonlinear objectives (such as Rosenbrock) due to non-vanishing higher-order Taylor series remainder terms $O(\|p\|^3)$.

### Rosenbrock Banana Valley Calibration
- **Objective:** Non-convex curved valley parameter-calibration landscape:
  $$f(x_1, x_2) = (1 - x_1)^2 + 100(x_2 - x_1^2)^2, \quad x^* = (1, 1), \quad f(x^*) = 0$$
- Evaluates resistance to zig-zagging, curvature tracking, and convergence speed across first-order and second-order methods.

---

## 2. Directory Structure

```
NonlinearLab/
├── main.py                         # Interactive CLI & runner
├── README.md                       # Project documentation
├── problems/
│   ├── quadratic_benchmark.py      # Quadratic benchmark formulation & solvers
│   └── rosenbrock_calibration.py   # Rosenbrock objective, analytical gradient & Hessian
├── experiments/
│   └── run_experiments.py         # End-to-end experiment pipeline
├── visualization/
│   └── plots.py                    # 2D contour maps with trajectory overlays & semi-log plots
├── results/
│   ├── plots/                      # Generated figures (.png)
│   ├── histories/                  # Optimization histories (.csv)
│   └── reports/                    # Benchmark report (.txt)
└── tests/
    └── test_nonlinear.py           # Automated unit tests
```

---

## 3. Running NonlinearLab

```bash
# Run experiments and generate plots/reports:
python main.py --all

# Run automated unit tests:
python main.py --test
```
