# OptiScope: Optimization-Laboratory — Master Project Report & Codebase Documentation
**Course:** Optimization Techniques for Data Science (IMA314)  
**Deliverable:** Exhaustive Technical Project Report, Mathematical Formulations, Architecture & File-by-File Documentation  
**System Status:** 100% Passing Unit Tests (77/77 Tests, 0.62s) | Complete Benchmark Suite Verified  

---

## 1. Executive Summary & Engineering Architecture

The **Optimization-Laboratory** is an experimental numerical computing laboratory built from the ground up to investigate the mechanics, convergence behavior, and physical engineering utility of **23 classical and modern optimization methods**. 

Modern machine learning platforms frequently mask the computational trade-offs, numerical instabilities, and geometric realities of optimization behind high-level APIs (such as `scipy.optimize.minimize`, `torch.optim`, or `sklearn.linear_model`). In contrast, this laboratory enforces a **strict Zero Black-Box policy**:
- Every vector norm, matrix multiplication, Gaussian elimination step, and finite-difference derivative is implemented manually in Python.
- Every gradient descent variant, line search schedule, momentum scheme, adaptive coordinate-wise learning rate update, subgradient rule, Newton step, BFGS rank-2 update, and KKT block saddle-point solver is written from pure mathematics.
- NumPy is used exclusively as a numerical container (`np.ndarray`) for memory-efficient vectorized operations, slicing, and 2D grid generation; no external optimization or automated differentiation solvers are used anywhere in the codebase.

### 1.1 Architecture Blueprint

The platform is structured into a modular hierarchy comprising a shared numerical engine (`shared_optimization/`) and five domain-driven industrial engineering projects:

```
Optimization-Laboratory/
│
├── shared_optimization/                 # Central numerical library
│   ├── core/                            # Vector/matrix algebra, linear systems, derivatives, convergence
│   ├── algorithms/                      # 10 optimizer modules (all 23 methods)
│   ├── problems/                        # Mathematical problem definitions & loss objectives
│   ├── visualization/                   # Contour generators, Marching Squares, SVG & model plotters
│   └── tests/                           # Unit tests covering core, algorithms, and problems
│
├── EnergyOpt/                           # Project 1: Load forecasting & economic generation dispatch
│   ├── data/                            # Synthetic diurnal electricity generation
│   ├── problems/                        # Demand regression & quadratic dispatch formulation
│   ├── visualization/                   # Prediction overlay & marginal cost dispatch plots
│   └── experiments/                     # 6-optimizer benchmark & KKT dispatch solver
│
├── PredictMaint/                        # Project 2: Industrial sensor failure classification
│   ├── data/                            # Sensor telemetry simulation (vibration, temp, pressure)
│   ├── problems/                        # Logistic regression & BCE log-loss formulation
│   ├── visualization/                   # 2D physical decision boundaries & threshold trade-offs
│   └── experiments/                     # GD, AdaGrad, RMSProp, Adam comparison & threshold sweep
│
├── NonlinearLab/                        # Project 3: Smooth nonlinear optimization & Newton benchmarks
│   ├── problems/                        # Ill-conditioned quadratic & Rosenbrock valley
│   ├── visualization/                   # 2D filled contour maps & optimizer path overlays
│   └── experiments/                     # 7-optimizer benchmark & Newton One-Step proof verification
│
├── SparseModel/                         # Project 4: High-dimensional sparse feature selection
│   ├── data/                            # High-dimensional sensor dataset with true sparsity
│   ├── problems/                        # Lasso composite loss & nonsmooth subgradient calculus
│   ├── visualization/                   # Regularization paths & parameter recovery bar charts
│   └── experiments/                     # Subgradient norm convergence & lambda sweep
│
├── ResourceOpt/                         # Project 5: Constrained production & resource allocation
│   ├── problems/                        # Constrained quadratic manufacturing formulation
│   ├── visualization/                   # Polytope feasible region & constraint contour maps
│   └── experiments/                     # 5-scenario evaluation (Interior, Boundary, Infeasible, LICQ)
│
├── run_all.py                           # Master benchmark runner across all five projects
├── FINAL_REPORT.md                      # Executive numerical summary and comparison tables
└── COMPLETE_PROJECT_REPORT.md           # Master technical documentation (this file)
```

---

## 2. Master Algorithm-to-Problem Traceability Matrix (All 23 Methods)

Every algorithm studied in *Optimization Techniques for Data Science* is assigned to a real-world engineering challenge where its specific mathematical properties (smoothness, curvature, sparsity, constraints) are leveraged:

| # | Algorithm / Method | Assigned Project | Mathematical Formulation & Update Rule | Problem Role & Target Space | Implementation File | Experiment Runner | Output Artifacts |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | Gradient Descent (Fixed Step) | **NonlinearLab** | $x_{k+1} = x_k - \alpha \nabla f(x_k)$ | Baseline smooth descent; $x \in \mathbb{R}^2$ | `algorithms/gradient_descent.py` | `NonlinearLab/.../run_experiments.py` | `quad_fixed_gd.csv`, `quadratic_trajectories.png` |
| **2** | GD with Exact Line Search in 2D | **NonlinearLab** | $\alpha^* = \frac{g_k^T g_k}{g_k^T Q g_k}$ | Optimal step on quadratics; orthogonal zigzagging | `algorithms/line_search.py` | `NonlinearLab/.../run_experiments.py` | `quad_exact_line_search_gd.csv` |
| **3** | Backtracking Line Search | **NonlinearLab** | $\alpha \leftarrow \rho \alpha$ until Armijo holds | Step contraction for non-convex landscapes | `algorithms/line_search.py` | `NonlinearLab/.../run_experiments.py` | Iteration step logs |
| **4** | Armijo Acceptance Rule | **NonlinearLab** | $f(x + \alpha p) \le f(x) + c_1 \alpha \nabla f(x)^T p$ | Sufficient decrease guarantee ($c_1 = 10^{-4}$) | `algorithms/line_search.py` | `NonlinearLab/.../run_experiments.py` | Armijo condition check logs |
| **5** | GD with Armijo Backtracking | **NonlinearLab** | $x_{k+1} = x_k - \alpha_k \nabla f(x_k)$ | Adaptive descent on Rosenbrock banana valley | `algorithms/gradient_descent.py` | `NonlinearLab/.../run_experiments.py` | `rosen_armijo_gd.csv`, `rosenbrock_trajectories.png` |
| **6** | Momentum-based GD | **NonlinearLab** | $v_{k+1} = \beta v_k + \alpha \nabla f(x_k); x_{k+1} = x_k - v_{k+1}$ | Polyak Heavy-Ball; oscillation damping in valleys | `algorithms/momentum.py` | `NonlinearLab/.../run_experiments.py` | `quad_polyak_momentum.csv` |
| **7** | Nesterov GD (NGD) | **NonlinearLab** | $v_{k+1} = \beta v_k + \alpha \nabla f(x_k - \beta v_k)$ | Accelerated gradient with predictive look-ahead | `algorithms/nesterov.py` | `NonlinearLab/.../run_experiments.py` | `quad_nesterov_gd.csv` |
| **8** | Batch GD for Linear Regression | **EnergyOpt** | $w \leftarrow w - \frac{\alpha}{n} X^T (Xw + b - y)$ | Full-batch deterministic electricity demand forecast | `algorithms/regression_optimizers.py` | `EnergyOpt/.../run_experiments.py` | `energy_reg_batch_gd.csv`, `regression_loss_curves.png` |
| **9** | Stochastic GD for Linear Reg. | **EnergyOpt** | $w \leftarrow w - \alpha (x_i^T w + b - y_i) x_i$ | Real-time streaming hourly demand updates | `algorithms/regression_optimizers.py` | `EnergyOpt/.../run_experiments.py` | `energy_reg_sgd.csv` |
| **10** | Mini-batch SGD for Linear Reg. | **EnergyOpt** | $w \leftarrow w - \frac{\alpha}{|B|} X_B^T (X_B w + b - y_B)$ | Batch-size $m=32$ stochastic gradient descent | `algorithms/regression_optimizers.py` | `EnergyOpt/.../run_experiments.py` | `energy_reg_mini_batch_sgd.csv`, `demand_predictions.png` |
| **11** | Logistic Regression | **PredictMaint** | $\min -\frac{1}{n}\sum [y_i \ln p_i + (1-y_i)\ln(1-p_i)]$ | Telemetry failure classification with sigmoid $\sigma(z)$ | `algorithms/regression_optimizers.py` | `PredictMaint/.../run_experiments.py` | `maint_opt_logistic_reg_gd.csv`, `decision_boundary_vibration_temp.png` |
| **12** | AdaGrad | **EnergyOpt** & **PredictMaint** | $G_{k,i} = G_{k-1,i} + g_{k,i}^2; x_{k+1,i} = x_{k,i} - \frac{\alpha}{\sqrt{G_{k,i}}+\epsilon} g_{k,i}$ | Coordinate-wise scaling for sparse features | `algorithms/adaptive_optimizers.py` | Both project runners | Convergence tables & loss logs |
| **13** | RMSProp | **EnergyOpt** & **PredictMaint** | $v_{k,i} = \beta v_{k-1,i} + (1-\beta) g_{k,i}^2$ | Moving-average scaling preventing learning rate stall | `algorithms/adaptive_optimizers.py` | Both project runners | Metric comparisons & ROC curves |
| **14** | Adam | **EnergyOpt** & **PredictMaint** | $m_k, v_k \rightarrow \hat{m}_k, \hat{v}_k; x_{k+1} = x_k - \frac{\alpha}{\sqrt{\hat{v}_k}+\epsilon}\hat{m}_k$ | Combined 1st & 2nd moments with bias corrections | `algorithms/adaptive_optimizers.py` | Both project runners | Confusion matrices & benchmark curves |
| **15** | Subgradient Method (Functions) | **SparseModel** | $x_{k+1} = x_k - \frac{\alpha_0}{\sqrt{k}} g_k, g_k \in \partial \|x\|_2$ | Minimization of nonsmooth Euclidean norm | `algorithms/subgradient.py` | `SparseModel/.../run_experiments.py` | `general_subgradient_norm.csv`, `subgradient_convergence.png` |
| **16** | Subgradient for Lasso Loss | **SparseModel** | $g_j = \frac{1}{n} X_j^T(Xw-y) + \lambda \text{sgn}(w_j)$ | Composite $L_2 + L_1$ loss without zero-snapping | `algorithms/subgradient.py` | `SparseModel/.../run_experiments.py` | `lasso_regularization_path.png`, `coefficient_comparison.png` |
| **17** | Newton's Method | **NonlinearLab** | $p_k = -[\nabla^2 f(x_k)]^{-1} \nabla f(x_k); x_{k+1} = x_k + p_k$ | Exact 2nd-order curvature via Gaussian elimination | `algorithms/newton.py` | `NonlinearLab/.../run_experiments.py` | `rosen_newton_method.csv` |
| **18** | Damped Newton's Method | **NonlinearLab** | $x_{k+1} = x_k + \alpha_k p_k$ with Armijo step | Globally convergent Newton method | `algorithms/newton.py` | `NonlinearLab/.../run_experiments.py` | `rosen_damped_newton.csv` |
| **19** | Newton One-Step Demonstration | **NonlinearLab** | $x_1 = x_0 - [\nabla^2 f]^{-1} \nabla f(x_0) \equiv x^*$ | Proof & verification on strictly convex quadratics | `algorithms/newton.py` | `NonlinearLab/.../run_experiments.py` | Theoretical log (`nonlinear_lab_summary.txt`) |
| **20** | BFGS Quasi-Newton | **NonlinearLab** | $H_{k+1} = (I - \rho s y^T) H_k (I - \rho y s^T) + \rho s s^T$ | Rank-2 inverse Hessian update without inversion | `algorithms/bfgs.py` | `NonlinearLab/.../run_experiments.py` | `rosen_bfgs.csv`, `rosenbrock_trajectories.png` |
| **21** | Lagrange Multiplier Solver | **EnergyOpt** & **ResourceOpt** | $\begin{bmatrix} Q & A^T \\ A & 0 \end{bmatrix} \begin{bmatrix} x \\ \lambda \end{bmatrix} = \begin{bmatrix} -c \\ b \end{bmatrix}$ | Exact saddle-point solution for equality constraints | `algorithms/constrained_optimization.py` | Both project runners | `generation_dispatch.png`, `production_polytope.png` |
| **22** | KKT Conditions Evaluator | **EnergyOpt** & **ResourceOpt** | Stationarity, Primal/Dual Feasibility, Slackness | Complete Karush-Kuhn-Tucker certificate verification | `algorithms/constrained_optimization.py` | Both project runners | `kkt_residuals_summary.png`, diagnostic logs |
| **23** | LICQ Checker | **EnergyOpt** & **ResourceOpt** | $\det(J_{\mathcal{A}} J_{\mathcal{A}}^T) > 0, \text{rank}(J_{\mathcal{A}}) = m$ | Linear Independence Constraint Qualification verification | `algorithms/constrained_optimization.py` | Both project runners | LICQ rank diagnostics across 5 scenarios |

---

## 3. Shared Optimization Library (`shared_optimization/`): File-by-File Technical Breakdown

### 3.1 Core Numerical Module (`shared_optimization/core/`)

#### 1. [vectors.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/shared_optimization/core/vectors.py)
- **Why We Need This:** Provides robust, type-checked vector operations ($L_2$ norm, inner product, linear combinations) supporting standard Python sequences and NumPy arrays without calling opaque library subroutines.
- **Mathematical Formulations:**
  - Dot Product: $\langle u, v \rangle = \sum_{i=1}^n u_i v_i$
  - Euclidean $L_2$ Norm: $\|u\|_2 = \sqrt{\sum_{i=1}^n u_i^2}$
  - Vector Addition & Scaling: $w = \alpha u + \beta v$ where $w_i = \alpha u_i + \beta v_i$
- **Representative Code Snippet:**
  ```python
  def norm_l2(v: Sequence[Number]) -> float:
      """Computes Euclidean norm ||v||_2 = sqrt(sum(v_i^2))."""
      v_val = validate_vector(v, "v")
      return math.sqrt(sum(x * x for x in v_val))

  def dot_product(u: Sequence[Number], v: Sequence[Number]) -> float:
      """Computes dot product <u, v> = sum(u_i * v_i)."""
      u_val = validate_vector(u, "u")
      v_val = validate_vector(v, "v")
      if len(u_val) != len(v_val):
          raise ValueError(f"Dimension mismatch: {len(u_val)} != {len(v_val)}")
      return sum(a * b for a, b in zip(u_val, v_val))
  ```

#### 2. [matrices.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/shared_optimization/core/matrices.py)
- **Why We Need This:** Encapsulates matrix-vector multiplication $y = Ax$, matrix-matrix multiplication $C = AB$, matrix transposition $A^T$, and symmetric identity matrix construction required by Newton systems and BFGS updates.
- **Mathematical Formulations:**
  - Matrix-Vector: $y_i = \sum_{j=1}^n A_{ij} x_j$
  - Matrix-Matrix: $C_{ij} = \sum_{k=1}^p A_{ik} B_{kj}$
  - Transposition: $(A^T)_{ji} = A_{ij}$
- **Representative Code Snippet:**
  ```python
  def mat_vec_mul(A: Sequence[Sequence[Number]], x: Sequence[Number]) -> List[float]:
      A_val = validate_matrix(A, "A")
      x_val = validate_vector(x, "x")
      m, n = matrix_shape(A_val)
      if n != len(x_val):
          raise ValueError(f"Matrix cols {n} != vector length {len(x_val)}")
      return [sum(A_val[i][j] * x_val[j] for j in range(n)) for i in range(m)]
  ```

#### 3. [linear_systems.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/shared_optimization/core/linear_systems.py)
- **Why We Need This:** Linear equation systems arise constantly in optimization (solving Newton directions $H p = -g$ and KKT block systems). This file implements **Gaussian Elimination with Partial Pivoting (GEPP)**, back-substitution, Gauss-Jordan matrix inversion, and determinant computation from scratch.
- **Mathematical Formulations:**
  - Forward Elimination with Row Swaps: Pivot row $p = \arg\max_{i \ge k} |A_{ik}|$. Row multiplier $m_{ik} = \frac{A_{ik}}{A_{kk}}$, $A_{i, \cdot} \leftarrow A_{i, \cdot} - m_{ik} A_{k, \cdot}$.
  - Back-Substitution: $x_i = \frac{1}{U_{ii}} \left(c_i - \sum_{j=i+1}^n U_{ij} x_j\right)$.
  - Determinant: $\det(A) = (-1)^s \prod_{k=1}^n U_{kk}$ where $s$ is the number of row permutations.
- **Representative Code Snippet:**
  ```python
  def solve_linear_system(A: Sequence[Sequence[Number]], b: Sequence[Number], tol: float = 1e-14) -> List[float]:
      U = [list(row) for row in A]
      c = list(b)
      n = len(c)
      for k in range(n - 1):
          # Partial pivoting
          max_val = abs(U[k][k])
          pivot_row = k
          for p in range(k + 1, n):
              if abs(U[p][k]) > max_val:
                  max_val = abs(U[p][k])
                  pivot_row = p
          if max_val < tol:
              raise SingularMatrixError(f"Singular matrix encountered at col {k}")
          if pivot_row != k:
              U[k], U[pivot_row] = U[pivot_row], U[k]
              c[k], c[pivot_row] = c[pivot_row], c[k]
          for i in range(k + 1, n):
              factor = U[i][k] / U[k][k]
              for j in range(k, n):
                  U[i][j] -= factor * U[k][j]
              c[i] -= factor * c[k]
      # Back-substitution
      x = [0.0] * n
      for i in range(n - 1, -1, -1):
          s = sum(U[i][j] * x[j] for j in range(i + 1, n))
          x[i] = (c[i] - s) / U[i][i]
      return x
  ```

#### 4. [derivatives.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/shared_optimization/core/derivatives.py)
- **Why We Need This:** Implements central finite-difference gradient and Hessian numerical approximations ($O(h^2)$ truncation error), used to verify analytical gradients and Hessians across all loss objectives.
- **Mathematical Formulations:**
  - Finite-Difference Gradient: $\frac{\partial f}{\partial x_i} \approx \frac{f(x + h e_i) - f(x - h e_i)}{2h}$
  - Diagonal Hessian: $\frac{\partial^2 f}{\partial x_i^2} \approx \frac{f(x + h e_i) - 2f(x) + f(x - h e_i)}{h^2}$
  - Off-Diagonal Cross Hessian: $\frac{\partial^2 f}{\partial x_i \partial x_j} \approx \frac{f(x + h e_i + h e_j) - f(x + h e_i - h e_j) - f(x - h e_i + h e_j) + f(x - h e_i - h e_j)}{4h^2}$
- **Representative Code Snippet:**
  ```python
  def finite_difference_gradient(f: Callable[[List[float]], float], x: Sequence[Number], h: float = 1e-5) -> List[float]:
      x_list = list(x)
      grad = []
      for i in range(len(x_list)):
          x_plus = list(x_list); x_plus[i] += h
          x_minus = list(x_list); x_minus[i] -= h
          grad.append((f(x_plus) - f(x_minus)) / (2.0 * h))
      return grad
  ```

#### 5. [convergence.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/shared_optimization/core/convergence.py)
- **Why We Need This:** Provides standardized dataclasses (`IterationRecord`, `OptimizationResult`, `ConvergenceCriterion`) that capture full trajectory states (iteration index, parameter vector $x_k$, objective value $f(x_k)$, gradient norm $\|\nabla f(x_k)\|_2$, execution time). Also implements robust multi-criteria convergence testing:
  - Gradient stationarity: $\|\nabla f(x_k)\|_2 < \epsilon_g$
  - Step progress: $\|x_{k+1} - x_k\|_2 < \epsilon_x$
  - Relative objective change: $|f(x_{k+1}) - f(x_k)| / (|f(x_k)| + 1) < \epsilon_f$

---

### 3.2 Optimization Algorithms Module (`shared_optimization/algorithms/`)

#### 6. [gradient_descent.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/shared_optimization/algorithms/gradient_descent.py)
- **Why We Need This:** Implements the foundational first-order optimization algorithms: Fixed Step Gradient Descent, Exact Line Search Gradient Descent (specialized for quadratics), and Armijo Backtracking Gradient Descent.
- **Mathematical Formulations:**
  - Fixed Step: $x_{k+1} = x_k - \alpha \nabla f(x_k)$
  - Exact Quadratic Line Search: For $f(x) = \frac{1}{2} x^T Q x - b^T x$, setting $\frac{d}{d\alpha} f(x - \alpha g) = 0$ yields:
    $$\alpha^* = \frac{g_k^T g_k}{g_k^T Q g_k}$$
  - Armijo Backtracking: At each iteration, finds the smallest integer $m \ge 0$ such that $\alpha = \alpha_0 \rho^m$ satisfies:
    $$f(x_k - \alpha \nabla f(x_k)) \le f(x_k) - c_1 \alpha \|\nabla f(x_k)\|_2^2$$
- **Representative Code Snippet:**
  ```python
  def gradient_descent_armijo(f, grad_f, x0, alpha_init=1.0, rho=0.5, c1=1e-4, criterion=None):
      crit = criterion or ConvergenceCriterion()
      x = list(x0)
      for k in range(crit.max_iters):
          g = grad_f(x)
          if norm_l2(g) < crit.grad_tol: break
          p = scalar_mul(-1.0, g)
          alpha, _ = backtracking_line_search(f, grad_f, x, p, alpha_init=alpha_init, rho=rho, c1=c1)
          x = vec_add(x, scalar_mul(alpha, p))
  ```

#### 7. [line_search.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/shared_optimization/algorithms/line_search.py)
- **Why We Need This:** Decouples step-length calculation from search directions. Provides standalone functions for closed-form quadratic line search, Armijo condition verification, and geometric contraction backtracking.
- **Mathematical Formulations:**
  - Armijo Condition: $f(x + \alpha p) \le f(x) + c_1 \alpha \nabla f(x)^T p$, where $p$ is a descent direction ($\nabla f(x)^T p < 0$).
  - Backtracking Contraction: $\alpha \leftarrow \rho \cdot \alpha$ with contraction ratio $\rho \in (0, 1)$ (typically $0.5$).
- **Representative Code Snippet:**
  ```python
  def backtracking_line_search(f, grad_f, x, p, alpha_init=1.0, rho=0.5, c1=1e-4, max_iters=50):
      alpha = alpha_init
      f_val = f(x)
      g = grad_f(x)
      for it in range(max_iters):
          if armijo_condition(f, x, p, alpha, c1, f_val, g):
              return alpha, it + 1
          alpha *= rho
      return alpha, max_iters
  ```

#### 8. [momentum.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/shared_optimization/algorithms/momentum.py)
- **Why We Need This:** Standard gradient descent oscillates severely across ill-conditioned ravines where the eigenvalues of the Hessian satisfy $\lambda_{\max} \gg \lambda_{\min}$. Polyak Heavy-Ball momentum accumulates velocity along consistent directions, damping transverse oscillations.
- **Mathematical Formulation:**
  $$v_{k+1} = \beta v_k + \alpha \nabla f(x_k)$$
  $$x_{k+1} = x_k - v_{k+1}$$
  where $\beta \in [0, 1)$ is the momentum decay factor (typically $0.9$).
- **Representative Code Snippet:**
  ```python
  def momentum_optimizer(f, grad_f, x0, learning_rate=0.01, beta=0.9, criterion=None):
      v = [0.0] * len(x0)
      x = list(x0)
      for k in range(criterion.max_iters):
          g = grad_f(x)
          v = [beta * v[i] + learning_rate * g[i] for i in range(len(x))]
          x = [x[i] - v[i] for i in range(len(x))]
  ```

#### 9. [nesterov.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/shared_optimization/algorithms/nesterov.py)
- **Why We Need This:** Polyak momentum computes the gradient at the current position $x_k$. Nesterov Accelerated Gradient (NAG) computes a predictive "look-ahead" gradient at the anticipated position $x_k - \beta v_k$, providing proactive braking when approaching a local minimum and achieving the theoretical optimal convergence rate $O(1/k^2)$ for convex smooth functions.
- **Mathematical Formulation:**
  $$x_{\text{look}} = x_k - \beta v_k$$
  $$v_{k+1} = \beta v_k + \alpha \nabla f(x_{\text{look}})$$
  $$x_{k+1} = x_k - v_{k+1}$$
- **Representative Code Snippet:**
  ```python
  def nesterov_optimizer(f, grad_f, x0, learning_rate=0.01, beta=0.9, criterion=None):
      v = [0.0] * len(x0)
      x = list(x0)
      for k in range(criterion.max_iters):
          x_look = [x[i] - beta * v[i] for i in range(len(x))]
          g_look = grad_f(x_look)
          v = [beta * v[i] + learning_rate * g_look[i] for i in range(len(x))]
          x = [x[i] - v[i] for i in range(len(x))]
  ```

#### 10. [regression_optimizers.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/shared_optimization/algorithms/regression_optimizers.py)
- **Why We Need This:** Implements specialized regression learning schemes: Batch Gradient Descent, Stochastic Gradient Descent (sample-by-sample update), Mini-batch SGD (subset $m=32$), and Logistic Regression for binary classification.
- **Mathematical Formulations:**
  - Batch GD (MSE): $\nabla_w J = \frac{1}{n} X^T (Xw + b - y)$, $\nabla_b J = \frac{1}{n} \mathbf{1}^T (Xw + b - y)$
  - SGD: For random sample $i$, $w \leftarrow w - \alpha (w^T x_i + b - y_i) x_i$, $b \leftarrow b - \alpha (w^T x_i + b - y_i)$
  - Mini-Batch SGD: For batch $B \subset \{1,\dots,n\}$, $w \leftarrow w - \frac{\alpha}{|B|} X_B^T (X_B w + b - y_B)$
  - Logistic Regression (BCE Loss):
    $$p_i = \sigma(w^T x_i + b) = \frac{1}{1 + e^{-(w^T x_i + b)}}$$
    $$\nabla_w J = \frac{1}{n} X^T (p - y), \quad \nabla_b J = \frac{1}{n} \sum_{i=1}^n (p_i - y_i)$$
- **Representative Code Snippet:**
  ```python
  def mini_batch_sgd_linear_regression(dataset, batch_size=32, learning_rate=0.01, epochs=50):
      w = [0.0] * dataset.n_features
      b = 0.0
      n = dataset.n_samples
      indices = list(range(n))
      for epoch in range(epochs):
          # In-place deterministic shuffle
          for i in range(n - 1, 0, -1):
              j = (epoch * 31 + i * 17) % (i + 1)
              indices[i], indices[j] = indices[j], indices[i]
          for start in range(0, n, batch_size):
              batch_idx = indices[start:start + batch_size]
              # Compute gradient over batch and update
              ...
  ```

#### 11. [adaptive_optimizers.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/shared_optimization/algorithms/adaptive_optimizers.py)
- **Why We Need This:** Features in real-world engineering datasets have vastly different units and scales. Fixed learning rates either diverge along high-curvature coordinates or stall along flat coordinates. This module implements **AdaGrad**, **RMSProp**, and **Adam**.
- **Mathematical Formulations:**
  - AdaGrad:
    $$G_{k,i} = G_{k-1,i} + g_{k,i}^2, \quad x_{k+1,i} = x_{k,i} - \frac{\alpha}{\sqrt{G_{k,i}} + \epsilon} g_{k,i}$$
  - RMSProp:
    $$v_{k,i} = \beta v_{k-1,i} + (1 - \beta) g_{k,i}^2, \quad x_{k+1,i} = x_{k,i} - \frac{\alpha}{\sqrt{v_{k,i}} + \epsilon} g_{k,i}$$
  - Adam:
    $$m_k = \beta_1 m_{k-1} + (1 - \beta_1) g_k, \quad v_k = \beta_2 v_{k-1} + (1 - \beta_2) g_k^2$$
    $$\hat{m}_k = \frac{m_k}{1 - \beta_1^k}, \quad \hat{v}_k = \frac{v_k}{1 - \beta_2^k}$$
    $$x_{k+1} = x_k - \frac{\alpha}{\sqrt{\hat{v}_k} + \epsilon} \hat{m}_k$$
- **Representative Code Snippet:**
  ```python
  def adam_optimizer(f, grad_f, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, criterion=None):
      m = [0.0] * len(x0)
      v = [0.0] * len(x0)
      x = list(x0)
      for k in range(1, criterion.max_iters + 1):
          g = grad_f(x)
          step = [0.0] * len(x)
          for i in range(len(x)):
              m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
              v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i]
              m_hat = m[i] / (1.0 - beta1 ** k)
              v_hat = v[i] / (1.0 - beta2 ** k)
              step[i] = - (learning_rate / (math.sqrt(v_hat) + epsilon)) * m_hat
          x = [x[i] + step[i] for i in range(len(x))]
  ```

#### 12. [subgradient.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/shared_optimization/algorithms/subgradient.py)
- **Why We Need This:** Classical gradient descent requires differentiable objectives. $L_1$ regularization (Lasso) and Euclidean norms have points of non-differentiability (kinks). This module implements subgradient calculus with diminishing step sizes to ensure asymptotic convergence on nonsmooth convex functions.
- **Mathematical Formulations:**
  - Subgradient Definition: $g \in \partial f(x) \iff f(y) \ge f(x) + g^T (y - x) \quad \forall y$.
  - Subdifferential of Absolute Value:
    $$\partial |w| = \begin{cases} \{+1\}, & w > 0 \\ \{-1\}, & w < 0 \\ [-1, 1], & w = 0 \end{cases}$$
  - Diminishing Step Size: $\alpha_k = \frac{\alpha_0}{\sqrt{k}}$, satisfying:
    $$\sum_{k=1}^\infty \alpha_k = \infty, \quad \sum_{k=1}^\infty \alpha_k^2 < \infty$$
  - Subgradient Update: $x_{k+1} = x_k - \alpha_k g_k$, tracking $x_{\text{best}} = \arg\min_{i \le k} f(x_i)$.
- **Representative Code Snippet:**
  ```python
  def subgradient_method(f, subgrad_f, x0, step_size_init=0.5, max_iters=500, tol=1e-4):
      x = list(x0)
      x_best = list(x)
      f_best = f(x)
      for k in range(1, max_iters + 1):
          g = subgrad_f(x)
          if norm_l2(g) < tol: break
          alpha_k = step_size_init / (k ** 0.5)
          x = [x[i] - alpha_k * g[i] for i in range(len(x))]
          f_val = f(x)
          if f_val < f_best:
              f_best = f_val
              x_best = list(x)
  ```

#### 13. [newton.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/shared_optimization/algorithms/newton.py)
- **Why We Need This:** Second-order methods use curvature information from the Hessian matrix $\nabla^2 f(x)$ to attain quadratic local convergence ($O(\epsilon^2)$), escaping the slow zig-zagging of gradient descent.
- **Mathematical Formulations:**
  - Pure Newton Direction: Solves the linear system:
    $$\nabla^2 f(x_k) p_k = -\nabla f(x_k) \implies x_{k+1} = x_k + p_k$$
  - Damped Newton Method: To ensure global convergence when far from the optimum, Newton's direction is combined with Armijo line search:
    $$x_{k+1} = x_k + \alpha_k p_k$$
  - Newton One-Step Theorem: For any strictly convex quadratic $f(x) = \frac{1}{2} x^T Q x - b^T x + c$ with $Q \succ 0$:
    $$\nabla f(x_0) = Q x_0 - b, \quad \nabla^2 f(x_0) = Q$$
    $$p_0 = -Q^{-1} (Q x_0 - b) = -x_0 + Q^{-1} b$$
    $$x_1 = x_0 + p_0 = Q^{-1} b \equiv x^*$$
    The exact minimizer is reached in **exactly one step** from any initial point $x_0 \in \mathbb{R}^n$.
- **Representative Code Snippet:**
  ```python
  def newton_method(f, grad_f, hess_f, x0, criterion=None):
      x = list(x0)
      for k in range(criterion.max_iters):
          g = grad_f(x)
          if norm_l2(g) < criterion.grad_tol: break
          H = hess_f(x)
          # Solve H * p = -g via Gaussian Elimination with Partial Pivoting
          neg_g = [-val for val in g]
          p = solve_linear_system(H, neg_g)
          x = vec_add(x, p)
  ```

#### 14. [bfgs.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/shared_optimization/algorithms/bfgs.py)
- **Why We Need This:** Newton's method requires $O(n^3)$ operations to invert the Hessian at each step and requires computing $\frac{n(n+1)}{2}$ second derivatives. BFGS maintains an approximation of the inverse Hessian matrix $H_k \approx (\nabla^2 f(x_k))^{-1}$ updated via rank-2 outer products with only $O(n^2)$ complexity.
- **Mathematical Formulation:**
  Let $s_k = x_{k+1} - x_k$ and $y_k = \nabla f(x_{k+1}) - \nabla f(x_k)$. With $\rho_k = \frac{1}{y_k^T s_k}$:
  $$H_{k+1} = (I - \rho_k s_k y_k^T) H_k (I - \rho_k y_k s_k^T) + \rho_k s_k s_k^T$$
  Guarantees positive definiteness of $H_{k+1}$ whenever $y_k^T s_k > 0$ (enforced by Armijo backtracking).
- **Representative Code Snippet:**
  ```python
  # BFGS Sherman-Morrison-Woodbury Inverse Hessian Update
  s = vec_sub(x_next, x)
  y = vec_sub(g_next, g)
  ys = dot_product(y, s)
  if ys > 1e-10:
      rho = 1.0 / ys
      # V = (I - rho * s * y^T)
      sy = outer_product(s, y)
      V1 = mat_sub(I, mat_scalar_mul(sy, rho))
      ys_mat = outer_product(y, s)
      V2 = mat_sub(I, mat_scalar_mul(ys_mat, rho))
      # H_{k+1} = V1 * H * V2 + rho * (s * s^T)
      H = mat_add(mat_mat_mul(mat_mat_mul(V1, H), V2), mat_scalar_mul(outer_product(s, s), rho))
  ```

#### 15. [constrained_optimization.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion Project/Optimization-Laboratory/shared_optimization/algorithms/constrained_optimization.py)
- **Why We Need This:** Real-world engineering problems feature physical limits (generator capacities, warehouse volume, mass balance). This module provides:
  1. An exact **Lagrange Multiplier Solver** for equality-constrained quadratic programs.
  2. A comprehensive **KKT Conditions Evaluator** checking all four Karush-Kuhn-Tucker criteria.
  3. A **Linear Independence Constraint Qualification (LICQ)** checker.
- **Mathematical Formulations:**
  - Equality Constrained QP:
    $$\min_{x} \frac{1}{2} x^T Q x + c^T x \quad \text{s.t.} \quad A x = b$$
    Forms the block saddle-point KKT system:
    $$\begin{bmatrix} Q & A^T \\ A & 0 \end{bmatrix} \begin{bmatrix} x^* \\ \lambda^* \end{bmatrix} = \begin{bmatrix} -c \\ b \end{bmatrix}$$
  - Karush-Kuhn-Tucker (KKT) Conditions for general inequality and equality constraints:
    1. **Stationarity:** $\nabla f(x^*) + \sum_{i=1}^m \lambda_i^* \nabla h_i(x^*) + \sum_{j=1}^p \mu_j^* \nabla g_j(x^*) = 0$
    2. **Primal Feasibility:** $h_i(x^*) = 0 \; \forall i, \quad g_j(x^*) \le 0 \; \forall j$
    3. **Dual Feasibility:** $\mu_j^* \ge 0 \; \forall j$
    4. **Complementary Slackness:** $\mu_j^* g_j(x^*) = 0 \; \forall j$
  - Linear Independence Constraint Qualification (LICQ):
    For active constraints $\mathcal{A}(x^*) = \{i : h_i(x^*) = 0\} \cup \{j : g_j(x^*) = 0\}$, the Jacobian matrix of active constraint gradients $J_{\mathcal{A}}$ must have full row rank:
    $$\text{rank}(J_{\mathcal{A}}) = |\mathcal{A}| \iff \det(J_{\mathcal{A}} J_{\mathcal{A}}^T) > 0$$
- **Representative Code Snippet:**
  ```python
  def check_licq(active_gradients, tol=1e-10):
      # Gram matrix G = J * J^T
      m = len(active_gradients)
      G = [[dot_product(active_gradients[i], active_gradients[j]) for j in range(m)] for i in range(m)]
      det_G = matrix_determinant(G)
      licq_holds = abs(det_G) > tol
      return {"licq_holds": licq_holds, "gram_determinant": det_G, "active_count": m}
  ```

---

### 3.3 Problem Formulations Module (`shared_optimization/problems/`)

- [quadratic.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/shared_optimization/problems/quadratic.py): Implements $f(x) = \frac{1}{2} x^T Q x - b^T x$, $\nabla f(x) = Q x - b$, and $\nabla^2 f(x) = Q$.
- [rosenbrock.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/shared_optimization/problems/rosenbrock.py): Implements the non-convex banana valley $f(x_1, x_2) = (1 - x_1)^2 + 100(x_2 - x_1^2)^2$ with analytical gradient and $2 \times 2$ Hessian:
  $$\nabla f = \begin{bmatrix} -2(1 - x_1) - 400 x_1 (x_2 - x_1^2) \\ 200(x_2 - x_1^2) \end{bmatrix}, \quad \nabla^2 f = \begin{bmatrix} 2 - 400 x_2 + 1200 x_1^2 & -400 x_1 \\ -400 x_1 & 200 \end{bmatrix}$$
- [linear_regression.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/shared_optimization/problems/linear_regression.py): Implements MSE loss, analytical vector gradients, and standardization utilities.
- [logistic_regression.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/shared_optimization/problems/logistic_regression.py): Implements Binary Cross-Entropy (BCE) loss with numerically stable branch-free evaluation to prevent floating-point underflow/overflow:
  $$\sigma(z) = \begin{cases} \frac{1}{1 + e^{-z}}, & z \ge 0 \\ \frac{e^z}{1 + e^z}, & z < 0 \end{cases}$$
- [lasso.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/shared_optimization/problems/lasso.py): Implements composite loss $J(w) = \frac{1}{2n}\|Xw - y\|_2^2 + \lambda \|w\|_1$ and exact subgradient vectors without artificial thresholding.
- [constrained.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/shared_optimization/problems/constrained.py): Data structure defining general nonlinear constrained optimization problems with equality constraints $h_i(x) = 0$ and inequality constraints $g_j(x) \le 0$.

---

### 3.4 Visualization Module (`shared_optimization/visualization/`)

- [marching_squares.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/shared_optimization/visualization/marching_squares.py): Implements the 2D Marching Squares isocontour polygon extraction algorithm from scratch without external dependencies.
- [contour.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/shared_optimization/visualization/contour.py): Generates 2D filled contour maps, draws trajectory paths for optimization algorithms, and renders constraint boundaries and feasible regions.
- [model_plots.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/shared_optimization/visualization/model_plots.py): Generates regression training/validation loss curves, scatter predictions, and confusion matrix visualizations.
- [svg_renderer.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/shared_optimization/visualization/svg_renderer.py): Standalone vector graphics engine that outputs raw XML `<svg>` strings for web-native, zero-dependency visual inspection.

---

## 4. Project-by-Project Exhaustive Analysis & Empirical Findings

### 4.1 Project 1: EnergyOpt — Electricity Demand Forecasting & Economic Dispatch

#### Problem Motivation & Real-World Utility
Electrical grid operators (ISOs) face two coupled challenges:
1. Predicting aggregate electricity demand hours in advance from environmental telemetry (ambient temperature, hour of day, calendar factors).
2. Dispatching heterogeneous power generation units (hydroelectric, combined-cycle natural gas, coal) to meet demand at minimal total cost while respecting generator power capacities.

#### Mathematical Formulations

##### Part 1: Demand Prediction via Linear Regression
We formulate demand forecasting as multivariate linear regression over $n = 500$ hourly observations with $d = 5$ features (Hour of day, Temperature, Squared temperature, Weekend indicator, Lag-1 demand):
$$\hat{y}_i = w^T x_i + b$$
$$J(w, b) = \frac{1}{2n} \sum_{i=1}^n (\hat{y}_i - y_i)^2$$

##### Part 2: Economic Dispatch with Quadratic Fuel Costs
Given the predicted peak demand $D = 317.89\text{ MW}$, we allocate power generation across 3 generation facilities $P = [P_1, P_2, P_3]^T$ minimizing quadratic generation costs:
$$\min_{P_1, P_2, P_3} C(P_1, P_2, P_3) = \sum_{i=1}^3 \left(\frac{1}{2} a_i P_i^2 + b_i P_i + c_i\right)$$
$$\text{subject to } P_1 + P_2 + P_3 = D, \quad P_i^{\min} \le P_i \le P_i^{\max}$$
The Lagrangian function is:
$$\mathcal{L}(P, \lambda) = \sum_{i=1}^3 \left(\frac{1}{2} a_i P_i^2 + b_i P_i + c_i\right) + \lambda \left(D - \sum_{i=1}^3 P_i\right)$$
The first-order necessary condition (stationarity) is:
$$\frac{\partial \mathcal{L}}{\partial P_i} = a_i P_i + b_i - \lambda = 0 \implies P_i^*(\lambda) = \frac{\lambda - b_i}{a_i}$$
This demonstrates that **all generators must operate at identical marginal cost $\lambda$** (the System Marginal Price).

#### Empirical Numerical Results

```
========================================================================================
                      ENERGYOPT: DEMAND PREDICTION BENCHMARK
========================================================================================
Optimizer             Train RMSE    Val RMSE     Val R^2     Iterations    Wall Time
----------------------------------------------------------------------------------------
Batch GD               19.390 MW    17.112 MW     0.6932        200         2.624 s
SGD                    23.615 MW    17.126 MW     0.6927         60         2.196 s
Mini-batch SGD (m=32)  19.056 MW    16.475 MW     0.7156         60         2.158 s
AdaGrad                23.127 MW    32.868 MW    -0.1318        200         0.038 s
RMSProp                25.668 MW    39.039 MW    -0.5967        200         0.044 s
Adam                   23.854 MW    35.100 MW    -0.2908        200         0.037 s
========================================================================================
```

#### Dispatch Optimization & KKT Validation
- **Target Demand Dispatched:** $317.89\text{ MW}$ (Mini-batch SGD forecast)
- **Optimal Unit Dispatch:**
  - Generator 1 (Hydro): $P_1^* = 150.49\text{ MW}$ (Cost: $0.04 P_1^2 + 12 P_1$)
  - Generator 2 (Gas): $P_2^* = 100.40\text{ MW}$ (Cost: $0.06 P_2^2 + 14 P_2$)
  - Generator 3 (Coal): $P_3^* = 67.00\text{ MW}$ (Cost: $0.09 P_3^2 + 18 P_3$)
- **System Marginal Price ($\lambda^*$):** $-\$24.04\text{/MWh}$
- **Stationarity Residual:** $\|\nabla_P \mathcal{L}\|_2 = 1.066 \times 10^{-14}$ (Passed)
- **Primal Feasibility Violation:** $|\sum P_i^* - D| = 0.000\text{ MW}$ (Passed)
- **LICQ Determinant:** $\det(J_A J_A^T) = 3.00$ (Full rank, Passed)
- **Financial Forecast Sensitivity:** A $+3.20\text{ MW}$ prediction error incurred an operational cost penalty of $\$76.65/\text{h}$ ($\approx \$671,454/\text{year}$).

---

### 4.2 Project 2: PredictMaint — Industrial Telemetry Failure Classification

#### Problem Motivation & Real-World Utility
Unplanned industrial equipment breakdowns in manufacturing and petrochemical plants lead to catastrophic downtime. Telemetry sensors stream high-frequency data: Vibration RMS ($\text{mm/s}$), Operating Temperature ($^\circ\text{C}$), Service Hours, Differential Pressure ($\text{bar}$), and Mechanical Load (%). We predict binary failure risk $y \in \{0, 1\}$.

#### Mathematical Formulation
Binary Logistic Regression with sigmoid link:
$$P(y_i = 1 \mid x_i) = \sigma(w^T x_i + b) = \frac{1}{1 + e^{-(w^T x_i + b)}}$$
Minimizing the empirical Binary Cross-Entropy (BCE) log-loss:
$$J(w, b) = -\frac{1}{n} \sum_{i=1}^n \left[y_i \ln(p_i) + (1 - y_i) \ln(1 - p_i)\right]$$
Numerically stable gradients:
$$\nabla_w J = \frac{1}{n} X^T (p - y), \quad \nabla_b J = \frac{1}{n} \sum_{i=1}^n (p_i - y_i)$$

#### Empirical Classification Results

```
========================================================================================
                   PREDICTMAINT: FAILURE CLASSIFICATION BENCHMARK
========================================================================================
Optimizer          Val BCE Loss    Accuracy    Precision     Recall     F1 Score   Time
----------------------------------------------------------------------------------------
Logistic Reg (GD)    0.3669         85.4%        77.8%        50.9%      0.615    5.692s
AdaGrad              0.3625         85.4%        75.0%        54.5%      0.632    0.006s
RMSProp              0.3630         85.4%        77.8%        50.9%      0.615    0.059s
Adam                 0.3625         85.4%        75.0%        54.5%      0.632    0.055s
========================================================================================
```

#### Confusion Matrix & Threshold Decision Sweeps
At the standard decision threshold $\tau = 0.50$:
$$\begin{bmatrix} \text{True Negatives (TN)} & \text{False Positives (FP)} \\ \text{False Negatives (FN)} & \text{True Positives (TP)} \end{bmatrix} = \begin{bmatrix} 175 & 10 \\ 25 & 30 \end{bmatrix}$$
Operational Trade-Off Curve across $\tau \in [0.10, 0.90]$:
- $\tau = 0.15$: Prioritizes safety. Recall surges to **87.3%** ($F_1 = 0.549$), eliminating undetected breakdowns at the cost of minor false alarms.
- $\tau = 0.50$: Optimal balance between precision ($75.0\%$) and recall ($54.5\%$), yielding peak $F_1 = 0.632$.
- $\tau = 0.70$: High-confidence dispatch. Precision reaches **100.0%** ($F_1 = 0.451$), preventing unnecessary maintenance crew callouts.

---

### 4.3 Project 3: NonlinearLab — Smooth Optimization & Newton Benchmarks

#### Problem Formulations
Evaluates smooth unconstrained optimization across two canonical geometries:
1. **Ill-Conditioned Quadratic Form:**
   $$f(x_1, x_2) = \frac{1}{2} (x_1^2 + 10 x_2^2), \quad Q = \begin{bmatrix} 1 & 0 \\ 0 & 10 \end{bmatrix}, \quad \kappa(Q) = 10.0$$
2. **Rosenbrock Banana Valley:**
   $$f(x_1, x_2) = (1 - x_1)^2 + 100(x_2 - x_1^2)^2$$
   Possesses a deep, parabolic, non-convex curved valley where gradient descent exhibits extreme zig-zagging.

#### Empirical Benchmark Comparison

```
========================================================================================
                     NONLINEARLAB: 7-OPTIMIZER BENCHMARK COMPARISON
========================================================================================
Benchmark / Optimizer          Iterations   Final Loss        Final Point       Status
----------------------------------------------------------------------------------------
QUADRATIC (kappa = 10.0)
- Fixed Step GD (lr=0.05)         86       1.854e-06     (0.0019, 0.0000)      Converged
- Exact Line Search GD            22       2.115e-06     (0.0020, 0.0001)      Converged
- Polyak Momentum (beta=0.8)      35       3.120e-06     (0.0024, 0.0002)      Converged
- Nesterov Accelerated GD         30       2.450e-06     (0.0021, 0.0001)      Converged
- Newton's Method                  1       0.000e+00     (0.0000, 0.0000)      EXACT 1-STEP

ROSENBROCK BANANA VALLEY
- Armijo Backtracking GD        2000       4.120e-02     (0.8034, 0.6432)      Max Iters
- Damped Newton's Method          24       3.210e-13     (1.0000, 1.0000)      Converged
- BFGS Quasi-Newton               38       8.410e-11     (1.0000, 1.0000)      Converged
========================================================================================
```

#### Verification of Newton's One-Step Theorem
When tested from arbitrary points $x_0 \in \{(-4, 3), (10, -5), (100, 200)\}$ on $f(x) = \frac{1}{2} x^T Q x - b^T x$, the Newton step yielded:
$$\|x_1 - x^*\|_2 < 10^{-15}$$
confirming that second-order Taylor expansion is exact for quadratics.

---

### 4.4 Project 4: SparseModel — High-Dimensional Parsimonious Feature Selection

#### Problem Motivation & Real-World Utility
In complex industrial facilities, hundreds of sensor telemetry streams are monitored, but only a small parsimonious subset actively drives mechanical failure or energy spikes. Ordinary Least Squares (OLS) overfits by assigning non-zero weights to all spurious noise features. Lasso ($L_1$) regularized regression enforces parameter sparsity.

#### Mathematical Formulation & Subgradient Mechanics
Composite nonsmooth objective:
$$\min_{w \in \mathbb{R}^d} J(w) = \frac{1}{2n} \|Xw - y\|_2^2 + \lambda \|w\|_1$$
The gradient of the MSE data term is smooth: $\nabla f_{\text{MSE}}(w) = \frac{1}{n} X^T (Xw - y)$.  
The $L_1$ penalty is nonsmooth at $w_j = 0$. The subgradient vector $g \in \partial J(w)$ is computed coordinate-wise:
$$g_j = \left[\frac{1}{n} X^T (Xw - y)\right]_j + \lambda v_j, \quad v_j = \begin{cases} +1, & w_j > 0 \\ -1, & w_j < 0 \\ 0, & w_j = 0 \end{cases}$$
No zero-snapping or heuristic thresholding is applied during optimization, strictly adhering to pure subgradient descent theory with step schedule $\alpha_k = \frac{\alpha_0}{\sqrt{k}}$.

#### Regularization Path & Sparse Feature Recovery
Generating the regularization path across $\lambda \in [0.001, 0.5]$ reveals the feature selection sequence:
- As $\lambda \to 0$: Recovers dense OLS coefficients, fitting background noise.
- At $\lambda^* = 0.05$: Spurious noise features ($w_4, w_5, w_6, w_7$) decay toward zero, cleanly isolating the true underlying physical drivers ($w_1 = 3.0, w_2 = -2.0, w_3 = 1.5$).
- As $\lambda \ge 0.30$: Excessive regularization suppresses true features, demonstrating underfitting.

---

### 4.5 Project 5: ResourceOpt — Constrained Production & Resource Allocation

#### Problem Formulation
A high-precision manufacturing plant optimizes daily production quantities $x = [x_1, x_2]^T$ of two precision parts to maximize profit (expressed as minimizing negative quadratic utility with production targets):
$$\min_{x_1, x_2} f(x_1, x_2) = \frac{1}{2} (x_1 - 4)^2 + \frac{1}{2} (x_2 - 5)^2$$
subject to:
1. Production Balance Equality: $x_1 + x_2 = 6$
2. Machine Capacity Inequality: $2 x_1 + x_2 \le 9$
3. Warehouse Storage Inequality: $x_1 + 2 x_2 \le 11$
4. Non-negativity: $x_1 \ge 0, x_2 \ge 0$

#### Systematic 5-Scenario KKT & LICQ Evaluation

```
========================================================================================
                    RESOURCEOPT: 5 CONSTRAINED SCENARIOS EVALUATION
========================================================================================
Scenario Tested            Point (x1, x2)   Active Constraints   LICQ   Stationarity Res
----------------------------------------------------------------------------------------
1. Feasible Interior       (2.50, 3.50)     Equality Only        HOLD   1.11e-16 (PASS)
2. Feasible Boundary       (3.00, 3.00)     Equality + Machine   HOLD   0.00e+00 (PASS)
3. Infeasible Candidate    (1.00, 6.00)     Storage Violated     N/A    Violated (+2.00)
4. Full Rank LICQ Hold     (3.00, 3.00)     Rank(J_A) = 2 == m   HOLD   det(G) = 1.00
5. Redundant LICQ Fail     (3.00, 3.00)     Rank(J_A) = 2 < 3    FAIL   det(G) = 0.00
========================================================================================
```

#### Analytical Insight on Constraint Qualifications
- **Scenario 4 (LICQ Holds):** Active constraints $x_1 + x_2 = 6$ and $2x_1 + x_2 = 9$ have gradients $\nabla g_1 = [1, 1]^T$ and $\nabla g_2 = [2, 1]^T$.  
  The Gram matrix $G = J_{\mathcal{A}} J_{\mathcal{A}}^T = \begin{bmatrix} 2 & 3 \\ 3 & 5 \end{bmatrix}$ has $\det(G) = 10 - 9 = 1.00 \ne 0$. The unique Lagrange multipliers are verified: $\lambda_1^* = 1.0, \mu_1^* = 0.0$.
- **Scenario 5 (LICQ Fails):** Adding a redundant constraint $2x_1 + 2x_2 \le 12$ causes row 3 of the active Jacobian to equal $2 \times \text{row } 1$.  
  The active Jacobian has rank 2 despite 3 active constraints; $\det(G) = 0.00$, proving that Lagrange multipliers lose uniqueness when constraint qualifications fail.

---

## 5. File-by-File Technical Directory for Domain Projects

### 5.1 Project 1: `EnergyOpt/`
- [synthetic_energy.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/EnergyOpt/data/synthetic_energy.py): Generates 500 hours of synthetic power grid telemetry combining sinusoidal diurnal cycles, parabolic temperature responses, and autoregressive noise.
- [demand_regression.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/EnergyOpt/problems/demand_regression.py): Wraps feature standardization, train/validation partitioning, MSE evaluation, and regression optimization adapters.
- [generation_dispatch.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/EnergyOpt/problems/generation_dispatch.py): Sets up the 3-generator quadratic cost dispatch problem, calls `lagrange_multiplier_solver`, and validates KKT conditions and LICQ.
- [plots.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/EnergyOpt/visualization/plots.py): Generates publication-ready figures: demand vs. actual scatter overlays, multi-optimizer MSE loss trajectories, and generator dispatch share bars.
- [run_experiments.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/EnergyOpt/experiments/run_experiments.py): End-to-end CLI runner executing all 6 optimizers, exporting CSV trajectories, running dispatch, and computing financial forecast sensitivity.
- [main.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/EnergyOpt/main.py): Project entry point supporting `--all`, `--train`, `--dispatch`, and `--test` flags.

### 5.2 Project 2: `PredictMaint/`
- [synthetic_maintenance.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/PredictMaint/data/synthetic_maintenance.py): Simulates 800 industrial machinery records with mechanical failure labels determined by non-linear thermal-vibrational interactions.
- [failure_classification.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/PredictMaint/problems/failure_classification.py): Encapsulates logistic regression modeling, BCE objective calculation, probability calibration, and threshold sweeping.
- [plots.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/PredictMaint/visualization/plots.py): Renders 2D decision boundary contours over Vibration vs. Temperature telemetry, confusion matrix heatmaps, and threshold sensitivity curves.
- [run_experiments.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/PredictMaint/experiments/run_experiments.py): Executes the 4-optimizer benchmark, computes classification metrics, and logs the confusion matrix.
- [main.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/PredictMaint/main.py): CLI entry point for standalone execution.

### 5.3 Project 3: `NonlinearLab/`
- [quadratic_benchmark.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/NonlinearLab/problems/quadratic_benchmark.py): Configures ill-conditioned quadratic surfaces ($\kappa=10$) and executes the Newton One-Step proof test.
- [rosenbrock_calibration.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/NonlinearLab/problems/rosenbrock_calibration.py): Sets up the Rosenbrock benchmark and verifies gradient/Hessian analytical calculations.
- [plots.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/NonlinearLab/visualization/plots.py): Generates overlaid optimizer trajectory paths on filled contour maps and semi-logarithmic loss vs. iteration curves.
- [run_experiments.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/NonlinearLab/experiments/run_experiments.py): Executes all 7 optimizers across both benchmark surfaces and logs metrics.
- [main.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/NonlinearLab/main.py): Standalone CLI runner.

### 5.4 Project 4: `SparseModel/`
- [synthetic_sparse.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/SparseModel/data/synthetic_sparse.py): Synthesizes high-dimensional datasets ($d=8$) with known ground-truth sparse coefficients ($w^* = [3.0, -2.0, 1.5, 0, 0, 0, 0, 0]^T$).
- [lasso_formulation.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/SparseModel/problems/lasso_formulation.py): Formulates Lasso composite loss, nonsmooth subdifferential, and diminishing step size subgradient schedules.
- [plots.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/SparseModel/visualization/plots.py): Plots subgradient norm convergence curves, full regularization paths, and OLS vs. Lasso parameter recovery bar charts.
- [run_experiments.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/SparseModel/experiments/run_experiments.py): Runs subgradient optimization across Euclidean norm and Lasso loss, performing a full $\lambda$ parameter sweep.
- [main.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/SparseModel/main.py): Standalone CLI runner.

### 5.5 Project 5: `ResourceOpt/`
- [production_allocation.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/ResourceOpt/problems/production_allocation.py): Implements the 2-product manufacturing optimization problem with linear equalities and inequalities, computing KKT residuals and LICQ Gram determinants.
- [plots.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/ResourceOpt/visualization/plots.py): Renders the feasible polytope, objective contours, constraint hyperplanes, and marks the 5 test scenarios.
- [run_experiments.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/ResourceOpt/experiments/run_experiments.py): Evaluates all 5 scenarios, logs KKT stationarity, feasibility, slackness residuals, and outputs summary tables.
- [main.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/ResourceOpt/main.py): Standalone CLI runner.

### 5.6 Master Orchestration: `run_all.py`
- [run_all.py](file:///d:/Downloads/Acamedics%20-%20%20SEM%205/Optim%20DS/Optimizaztion%20Project/Optimization-Laboratory/run_all.py): Master CLI script that traverses all 5 domain projects sequentially, runs experiments, verifies unit tests, and compiles runtime statistics into a consolidated report.

---

## 6. Verification & Reproduction Guide

### 6.1 Executing the Automated Unit Test Suite
The laboratory includes 77 automated unit tests validating linear algebra, optimizer convergence, analytical gradients, KKT residuals, and constraint qualifications. Run the complete test suite:
```bash
cd Optimization-Laboratory
python -m unittest discover -s shared_optimization/tests -p "test_*.py"
```
**Current Output:**
```
Ran 77 tests in 0.624s
OK
```

### 6.2 Running the Complete Benchmark Suite
To execute all 5 projects end-to-end, regenerate all CSV logs, and produce publication-quality plots:
```bash
python run_all.py --all
```
Total pipeline execution completes in approximately **61 seconds**.

### 6.3 Standalone Project Execution
Each domain project can also be run independently:
```bash
# Project 1: EnergyOpt
cd EnergyOpt && python main.py --all

# Project 2: PredictMaint
cd PredictMaint && python main.py --all

# Project 3: NonlinearLab
cd NonlinearLab && python main.py --all

# Project 4: SparseModel
cd SparseModel && python main.py --all

# Project 5: ResourceOpt
cd ResourceOpt && python main.py --all
```

---

## 7. Conclusion & Key Academic Insights

1. **First-Order vs. Second-Order Curvature:**
   Fixed-step gradient descent suffers severe performance degradation on ill-conditioned objectives ($\kappa = 10.0$ required 86 iterations; Rosenbrock failed to converge within 2000 steps). Newton's method converged in **1 step** on quadratics and **24 iterations** on Rosenbrock, proving that second-order curvature information completely eliminates valley oscillations.
2. **Adaptive Optimizers in Practice:**
   While Adam and AdaGrad excelled on classification tasks with varying feature dimensions (`PredictMaint`), Mini-batch SGD demonstrated superior generalization on continuous linear regression (`EnergyOpt`), achieving a validation RMSE of $16.48\text{ MW}$ ($R^2 = 0.7156$) compared to Adam's $35.10\text{ MW}$.
3. **Subgradient Calculus for Sparsity:**
   Subgradient descent with diminishing step size $\alpha_k = \frac{\alpha_0}{\sqrt{k}}$ proved effective for nonsmooth composite optimization, cleanly uncovering true sparse parameters without heuristic zero-snapping.
4. **Constrained Optimization & Constraint Qualifications:**
   The Karush-Kuhn-Tucker (KKT) conditions provide an exact optimality certificate for constrained engineering systems. However, as demonstrated in `ResourceOpt`, the Lagrange multipliers are unique and physically meaningful only when the **Linear Independence Constraint Qualification (LICQ)** holds ($\det(J_{\mathcal{A}} J_{\mathcal{A}}^T) > 0$). When redundant constraints are introduced, LICQ fails ($\det(G) = 0$), illustrating why modern solvers must check constraint qualifications.
