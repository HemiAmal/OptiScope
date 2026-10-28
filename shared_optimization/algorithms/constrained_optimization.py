"""
Constrained optimization routines: Lagrange multiplier solver for equality constraints,
KKT optimality conditions evaluator, and LICQ constraint qualification checker.
"""

from typing import Dict, List, Sequence, Tuple, Union
from shared_optimization.core.vectors import validate_vector, vec_add, scalar_mul, dot_product, norm_l2, Number
from shared_optimization.core.matrices import validate_matrix, matrix_shape, transpose, mat_vec_mul, Matrix
from shared_optimization.core.linear_systems import solve_linear_system, matrix_determinant
from shared_optimization.problems.constrained import ConstrainedProblem


def lagrange_multiplier_solver(
    Q: Sequence[Sequence[Number]],
    c: Sequence[Number],
    A: Sequence[Sequence[Number]],
    b: Sequence[Number],
) -> Dict[str, Union[List[float], float]]:
    """
    Solves equality-constrained quadratic minimization:
        min 0.5 * x^T * Q * x + c^T * x   s.t.   A * x = b

    Solves the block saddle-point KKT system [[Q, A^T], [A, 0]] [x, lambda]^T = [-c, b]^T.
    """
    Q_val = validate_matrix(Q, "Q")
    A_val = validate_matrix(A, "A")
    c_val = validate_vector(c, "c")
    b_val = validate_vector(b, "b")

    n, n_cols = matrix_shape(Q_val)
    m, m_cols = matrix_shape(A_val)

    if n != n_cols:
        raise ValueError(f"Matrix Q must be square, got ({n}, {n_cols}).")
    if m_cols != n:
        raise ValueError(f"Constraint matrix A columns {m_cols} != variables dimension {n}.")
    if len(c_val) != n:
        raise ValueError(f"Vector c length {len(c_val)} != variables dimension {n}.")
    if len(b_val) != m:
        raise ValueError(f"Vector b length {len(b_val)} != constraints count {m}.")

    # Build (n + m) x (n + m) KKT block matrix
    # [ Q   A^T ]
    # [ A    0  ]
    At = transpose(A_val)
    KKT_size = n + m
    KKT_matrix: Matrix = [[0.0] * KKT_size for _ in range(KKT_size)]

    # Top-left: Q (n x n)
    for i in range(n):
        for j in range(n):
            KKT_matrix[i][j] = Q_val[i][j]

    # Top-right: A^T (n x m)
    for i in range(n):
        for j in range(m):
            KKT_matrix[i][n + j] = At[i][j]

    # Bottom-left: A (m x n)
    for i in range(m):
        for j in range(n):
            KKT_matrix[n + i][j] = A_val[i][j]

    # Bottom-right: 0 (m x m) -> already 0.0

    # RHS vector: [-c, b]
    rhs = [-c_val[i] for i in range(n)] + list(b_val)

    # Solve KKT system via Gaussian Elimination
    sol = solve_linear_system(KKT_matrix, rhs)

    x_opt = sol[:n]
    lambda_opt = sol[n:]

    # Compute objective value: 0.5 * x^T * Q * x + c^T * x
    Qx = mat_vec_mul(Q_val, x_opt)
    f_val = 0.5 * dot_product(x_opt, Qx) + dot_product(c_val, x_opt)

    # Compute residuals
    Ax = mat_vec_mul(A_val, x_opt)
    primal_residual = norm_l2([Ax[i] - b_val[i] for i in range(m)])

    At_lambda = mat_vec_mul(At, lambda_opt)
    stationarity_vec = [Qx[i] + c_val[i] + At_lambda[i] for i in range(n)]
    stationarity_residual = norm_l2(stationarity_vec)

    return {
        "x": x_opt,
        "lambda": lambda_opt,
        "objective_value": f_val,
        "primal_residual": primal_residual,
        "stationarity_residual": stationarity_residual,
    }


def kkt_conditions_solver(
    problem: ConstrainedProblem,
    x: Sequence[Number],
    lambda_eq: Sequence[Number] = (),
    mu_ineq: Sequence[Number] = (),
    tol: float = 1e-4,
) -> Dict[str, Union[bool, float, Dict[str, bool]]]:
    """
    Evaluates Karush-Kuhn-Tucker (KKT) first-order optimality conditions:
    1. Stationarity: || grad f(x) + sum lambda_i * grad h_i(x) + sum mu_j * grad g_j(x) || <= tol
    2. Primal Feasibility: |h_i(x)| <= tol and g_j(x) <= tol
    3. Dual Feasibility: mu_j >= -tol
    4. Complementary Slackness: |mu_j * g_j(x)| <= tol
    """
    x_val = validate_vector(x, "x")
    n = len(x_val)
    m = len(problem.eq_constraints)
    p = len(problem.ineq_constraints)

    lam = list(lambda_eq) if lambda_eq else [0.0] * m
    mu = list(mu_ineq) if mu_ineq else [0.0] * p

    if len(lam) != m:
        raise ValueError(f"Expected {m} equality multipliers lambda, got {len(lam)}.")
    if len(mu) != p:
        raise ValueError(f"Expected {p} inequality multipliers mu, got {len(mu)}.")

    # 1. Stationarity Check
    # grad_L = nabla f(x) + sum lambda_i * nabla h_i + sum mu_j * nabla g_j
    grad_L = list(problem.grad_f(x_val))
    for i in range(m):
        grad_h_i = problem.eq_gradients[i](x_val)
        for d in range(n):
            grad_L[d] += lam[i] * grad_h_i[d]
    for j in range(p):
        grad_g_j = problem.ineq_gradients[j](x_val)
        for d in range(n):
            grad_L[d] += mu[j] * grad_g_j[d]

    stationarity_residual = norm_l2(grad_L)
    stationarity_passed = stationarity_residual <= tol

    # 2. Primal Feasibility Check
    primal_eq_max = 0.0
    for i in range(m):
        h_val = abs(problem.eq_constraints[i](x_val))
        if h_val > primal_eq_max:
            primal_eq_max = h_val
    primal_eq_passed = primal_eq_max <= tol

    primal_ineq_max_violation = 0.0
    for j in range(p):
        g_val = problem.ineq_constraints[j](x_val)
        if g_val > primal_ineq_max_violation:
            primal_ineq_max_violation = g_val
    primal_ineq_passed = primal_ineq_max_violation <= tol

    primal_feasibility_passed = primal_eq_passed and primal_ineq_passed

    # 3. Dual Feasibility Check: mu_j >= 0
    dual_min = min(mu) if p > 0 else 0.0
    dual_feasibility_passed = dual_min >= -tol

    # 4. Complementary Slackness Check: |mu_j * g_j(x)| == 0
    comp_slack_max = 0.0
    for j in range(p):
        slack = abs(mu[j] * problem.ineq_constraints[j](x_val))
        if slack > comp_slack_max:
            comp_slack_max = slack
    comp_slack_passed = comp_slack_max <= tol

    all_kkt_passed = (
        stationarity_passed
        and primal_feasibility_passed
        and dual_feasibility_passed
        and comp_slack_passed
    )

    return {
        "kkt_satisfied": all_kkt_passed,
        "stationarity_passed": stationarity_passed,
        "primal_feasibility_passed": primal_feasibility_passed,
        "dual_feasibility_passed": dual_feasibility_passed,
        "complementary_slackness_passed": comp_slack_passed,
        "stationarity_residual": stationarity_residual,
        "primal_eq_max_residual": primal_eq_max,
        "primal_ineq_max_violation": primal_ineq_max_violation,
        "complementary_slackness_max": comp_slack_max,
    }


def check_licq(
    problem: ConstrainedProblem,
    x: Sequence[Number],
    active_tol: float = 1e-4,
    tol_indep: float = 1e-8,
) -> Dict[str, Union[bool, int, List[int], float]]:
    """
    Evaluates whether the gradients of active constraints are linearly independent (LICQ).
    """
    x_val = validate_vector(x, "x")
    n = len(x_val)
    m = len(problem.eq_constraints)
    p = len(problem.ineq_constraints)

    active_gradients: List[List[float]] = []

    # All equality constraints are active by definition
    for i in range(m):
        grad_h = problem.eq_gradients[i](x_val)
        active_gradients.append(grad_h)

    # Identify active inequality constraints: g_j(x) approx 0
    active_ineq_indices: List[int] = []
    for j in range(p):
        g_val = problem.ineq_constraints[j](x_val)
        if abs(g_val) <= active_tol:
            active_ineq_indices.append(j)
            grad_g = problem.ineq_gradients[j](x_val)
            active_gradients.append(grad_g)

    num_active = len(active_gradients)

    # If no active constraints, LICQ trivially holds
    if num_active == 0:
        return {
            "licq_satisfied": True,
            "num_active_constraints": 0,
            "active_inequality_indices": [],
            "gram_determinant": 1.0,
        }

    # If active constraints exceed dimension n, they cannot be linearly independent in R^n
    if num_active > n:
        return {
            "licq_satisfied": False,
            "num_active_constraints": num_active,
            "active_inequality_indices": active_ineq_indices,
            "gram_determinant": 0.0,
        }

    # Check linear independence via Gram Matrix G = M * M^T (size num_active x num_active)
    # Vectors are linearly independent iff det(G) > 0
    G: Matrix = [[0.0] * num_active for _ in range(num_active)]
    for i in range(num_active):
        for j in range(num_active):
            G[i][j] = dot_product(active_gradients[i], active_gradients[j])

    det_G = matrix_determinant(G)
    is_independent = abs(det_G) > tol_indep

    return {
        "licq_satisfied": is_independent,
        "num_active_constraints": num_active,
        "active_inequality_indices": active_ineq_indices,
        "gram_determinant": det_G,
    }
