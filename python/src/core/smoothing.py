import numpy as np
from typing import Tuple, Optional

def eval_H_smooth(
    problem,
    bundle,
    p: np.ndarray,
    x: np.ndarray,
    t: float,
    delta: float,
    dt: Optional[float] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate the smoothed surrogate Hamiltonian H_δ(p,x,t) and its gradients.

    The piecewise-affine surrogate \bar{H}(p,x,t) is a minimum over planes

        \bar{H}(p,x,t) = \min_{i} \{ p \cdot f(x,a_i,t) + l(x,a_i,t) \},

    where the minimisation is over controls stored in `bundle`.  To enable
    differentiability, we compute a smooth approximation using a log-sum-exp
    formulation:

        H_δ(p,x,t) = -δ log sum_i exp(-(g_i(p,x,t))/δ),

    where g_i = p ⋅ f(x,a_i,t) + l(x,a_i,t).  This yields H_δ ≤ \bar{H} and
    H_δ → \bar{H} as δ ↓ 0.  The gradients are

        ∂H_δ/∂p = \sum_i w_i f_i,
        ∂H_δ/∂x = \sum_i w_i (∂(g_i)/∂x),

    where w_i are the soft-min weights.  The derivative ∂(g_i)/∂x is computed
    numerically by central finite differences on f and l.

    Parameters
    ----------
    problem : OCPProblem
        Problem providing dynamics and costs.
    bundle : PABundle
        Bundle of control planes.
    p : np.ndarray
        Costate vector.
    x : np.ndarray
        State vector.
    t : float
        Time instant.
    delta : float
        Smoothing parameter (δ > 0).

    Returns
    -------
    (float, np.ndarray, np.ndarray)
        H_δ value, gradient with respect to p (shape (n,)), and gradient with respect
        to x (shape (n,)).
    """
    if getattr(problem, "hamiltonian_smooth_fn", None) is not None:
        return problem.hamiltonian_smooth(x, p, t, delta)

    # evaluate planes
    feasible_controls = [np.asarray(u, dtype=float) for u in bundle.controls]
    m = len(feasible_controls)
    if m == 0:
        raise RuntimeError("PABundle is empty; cannot compute smooth Hamiltonian.")
    n = p.size
    g_vals = np.empty(m)
    f_vals = np.empty((m, n))
    # evaluate g_i = p·f + l for each control
    for i, u in enumerate(feasible_controls):
        f_i = problem.f(x, u, t)
        f_vals[i, :] = f_i
        g_vals[i] = float(np.dot(p, f_i) + problem.l(x, u, t))
    # stable log-sum-exp
    g_min = np.min(g_vals)
    # weights proportional to exp(-(g_i - g_min)/delta)
    exps = np.exp(-(g_vals - g_min) / max(delta, 1e-12))
    # compute soft-min value
    sum_exps = np.sum(exps)
    # H_delta = g_min - delta * log(sum_exps)
    H_delta = g_min - delta * np.log(sum_exps + 1e-300)
    # compute softmin weights normalized
    weights = exps / sum_exps
    # gradient w.r.t. p
    grad_p = np.sum(weights[:, None] * f_vals, axis=0)
    # gradient w.r.t. x
    grad_x = np.zeros_like(x)
    # approximate partial derivatives of g_i w.r.t x by central finite differences
    # note: g_i = p·f(x,a_i) + l(x,a_i)
    eps = 1e-6
    #loop over the components of x to compute x_k^+ and x_k^-
    for dim in range(x.size):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[dim] += eps
        x_minus[dim] -= eps
        # compute g_i at x_plus and x_minus for each plane
        g_plus = np.empty(m)
        g_minus = np.empty(m)
        #loop over the planes (controls) to compute the sum over the planes
        for i, u in enumerate(feasible_controls):
            f_plus = problem.f(x_plus, u, t)
            f_minus = problem.f(x_minus, u, t)
            l_plus = problem.l(x_plus, u, t)
            l_minus = problem.l(x_minus, u, t)
            g_plus[i] = np.dot(p, f_plus) + l_plus
            g_minus[i] = np.dot(p, f_minus) + l_minus
        # finite difference derivative of g_i
        dg_dx = (g_plus - g_minus) / (2.0 * eps)
        # grad_x = sum weights * dg_dx
        grad_x[dim] = float(np.sum(weights * dg_dx))
    return H_delta, grad_p, grad_x
