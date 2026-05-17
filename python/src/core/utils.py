import numpy as np

def fd_gradient(func, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Approximate the gradient of a scalar function at a point x using central finite differences.

    Parameters
    ----------
    func : Callable[[np.ndarray], float]
        Function that takes a vector and returns a scalar.
    x : np.ndarray
        Point at which to compute the gradient.
    eps : float
        Perturbation size for finite differences.

    Returns
    -------
    np.ndarray
        Approximation of the gradient of func at x.
    """
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for i in range(x.size):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        f_plus = func(x_plus)
        f_minus = func(x_minus)
        grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad