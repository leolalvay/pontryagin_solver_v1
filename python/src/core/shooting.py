import numpy as np

from .integrators import assemble_residual, assemble_jacobian, unpack_unknowns, pack_unknowns

def shooting_residual(problem, t_nodes: np.ndarray, z: np.ndarray, bundle, delta: float, use_explicit_gradients: bool = False) -> np.ndarray:
    """
    Compute the shooting residual for the current unknown vector.

    Parameters
    ----------
    problem : OCPProblem
        The optimal control problem.
    t_nodes : np.ndarray
        Array of time nodes of length N+1.
    z : np.ndarray
        Flattened unknown vector containing x_1,...,x_N and p_0,...,p_N.
    bundle : PABundle
        Bundle of controls.
    delta : float
        Smoothing parameter.

    Returns
    -------
    np.ndarray
        Residual vector.
    """
    # reconstruct X and P from z
    x0 = problem.x0
    X, P = unpack_unknowns(z, x0)
    return assemble_residual(problem, t_nodes, X, P, bundle, delta, use_explicit_gradients=use_explicit_gradients)

def shooting_jacobian(problem, t_nodes: np.ndarray, z: np.ndarray, bundle, delta: float, use_explicit_gradients: bool = False) -> np.ndarray:
    """
    Compute the Jacobian of the shooting residual with respect to the unknown vector.
    """
    x0 = problem.x0
    X, P = unpack_unknowns(z, x0)
    return assemble_jacobian(problem, t_nodes, X, P, bundle, delta, use_explicit_gradients=use_explicit_gradients)