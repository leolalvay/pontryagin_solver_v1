import numpy as np

def enforce_state_bounds(x: np.ndarray, x_min: np.ndarray, x_max: np.ndarray) -> np.ndarray:
    """
    Clip the state vector x into [x_min, x_max] componentwise.

    Parameters
    ----------
    x : np.ndarray
        State vector to be clipped.
    x_min : np.ndarray
        Lower bounds on each state dimension.
    x_max : np.ndarray
        Upper bounds on each state dimension.

    Returns
    -------
    np.ndarray
        Clipped state vector.
    """
    return np.minimum(np.maximum(x, x_min), x_max)

def project_control(u: np.ndarray, u_min: np.ndarray, u_max: np.ndarray) -> np.ndarray:
    """
    Clip the control vector u into [u_min, u_max] componentwise.
    """
    return np.minimum(np.maximum(u, u_min), u_max)