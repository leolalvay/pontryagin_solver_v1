import numpy as np
from typing import List

class PABundle:
    """
    Piecewise-affine surrogate for the Hamiltonian.

    The bundle stores a set of candidate controls a_i.  For a fixed state x,
    costate p and time t, the surrogate Hamiltonian \bar{H}(p,x,t) is given by

        \bar{H}(p,x,t) = \min_{i} \{ p \cdot f(x,a_i,t) + \ell(x,a_i,t) \}.

    This class only manages the collection of control vectors; the actual
    evaluation of f and l is delegated to the OCPProblem instance.
    """

    def __init__(self):
        self.controls: List[np.ndarray] = []

    def num_planes(self) -> int:
        """Return the number of stored control candidates."""
        return len(self.controls)

    def add_control(self, u: np.ndarray, tol: float = 1e-8) -> None:
        """
        Add a new control candidate to the bundle if it is not already present
        within a tolerance.

        Parameters
        ----------
        u : np.ndarray
            Control vector to add.
        tol : float
            L2 distance tolerance to determine uniqueness.
        """
        u = np.asarray(u, dtype=float)
        for v in self.controls:
            if np.linalg.norm(u - v) < tol:
                return
        self.controls.append(u)

    def evaluate(
        self,
        problem,
        p: np.ndarray,
        x: np.ndarray,
        t: float,
        dt: float | None = None,
        *,
        restricted: bool = True,
        fallback_unrestricted: bool = True,
    ) -> tuple:
        """
        Evaluate the surrogate Hamiltonian \bar{H}(p,x,t).

        Parameters
        ----------
        problem : OCPProblem
            The optimal control problem providing dynamics and cost.
        p : np.ndarray
            Costate vector.
        x : np.ndarray
            State vector.
        t : float
            Time instant.

        Returns
        -------
        (float, int)
            The value of \bar{H} and the index of the active plane.
        """
        if not self.controls:
            raise RuntimeError("PABundle has no control candidates to evaluate.")
        best_val = np.inf
        best_idx = -1
        tried_any = False
        for i, u in enumerate(self.controls):
            if restricted and not problem.local_control_feasible(x, u, t, restricted=True, dt=dt):
                continue
            tried_any = True
            val = float(np.dot(p, problem.f(x, u, t)) + problem.l(x, u, t))
            if val < best_val:
                best_val = val
                best_idx = i
        if fallback_unrestricted and not tried_any:
            for i, u in enumerate(self.controls):
                val = float(np.dot(p, problem.f(x, u, t)) + problem.l(x, u, t))
                if val < best_val:
                    best_val = val
                    best_idx = i
        if best_idx < 0:
            raise RuntimeError("PABundle has no locally feasible control candidates to evaluate.")
        return best_val, best_idx
