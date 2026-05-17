import numpy as np
from itertools import product
from typing import List, Tuple, Optional
from typing import Optional
from scipy.optimize import minimize_scalar

def compute_H(
    problem,
    p: np.ndarray,
    x: np.ndarray,
    t: float,
    candidate_controls: List[np.ndarray],
    restricted: bool = False,
    use_oracle: bool = False,
    dt: Optional[float] = None,
) -> Tuple[float, np.ndarray]:
    """
    Compute the true Hamiltonian H(p,x,t) or the restricted Hamiltonian H_K(p,x,t).

    The Hamiltonian is defined by

        H(p,x,t) = \min_{u ∈ A} \{ p · f(x,u,t) + ℓ(x,u,t) \},

    where A is the admissible control set (typically a box).  For the restricted
    version H_K, the minimisation is further restricted to controls for which the
    resulting velocity lies in the tangent cone of the state constraint set K.

    Parameters
    ----------
    problem : OCPProblem
        Problem providing dynamics and costs and constraint information.
    p : np.ndarray
        Costate vector.
    x : np.ndarray
        State vector.
    t : float
        Time instant.
    candidate_controls : list of np.ndarray
        A list of control vectors to consider in addition to the extreme
        combinations of the control bounds.  Typically, this list includes
        controls currently stored in a PABundle.
    restricted : bool
        If True, only controls that maintain viability (i.e. f(x,u,t) in
        tangent cone of K at x) are considered.

    Returns
    -------
    (float, np.ndarray)
        The minimal Hamiltonian value and the corresponding control u*.
    """
    # gather control candidates: extremes + provided
    candidates: List[np.ndarray] = []
    # extremes based on bounds
    bounds = problem.control_bounds_tuple()
# --- Optional oracle fast-path: if u*(x,p,t) exists, return immediately ---
    if use_oracle and hasattr(problem, "u_star"):
        u_oracle, ok = problem.u_star(x, p, t, restricted=restricted, dt=dt)
        if (u_oracle is not None) and (not restricted or ok):
            u_oracle = np.asarray(u_oracle, dtype=float)
            if bounds is not None:
                u_oracle = problem.project_control(u_oracle)
            candidates.append(u_oracle)

    if bounds is not None:
        u_min, u_max = bounds
        m = u_min.size
        # generate all 2^m combinations of min and max for each dimension
        for combo in product([0, 1], repeat=m):
            u = np.where(np.array(combo) == 0, u_min, u_max)
            candidates.append(u)
        # --- minimal fix: enrich candidate set for scalar control (m=1) ---
        #if m == 1:
            #u_grid = np.linspace(float(u_min[0]), float(u_max[0]), 203)
            #for a in u_grid:
                #candidates.append(np.array([a], dtype=float))
# --- Replace dense grid (m=1) by bounded Brent
        if (m == 1): 
            a_lo = float(u_min[0])
            a_hi = float(u_max[0])

            def obj(a: float) -> float:
                u = np.array([a], dtype=float)
                # keep consistent with your pipeline
                u = problem.project_control(u)
                # NEW: viability when restricted=True
                if not problem.local_control_feasible(x, u, t, restricted=restricted, dt=dt):
                    return 1.0e30
                return float(np.dot(p, problem.f(x, u, t)) + problem.l(x, u, t))

            res = minimize_scalar(
                obj,
                bounds=(a_lo, a_hi),
                method="bounded",
                options={"xatol": 1e-6, "maxiter": 80},
            )
            if res.success and np.isfinite(res.fun):
                candidates.append(np.array([float(res.x)], dtype=float))  

    # include provided controls
    for u in candidate_controls:
        # ensure u is within bounds (project if necessary)
        if bounds is not None:
            u = problem.project_control(u)
        candidates.append(u)
    # remove duplicates (within small tolerance)
    unique = []
    for u in candidates:
        is_new = True
        for v in unique:
            if np.linalg.norm(u - v) < 1e-10:
                is_new = False
                break
        if is_new:
            unique.append(u)
    candidates = unique
    # Candidate evaluation loop
    best_val = np.inf
    best_control = None
    for u in candidates:
        # check viability
        if not problem.local_control_feasible(x, u, t, restricted=restricted, dt=dt):
            continue
        val = float(np.dot(p, problem.f(x, u, t)) + problem.l(x, u, t))
        if val < best_val:
            best_val = val
            best_control = u
    if best_control is None:
        if restricted:
            return float("inf"), None
        for u in candidates:
            val = float(np.dot(p, problem.f(x, u, t)) + problem.l(x, u, t))
            if val < best_val:
                best_val = val
                best_control = u
    if best_control is None:
        return float("inf"), None
    return best_val, best_control
