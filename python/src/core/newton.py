import numpy as np
import time
from .integrators import unpack_unknowns, pack_unknowns
from .shooting import shooting_residual, shooting_jacobian
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from scipy.sparse import issparse
from scipy.optimize import least_squares


def _trial_trajectory(problem, t_nodes, z_trial, feasibility_tol):
    X_trial, P_trial = unpack_unknowns(z_trial, problem.x0)
    feasible = problem.trajectory_feasible(X_trial, t_nodes, tol=feasibility_tol)
    return X_trial, P_trial, feasible


def solve_tpbvp(problem, t_nodes: np.ndarray, bundle, delta: float,
                 X_init: np.ndarray = None, P_init: np.ndarray = None,
                 tol: float = 1e-10, max_iter: int = 50,  use_explicit_hamiltonian_gradients: bool = False,
                 fallback_solver: str | None = "least_squares") -> tuple:
    """
    Solve the two-point boundary value problem by damped Newton method.

    Parameters
    ----------
    problem : OCPProblem
        The optimal control problem to solve.
    t_nodes : np.ndarray
        Discretised time mesh (length N+1).
    bundle : PABundle
        Bundle of control candidates for Hamiltonian smoothing.
    delta : float
        Smoothing parameter for the Hamiltonian.
    X_init : np.ndarray, optional
        Initial guess for state trajectory (shape (N+1, n)).  If None,
        a linear interpolation between x0 and zeros is used.
    P_init : np.ndarray, optional
        Initial guess for costate trajectory (shape (N+1, n)).  If None,
        the costate is initialised to zeros.
    tol : float
        Tolerance for residual norm to declare convergence.
    max_iter : int
        Maximum number of Newton iterations.

    Returns
    -------
    (X, P, info)
        Solved state and costate trajectories and an info dict containing
        convergence diagnostics.
    """
    N_plus_1 = t_nodes.size
    n = problem.x0.size
    # initial guess
    if X_init is None:
        # linearly interpolate from x0 to zeros (rough guess)
        X_init = np.zeros((N_plus_1, n))
        X_init[0] = problem.x0
        for i in range(1, N_plus_1):
            alpha = i / (N_plus_1 - 1)
            X_init[i] = (1 - alpha) * problem.x0
    if P_init is None:
        P_init = np.zeros((N_plus_1, n))
    # pack unknowns z: x1..xN, p0..pN
    z = pack_unknowns(X_init, P_init)
    solver_phase = "newton"
    fallback_used = False

    def residual(vec):
        return shooting_residual(
            problem,
            t_nodes,
            vec,
            bundle,
            delta,
            use_explicit_gradients=use_explicit_hamiltonian_gradients,
        )

    def jacobian(vec):
        J = shooting_jacobian(
            problem,
            t_nodes,
            vec,
            bundle,
            delta,
            use_explicit_gradients=use_explicit_hamiltonian_gradients,
        )
        return J.toarray() if issparse(J) else J

    # Newton iteration
    feasibility_tol = 1e-10
    n_feasibility_rejections = 0
    n_projection_fallbacks = 0
    for it in range(max_iter):
        F = residual(z)
        normF = np.linalg.norm(F, ord=np.inf)
        if normF < tol:
            # converged
            break
        J = shooting_jacobian(problem, t_nodes, z, bundle, delta, use_explicit_gradients=use_explicit_hamiltonian_gradients)
        # solve J * dz = -F (sparse LU on a CSC view of J)
        try:
            lu = splu(csc_matrix(J), permc_spec="COLAMD")
            dz = lu.solve(-F)
        except Exception:
            J_dense = J.toarray() if issparse(J) else J
            try:
                dz = np.linalg.solve(J_dense, -F)
            except np.linalg.LinAlgError:
                dz, *_ = np.linalg.lstsq(J_dense, -F, rcond=None)

        X_curr, _ = unpack_unknowns(z, problem.x0)
        dX, _ = unpack_unknowns(dz, np.zeros_like(problem.x0))
        lam = min(1.0, float(problem.fraction_to_boundary_step(X_curr, dX, t_nodes, safety=0.99, tol=feasibility_tol)))
        lam = max(lam, 1e-8)

        accepted = False
        last_trial = None
        while lam > 1e-8:
            z_trial = z + lam * dz
            X_trial, _, feasible = _trial_trajectory(problem, t_nodes, z_trial, feasibility_tol)
            last_trial = z_trial
            if not feasible:
                n_feasibility_rejections += 1
                lam *= 0.5
                continue
            F_new = residual(z_trial)
            normF_new = np.linalg.norm(F_new, ord=np.inf)
            if normF_new <= (1 - 1e-4 * lam) * normF:
                z = z_trial
                accepted = True
                break
            lam *= 0.5

        if not accepted and last_trial is not None and problem.project_state_fn is not None:
            X_trial, P_trial, _ = _trial_trajectory(problem, t_nodes, last_trial, feasibility_tol)
            X_proj = problem.project_trajectory(X_trial, t_nodes, tol=feasibility_tol)
            if problem.trajectory_feasible(X_proj, t_nodes, tol=feasibility_tol):
                z_proj = pack_unknowns(X_proj, P_trial)
                F_proj = residual(z_proj)
                normF_proj = np.linalg.norm(F_proj, ord=np.inf)
                if normF_proj < normF:
                    z = z_proj
                    accepted = True
                    n_projection_fallbacks += 1

        if not accepted:
            break

    final_residual = residual(z)
    final_norm = np.linalg.norm(final_residual, ord=np.inf)

    # If the damped Newton loop stalls, use a robust nonlinear least-squares fallback.
    if (final_norm >= tol) and (fallback_solver == "least_squares"):
        lsq = least_squares(
            residual,
            z,
            jac=jacobian,
            method="trf",
            xtol=tol,
            ftol=tol,
            gtol=tol,
            max_nfev=max(200, 10 * (max_iter + 1)),
        )
        z = lsq.x
        X_lsq, P_lsq = unpack_unknowns(z, problem.x0)
        if (not problem.trajectory_feasible(X_lsq, t_nodes, tol=feasibility_tol)) and (problem.project_state_fn is not None):
            X_proj = problem.project_trajectory(X_lsq, t_nodes, tol=feasibility_tol)
            if problem.trajectory_feasible(X_proj, t_nodes, tol=feasibility_tol):
                z = pack_unknowns(X_proj, P_lsq)
                n_projection_fallbacks += 1
        final_residual = residual(z)
        final_norm = np.linalg.norm(final_residual, ord=np.inf)
        solver_phase = "least_squares_fallback"
        fallback_used = True

    # reconstruct solution
    X_sol, P_sol = unpack_unknowns(z, problem.x0)
    info = {
        'iterations': it + 1,
        'residual_norm': final_norm,
        'solver_phase': solver_phase,
        'fallback_used': fallback_used,
        'feasibility_rejections': int(n_feasibility_rejections),
        'projection_fallbacks': int(n_projection_fallbacks),
    }
    return X_sol, P_sol, info
