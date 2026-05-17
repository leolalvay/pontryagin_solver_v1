"""
problem.py
-------------

This module defines the ``OCPProblem`` class, which encapsulates all
problem‑specific data for an optimal control problem (OCP).  It stores
the system dynamics, running cost, terminal cost, initial state, final
horizon, and optional control and state constraints.  The solver code
delegates calls to these methods and properties so that the core
algorithms remain problem‑agnostic.  The interface mirrors the
``OCPProblem`` definition used in the MATLAB implementation.

Functions are vectorized wherever possible.  Control and state bounds
are specified as NumPy arrays of shape ``(m,)`` and ``(n,)``
respectively, where ``m`` is the number of control inputs and ``n`` is
the number of states.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Sequence
import numpy as np


class OCPProblem:
    """Encapsulates an optimal control problem.

    The problem is specified by the continuous dynamics

        ẋ(t) = f(x(t), u(t), t),

    a running cost ℓ(x,u,t), and a terminal cost g(x(T)).  The initial
    state x(0) = x0 and horizon T are fixed.  Optionally, box
    constraints on the control (u) and state (x) can be supplied.

    Parameters
    ----------
    dynamics : Callable[[np.ndarray, np.ndarray, float], np.ndarray]
        The system dynamics function f(x,u,t) → ẋ of dimension n.
    stage_cost : Callable[[np.ndarray, np.ndarray, float], float]
        The running cost ℓ(x,u,t).
    terminal_cost : Callable[[np.ndarray], float]
        The terminal cost g(x).  It is a function of the terminal
        state only.
    x0 : np.ndarray
        Initial state vector of shape (n,).
    T : float
        Final horizon.  Must be positive.
    control_bounds : Optional[Tuple[np.ndarray, np.ndarray]], optional
        Tuple (u_min, u_max) specifying componentwise lower and upper
        bounds for the control.  Each is a 1D array of shape (m,).
        If None, the control is unconstrained.  Default is None.
    state_bounds : Optional[Tuple[np.ndarray, np.ndarray]], optional
        Tuple (x_min, x_max) specifying componentwise bounds for the
        state.  Each is a 1D array of shape (n,).  If None, the
        state is unconstrained.  Default is None.
    """

    def __init__(
        self,
        dynamics: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
        stage_cost: Callable[[np.ndarray, np.ndarray, float], float],
        terminal_cost: Callable[[np.ndarray], float],
        x0: np.ndarray,
        T: float,
        control_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        state_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        hamiltonian_true: Optional[Callable[[np.ndarray, np.ndarray, float], float]] = None,
        u_star_fn: Optional[Callable[[np.ndarray, np.ndarray, float], np.ndarray]] = None,
        hamiltonian_grad_fn: Optional[Callable[[np.ndarray, np.ndarray, float], Tuple[np.ndarray, np.ndarray]]] = None,
        hamiltonian_smooth_fn: Optional[
            Callable[[np.ndarray, np.ndarray, float, float], Tuple[float, np.ndarray, np.ndarray]]
        ] = None,
        barrier_stage_cost_fn: Optional[
            Callable[[np.ndarray, np.ndarray, float, float], float]
        ] = None,
        barrier_grad_x_fn: Optional[
            Callable[[np.ndarray, float, float], np.ndarray]
        ] = None,
        barrier_margin_fn: Optional[
            Callable[[np.ndarray, float], float]
        ] = None,
        u_star_local_fn: Optional[
            Callable[[np.ndarray, np.ndarray, float, bool, float, Optional[float]], Tuple[Optional[np.ndarray], bool]]
        ] = None,
        tangent_ok_fn: Optional[Callable[[np.ndarray, np.ndarray, float, float], bool]] = None,
        step_feasible_control_fn: Optional[
            Callable[[np.ndarray, np.ndarray, float, float, float], bool]
        ] = None,
        feasibility_refinement_fn: Optional[
            Callable[[np.ndarray, np.ndarray, float, float, float], Optional[dict]]
        ] = None,
        state_feasible_fn: Optional[Callable[[np.ndarray, float, float], bool]] = None,
        project_state_fn: Optional[Callable[[np.ndarray, float, float], np.ndarray]] = None,
        fraction_to_boundary_fn: Optional[Callable[[np.ndarray, np.ndarray, float, float, float], float]] = None,
    ) -> None:
        self.f_fn = dynamics
        self.l_fn = stage_cost
        self.g_fn = terminal_cost
        self.x0 = np.asarray(x0, dtype=float)
        self.T = float(T)
        self.hamiltonian_true_fn = hamiltonian_true
        self.u_star_fn = u_star_fn
        self.u_star_local_fn = u_star_local_fn
        self.hamiltonian_grad_fn = hamiltonian_grad_fn
        self.hamiltonian_smooth_fn = hamiltonian_smooth_fn
        self.barrier_stage_cost_fn = barrier_stage_cost_fn
        self.barrier_grad_x_fn = barrier_grad_x_fn
        self.barrier_margin_fn = barrier_margin_fn
        self.mu_barrier = 0.0
        self.tangent_ok_fn = tangent_ok_fn
        self.step_feasible_control_fn = step_feasible_control_fn
        self.feasibility_refinement_fn = feasibility_refinement_fn
        self.state_feasible_fn = state_feasible_fn
        self.project_state_fn = project_state_fn
        self.fraction_to_boundary_fn = fraction_to_boundary_fn
        # copy bounds if provided
        if control_bounds is not None:
            u_min, u_max = control_bounds
            self.u_min = np.asarray(u_min, dtype=float).copy()
            self.u_max = np.asarray(u_max, dtype=float).copy()
        else:
            self.u_min = None
            self.u_max = None
        if state_bounds is not None:
            x_min, x_max = state_bounds
            self.x_min = np.asarray(x_min, dtype=float).copy()
            self.x_max = np.asarray(x_max, dtype=float).copy()
        else:
            self.x_min = None
            self.x_max = None

    # ------------------------------------------------------------------
    # Dynamics and cost wrappers
    def f(self, x: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        """Compute state derivative f(x,u,t).

        Parameters
        ----------
        x : np.ndarray
            State vector of shape (n,).
        u : np.ndarray
            Control vector of shape (m,).
        t : float
            Time.

        Returns
        -------
        np.ndarray
            The state derivative of shape (n,).
        """
        return self.f_fn(x, u, t)

    def l(self, x: np.ndarray, u: np.ndarray, t: float) -> float:
        """Compute running cost ℓ(x,u,t).

        Parameters
        ----------
        x : np.ndarray
        u : np.ndarray
        t : float

        Returns
        -------
        float
        """
        value = float(self.l_fn(x, u, t))
        if self.barrier_stage_cost_fn is not None and float(self.mu_barrier) > 0.0:
            value += float(self.barrier_stage_cost_fn(x, u, t, float(self.mu_barrier)))
        return value

    def g(self, x: np.ndarray) -> float:
        """Compute terminal cost g(x).

        Parameters
        ----------
        x : np.ndarray
            Terminal state vector of shape (n,).

        Returns
        -------
        float
        """
        return self.g_fn(x)


    def u_star(
        self,
        x: np.ndarray,
        p: np.ndarray,
        t: float,
        restricted: bool = False,
        tol: float = 1e-8,
        dt: Optional[float] = None,
    ):
        """
        Returns (u, feasible). If u_star_fn is not provided -> (None, False).
        Always projects to control bounds (if any). If restricted=True, checks tangent_ok.
        """
        if self.u_star_local_fn is not None:
            u_raw, feasible = self.u_star_local_fn(x, p, t, restricted, tol, dt)
            if u_raw is None:
                return None, False
            u = np.atleast_1d(np.asarray(u_raw, dtype=float))
            u = self.project_control(u)
            return u, bool(feasible)

        if self.u_star_fn is None:
            return None, False

        u_raw = self.u_star_fn(x, p, t)
        u = np.atleast_1d(np.asarray(u_raw, dtype=float))
        u = self.project_control(u)

        if restricted and not self.local_control_feasible(x, u, t, dt=dt, tol=tol):
            return u, False

        return u, True

    def hamiltonian_true(
        self,
        x: np.ndarray,
        p: np.ndarray,
        t: float,
        restricted: bool = False,
        tol: float = 1e-8,
        dt: Optional[float] = None,
    ):
        """
        Returns (H, u, feasible).

        - Uses u_star_fn if available (through self.u_star), so bounds are enforced.
        - If restricted=True and u_star is not feasible, returns (inf, u, False).
        """
        u, ok = self.u_star(x, p, t, restricted=restricted, tol=tol, dt=dt)

        if u is None:
            raise ValueError("u_star_fn is not provided, cannot compute true Hamiltonian from u_star.")

        if restricted and not ok:
            return float("inf"), u, False

        H = float(p @ self.f_fn(x, u, t) + self.l_fn(x, u, t))
        return H, u, True

    def hamiltonian_gradients(self, x, p, t):
        if self.hamiltonian_grad_fn is None:
            raise ValueError("hamiltonian_grad_fn is not provided.")
        grad_p, grad_x = self.hamiltonian_grad_fn(x, p, t)
        grad_p = np.atleast_1d(np.asarray(grad_p, dtype=float))
        grad_x = np.atleast_1d(np.asarray(grad_x, dtype=float))
        if self.barrier_grad_x_fn is not None and float(self.mu_barrier) > 0.0:
            grad_x = grad_x + np.atleast_1d(np.asarray(self.barrier_grad_x_fn(x, t, float(self.mu_barrier)), dtype=float))
        return grad_p, grad_x

    def hamiltonian_smooth(self, x, p, t, delta):
        if self.hamiltonian_smooth_fn is None:
            raise ValueError("hamiltonian_smooth_fn is not provided.")
        H_delta, grad_p, grad_x = self.hamiltonian_smooth_fn(x, p, t, delta)
        grad_p = np.atleast_1d(np.asarray(grad_p, dtype=float))
        grad_x = np.atleast_1d(np.asarray(grad_x, dtype=float))
        if self.barrier_stage_cost_fn is not None and float(self.mu_barrier) > 0.0:
            H_delta = float(H_delta) + float(self.barrier_stage_cost_fn(x, np.zeros(self.m or 1), t, float(self.mu_barrier)))
        if self.barrier_grad_x_fn is not None and float(self.mu_barrier) > 0.0:
            grad_x = grad_x + np.atleast_1d(np.asarray(self.barrier_grad_x_fn(x, t, float(self.mu_barrier)), dtype=float))
        return float(H_delta), grad_p, grad_x

    def barrier_margin(self, x: np.ndarray, t: float) -> float:
        if self.barrier_margin_fn is None:
            raise ValueError("barrier_margin_fn is not provided.")
        return float(self.barrier_margin_fn(x, t))


    # ------------------------------------------------------------------
    # Control bounds
    def get_control_bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return control bounds (u_min, u_max) if specified.

        Returns
        -------
        Optional[Tuple[np.ndarray, np.ndarray]]
            Tuple of lower and upper control bounds, or None if
            unconstrained.
        """
        if self.u_min is None or self.u_max is None:
            return None
        return (self.u_min.copy(), self.u_max.copy())

    def admissible_control(self, u: np.ndarray, x: Optional[np.ndarray] = None, t: Optional[float] = None) -> bool:
        """Check if control u satisfies the control bounds.

        Parameters
        ----------
        u : np.ndarray
            Control vector.
        x, t : optional
            State and time (unused here but kept for uniform interface).

        Returns
        -------
        bool
            True if u is within bounds or no bounds defined.
        """
        if self.u_min is None or self.u_max is None:
            return True
        return np.all(u >= self.u_min) and np.all(u <= self.u_max)

    def step_feasible_control(
        self,
        x: np.ndarray,
        u: np.ndarray,
        t: float,
        dt: float,
        tol: float = 1e-8,
    ) -> bool:
        if self.step_feasible_control_fn is not None:
            return bool(self.step_feasible_control_fn(x, u, t, dt, tol))
        x_trial = np.asarray(x, dtype=float) + float(dt) * self.f(np.asarray(x, dtype=float), np.asarray(u, dtype=float), t)
        return self.state_feasible(x_trial, float(t) + float(dt), tol=tol)

    def local_control_feasible(
        self,
        x: np.ndarray,
        u: np.ndarray,
        t: float,
        *,
        restricted: bool = True,
        dt: Optional[float] = None,
        tol: float = 1e-8,
    ) -> bool:
        if not self.admissible_control(u):
            return False
        if not restricted:
            return True
        if dt is not None:
            if not self.state_feasible(x, t, tol=tol):
                return True
            return self.step_feasible_control(x, u, t, dt, tol=tol)
        return self.tangent_ok(x, u, t, tol=tol)

    def project_control(self, u: np.ndarray) -> np.ndarray:
        """Project control u onto the control bounds by clipping.

        If no bounds are defined, returns u unchanged.

        Parameters
        ----------
        u : np.ndarray
            Control vector.

        Returns
        -------
        np.ndarray
            The clipped control.
        """
        if self.u_min is None or self.u_max is None:
            return u
        return np.minimum(np.maximum(u, self.u_min), self.u_max)

    # ------------------------------------------------------------------
    # State bounds
    def get_state_bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return state bounds (x_min, x_max) if specified.

        Returns
        -------
        Optional[Tuple[np.ndarray, np.ndarray]]
            Tuple of lower and upper state bounds, or None if
            unconstrained.
        """
        if self.x_min is None or self.x_max is None:
            return None
        return (self.x_min.copy(), self.x_max.copy())

    def admissible_state(self, x: np.ndarray) -> bool:
        """Check if state x satisfies the state bounds.

        Parameters
        ----------
        x : np.ndarray
            State vector.

        Returns
        -------
        bool
            True if x is within bounds or no bounds defined.
        """
        if self.x_min is None or self.x_max is None:
            return True
        return np.all(x >= self.x_min) and np.all(x <= self.x_max)

    def state_feasible(self, x: np.ndarray, t: float, tol: float = 1e-8) -> bool:
        x = np.asarray(x, dtype=float)
        if self.state_feasible_fn is not None:
            return bool(self.state_feasible_fn(x, t, tol))
        return self.admissible_state(x)

    def trajectory_feasible(self, X: np.ndarray, t_nodes: np.ndarray, tol: float = 1e-8) -> bool:
        X = np.asarray(X, dtype=float)
        t_nodes = np.asarray(t_nodes, dtype=float)
        if X.ndim != 2 or t_nodes.ndim != 1 or X.shape[0] != t_nodes.size:
            raise ValueError("X must have shape (N+1,n) and t_nodes must have length N+1.")
        for i, t_i in enumerate(t_nodes):
            if not self.state_feasible(X[i], float(t_i), tol=tol):
                return False
        return True

    def project_state(self, x: np.ndarray, t: float, tol: float = 1e-8) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.project_state_fn is not None:
            projected = np.asarray(self.project_state_fn(x, t, tol), dtype=float)
            return projected
        if self.x_min is None or self.x_max is None:
            return x.copy()
        return np.minimum(np.maximum(x, self.x_min), self.x_max)

    def project_trajectory(self, X: np.ndarray, t_nodes: np.ndarray, tol: float = 1e-8) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        t_nodes = np.asarray(t_nodes, dtype=float)
        if X.ndim != 2 or t_nodes.ndim != 1 or X.shape[0] != t_nodes.size:
            raise ValueError("X must have shape (N+1,n) and t_nodes must have length N+1.")
        projected = X.copy()
        projected[0] = self.x0.copy()
        for i in range(1, t_nodes.size):
            projected[i] = self.project_state(projected[i], float(t_nodes[i]), tol=tol)
        return projected

    def fraction_to_boundary_step(
        self,
        X: np.ndarray,
        dX: np.ndarray,
        t_nodes: np.ndarray,
        safety: float = 0.99,
        tol: float = 1e-8,
    ) -> float:
        X = np.asarray(X, dtype=float)
        dX = np.asarray(dX, dtype=float)
        t_nodes = np.asarray(t_nodes, dtype=float)
        if X.shape != dX.shape or X.shape[0] != t_nodes.size:
            raise ValueError("X, dX, and t_nodes must be trajectory-aligned.")
        if self.fraction_to_boundary_fn is None:
            return 1.0
        lam = 1.0
        for i, t_i in enumerate(t_nodes):
            lam_i = float(self.fraction_to_boundary_fn(X[i], dX[i], float(t_i), safety, tol))
            if np.isfinite(lam_i):
                lam = min(lam, lam_i)
        return float(np.clip(lam, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Convenience accessors used by adaptivity module
    def control_bounds_tuple(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return the control bounds as a tuple (u_min, u_max) or None.

        The adaptivity module calls this method to obtain the control
        range when initializing the PA bundle.  If no bounds are set,
        returns None.

        Returns
        -------
        Optional[Tuple[np.ndarray, np.ndarray]]
        """
        return self.get_control_bounds()

    def state_bounds_tuple(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return the state bounds as a tuple (x_min, x_max) or None.
        Similar to ``control_bounds_tuple`` for states.
        """
        return self.get_state_bounds()

    @property
    def n(self) -> int:
        """Alias for state dimension (number of states).  Provided for
        compatibility with the MATLAB version and the adaptivity
        implementation.  Equivalent to ``state_dim``.
        """
        return self.state_dim

    @property
    def m(self) -> Optional[int]:
        """Alias for control dimension.  Provided for compatibility.

        Returns
        -------
        Optional[int]
            The dimension of the control if control bounds are known,
            otherwise None.
        """
        if self.u_min is not None:
            return self.u_min.size
        if self.u_max is not None:
            return self.u_max.size
        # unknown until a control vector is passed to f or l; return None
        return None

    # ------------------------------------------------------------------
    # Viability / tangent cone filter
    def tangent_cone_filter(self, x: np.ndarray, f_candidates: Sequence[np.ndarray], tol: float = 1e-8) -> np.ndarray:
        """Return a boolean mask of candidates that lie in the tangent cone of the state constraints.

        For each candidate derivative f(x,a,t) in ``f_candidates``, this
        checks whether moving in that direction at the current state x
        remains within the state bounds K, using a box constraint model.

        If ``self.x_min`` or ``self.x_max`` are None (i.e. no state
        constraints), all candidates are marked as feasible.

        Parameters
        ----------
        x : np.ndarray of shape (n,)
            Current state.
        f_candidates : Sequence[np.ndarray]
            List or array of candidate state derivatives of shape (k, n) or
            list of shape (n,).
        tol : float
            Tolerance for boundary checks.  If ``|x_i - x_min_i| < tol``
            or ``|x_i - x_max_i| < tol`` then x is treated as on the
            boundary in dimension i.

        Returns
        -------
        np.ndarray
            Boolean array of length equal to len(f_candidates), where
            True indicates the candidate derivative is viable (does not
            point outside the feasible set).
        """
        # If no state bounds, all are viable
        if self.x_min is None or self.x_max is None:
            return np.ones(len(f_candidates), dtype=bool)
        x = np.asarray(x, dtype=float)
        feasible = np.ones(len(f_candidates), dtype=bool)
        for idx, f_vec in enumerate(f_candidates):
            f_vec = np.asarray(f_vec, dtype=float)
            ok = True
            for i in range(x.size):
                if (self.x_min is not None and abs(x[i] - self.x_min[i]) < tol and f_vec[i] < 0.0):
                    # on lower boundary and pointing outwards
                    ok = False
                    break
                if (self.x_max is not None and abs(x[i] - self.x_max[i]) < tol and f_vec[i] > 0.0):
                    ok = False
                    break
            feasible[idx] = ok
        return feasible

    def tangent_ok(self, x: np.ndarray, u: np.ndarray, t: float, tol: float = 1e-8) -> bool:
        """Check if the velocity f(x,u,t) lies in the tangent cone of state constraints.

        This convenience method evaluates the state derivative at (x,u,t) and
        then calls :meth:`tangent_cone_filter` on that single vector.  It
        returns True if the motion does not violate the box constraints
        (i.e. it points inward or tangent on active boundaries).

        Parameters
        ----------
        x : np.ndarray
            State vector.
        u : np.ndarray
            Control vector.
        t : float
            Time.
        tol : float
            Boundary tolerance for viability.

        Returns
        -------
        bool
            True if f(x,u,t) ∈ T_K(x) or if no state constraints.
        """
        if self.tangent_ok_fn is not None:
            return bool(self.tangent_ok_fn(x, u, t, tol))
        # if no state bounds defined, everything is viable
        if self.x_min is None or self.x_max is None:
            return True
        f_vec = self.f(x, u, t)
        mask = self.tangent_cone_filter(x, [f_vec], tol)
        return bool(mask[0])

    # ------------------------------------------------------------------
    # Helper for dimension inference
    @property
    def state_dim(self) -> int:
        """Return the dimension of the state vector."""
        return self.x0.size

    @property
    def control_dim(self) -> int:
        """Return the dimension of the control vector.  If control bounds
        are not specified, this is inferred the first time a control
        vector is passed to ``f`` or ``l``.  Here we attempt to infer
        from bounds if available.  If bounds are None, returns 0
        (caller should handle appropriately).
        """
        if self.u_min is not None:
            return self.u_min.size
        elif self.u_max is not None:
            return self.u_max.size
        else:
            return 0

    # ------------------------------------------------------------------
    # User convenience wrappers
    def __repr__(self) -> str:
        return (f"OCPProblem(state_dim={self.state_dim}, control_dim={self.control_dim}, "
                f"T={self.T}, bounds={'yes' if self.u_min is not None else 'no'})")
