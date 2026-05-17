import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pathlib import Path
from functools import partial

def build_lqr_example_1():
    """
    Return the data for Example 1 in the manuscript.

    The optimal control problem is:
        min_u x(T)^T Q_T x(T) + integral_0^T [x(t)^T Q x(t) + u(t)^T R u(t)] dt
    subject to
        x_dot = A x + B u,    x(0) = x0.

    Returns
    -------
    data : dict
        Dictionary containing the matrices and vectors:
        A, B, Q, Q_T, R, R_inv, T, x0.
    """
    A = np.array([[0.0, 1.0],
                  [0.0, 0.0]])

    B = np.array([[0.0],
                  [1.0]])

    Q = np.eye(2)
    Q_T = np.eye(2)
    R = np.array([[1.0e-2]])
    R_inv = np.linalg.inv(R)

    T = 1.0
    x0 = np.array([1.0, 0.0])

    return {
        "A": A,
        "B": B,
        "Q": Q,
        "Q_T": Q_T,
        "R": R,
        "R_inv": R_inv,
        "T": T,
        "x0": x0,
    }


def riccati_rhs(t, P_flat, A, B, Q, R_inv):
    """
    Right-hand side of the continuous-time Riccati ODE written forward in time.

    We use the convention
        p(t) = 2 P(t) x(t),
    so P solves
        -P_dot = A^T P + P A - P B R^{-1} B^T P + Q,
        P(T) = Q_T.

    Therefore, when written as P_dot = F(P),
        P_dot = -A^T P - P A + P B R^{-1} B^T P - Q.

    Parameters
    ----------
    t : float
        Time variable (included for compatibility with ODE solvers).
    P_flat : ndarray of shape (n*n,)
        Flattened matrix P(t).
    A, B, Q, R_inv : ndarray
        LQR problem data.

    Returns
    -------
    dP_flat : ndarray of shape (n*n,)
        Flattened value of P_dot.
    """
    n = A.shape[0]
    #P_flat.reshape(n,n) converts the flattened (1D) array P_flat into a matrix n*n
    P = P_flat.reshape(n, n)

    dP = -A.T @ P - P @ A + P @ B @ R_inv @ B.T @ P - Q

    #reshape(-1) flattens the matrix into a 1-dim array
    return dP.reshape(-1)




def solve_riccati_equation(data, num_eval_points=1000, rtol=1e-10, atol=1e-12):
    """
    Solve the Riccati ODE for Example 1 backward in time.

    We solve
        -P_dot = A^T P + P A - P B R^{-1} B^T P + Q,
        P(T) = Q_T,
    equivalently
        P_dot = -A^T P - P A + P B R^{-1} B^T P - Q.

    The integration is performed backward from t = T to t = 0.

    Parameters
    ----------
    data : dict
        Output of build_lqr_example_1().
    num_eval_points : int, optional
        Number of time points used to store the numerical solution.
    rtol : float, optional
        Relative tolerance for solve_ivp.
    atol : float, optional
        Absolute tolerance for solve_ivp.

    Returns
    -------
    result : dict
        Dictionary containing:
        - "t_grid_desc": descending time grid from T to 0
        - "P_grid_desc": array of shape (m, n, n) with P(t) on the descending grid
        - "sol": raw solve_ivp solution object
        - "P_of_t": callable returning P(t) for any t in [0, T]
        - "terminal_condition_error": norm of P(T) - Q_T
    """
    A = data["A"]
    B = data["B"]
    Q = data["Q"]
    Q_T = data["Q_T"]
    R_inv = data["R_inv"]
    T = data["T"]

    n = A.shape[0]
    P_T_flat = Q_T.reshape(-1)
    #time grid of num_eval_points points descending, i.e., from T to 0
    t_eval_desc = np.linspace(T, 0.0, num_eval_points)

    sol = solve_ivp(
        fun=lambda t, y: riccati_rhs(t, y, A, B, Q, R_inv),
        t_span=(T, 0.0),
        y0=P_T_flat, #flattened terminal condition
        t_eval=t_eval_desc,
        method="RK45",
        rtol=rtol,
        atol=atol,
        dense_output=True,
    )

    if not sol.success:
        raise RuntimeError(f"Riccati solver failed: {sol.message}")

    #reshape solution into a matrix of shape (N,n,n)
    P_grid_desc = sol.y.T.reshape(-1, n, n)
    #P_grid_desc[0] is P(T) and P_grid_desc[-1] is P(0)

    def P_of_t(t):
        """
        Evaluate the Riccati solution P(t) at a scalar time t in [0, T].
        """
        t = float(t)
        if not (0.0 <= t <= T):
            raise ValueError(f"t must lie in [0, {T}]")
        return sol.sol(t).reshape(n, n)

    terminal_condition_error = np.linalg.norm(P_grid_desc[0] - Q_T)

    return {
        "t_grid_desc": sol.t,
        "P_grid_desc": P_grid_desc,
        "sol": sol,
        "P_of_t": P_of_t,
        "terminal_condition_error": terminal_condition_error,
    }

def riccati_diagnostics(data, riccati_result):
    """
    Compute basic diagnostics for the Riccati solution.

    Diagnostics
    -----------
    1. Terminal condition error:
           ||P(T) - Q_T||
    2. Maximum symmetry defect on the stored grid:
           max_t ||P(t) - P(t)^T||
    3. Initial Riccati matrix:
           P(0)

    Parameters
    ----------
    data : dict
        Output of build_lqr_example_1().
    riccati_result : dict
        Output of solve_riccati_equation().

    Returns
    -------
    diagnostics : dict
        Dictionary with diagnostic quantities.
    """
    Q_T = data["Q_T"]
    P_grid_desc = riccati_result["P_grid_desc"]
    t_grid_desc = riccati_result["t_grid_desc"]

    terminal_condition_error = np.linalg.norm(P_grid_desc[0] - Q_T)

    symmetry_errors = np.array([
        np.linalg.norm(P - P.T) for P in P_grid_desc
    ])
    max_symmetry_error = np.max(symmetry_errors)
    argmax_symmetry_error = t_grid_desc[np.argmax(symmetry_errors)]

    P0 = P_grid_desc[-1]

    return {
        "terminal_condition_error": terminal_condition_error,
        "max_symmetry_error": max_symmetry_error,
        "time_of_max_symmetry_error": argmax_symmetry_error,
        "P0": P0,
    }


def state_rhs(t, x, A, B, R_inv, P_of_t):
    """
    Right-hand side of the closed-loop state equation

        x_dot = (A - B R^{-1} B^T P(t)) x.

    Parameters
    ----------
    t : float
        Time.
    x : ndarray of shape (n,)
        State vector.
    A, B, R_inv : ndarray
        LQR problem data.
    P_of_t : callable
        Function returning the Riccati matrix P(t).

    Returns
    -------
    dx : ndarray of shape (n,)
        Time derivative x_dot.
    """
    P = P_of_t(t)
    A_cl = A - B @ R_inv @ B.T @ P
    return A_cl @ x


def solve_state_equation(data, riccati_result, num_eval_points=1000, rtol=1e-10, atol=1e-12):
    """
    Solve the closed-loop state equation forward in time.

    Parameters
    ----------
    data : dict
        Output of build_lqr_example_1().
    riccati_result : dict
        Output of solve_riccati_equation().
    num_eval_points : int, optional
        Number of stored time points.
    rtol : float, optional
        Relative tolerance for solve_ivp.
    atol : float, optional
        Absolute tolerance for solve_ivp.

    Returns
    -------
    result : dict
        Dictionary containing:
        - "t_grid": ascending time grid from 0 to T
        - "x_grid": array of shape (m, n)
        - "u_grid": array of shape (m, m_u)
        - "p_grid": array of shape (m, n)
        - "sol": raw solve_ivp solution object
        - "initial_condition_error": norm of x(0) - x0
    """
    A = data["A"]
    B = data["B"]
    R_inv = data["R_inv"]
    x0 = data["x0"]
    T = data["T"]

    P_of_t = riccati_result["P_of_t"]

    t_eval = np.linspace(0.0, T, num_eval_points)

    sol = solve_ivp(
        fun=lambda t, y: state_rhs(t, y, A, B, R_inv, P_of_t),
        t_span=(0.0, T),
        y0=x0,
        t_eval=t_eval,
        method="RK45",
        rtol=rtol,
        atol=atol,
        dense_output=True,
    )

    if not sol.success:
        raise RuntimeError(f"State solver failed: {sol.message}")

    x_grid = sol.y.T

    u_list = []
    p_list = []
    for t, x in zip(sol.t, x_grid):
        P = P_of_t(t)
        u = -R_inv @ B.T @ P @ x
        p = 2.0 * P @ x
        u_list.append(u.reshape(-1))
        p_list.append(p)

    u_grid = np.vstack(u_list)
    p_grid = np.vstack(p_list)

    initial_condition_error = np.linalg.norm(x_grid[0] - x0)

    return {
        "t_grid": sol.t,
        "x_grid": x_grid,
        "u_grid": u_grid,
        "p_grid": p_grid,
        "sol": sol,
        "initial_condition_error": initial_condition_error,
    }


def compute_lqr_cost(data, state_result):
    """
    Compute the LQR objective value

        J = x(T)^T Q_T x(T) + integral_0^T [x(t)^T Q x(t) + u(t)^T R u(t)] dt

    using the stored time grid and the trapezoidal rule.

    Parameters
    ----------
    data : dict
        Output of build_lqr_example_1().
    state_result : dict
        Output of solve_state_equation().

    Returns
    -------
    J : float
        Numerical approximation of the objective value.
    running_cost_integral : float
        Numerical approximation of the running-cost integral.
    terminal_cost : float
        Terminal contribution x(T)^T Q_T x(T).
    """
    Q = data["Q"]
    Q_T = data["Q_T"]
    R = data["R"]

    t_grid = state_result["t_grid"]
    x_grid = state_result["x_grid"]
    u_grid = state_result["u_grid"]

    running_cost_values = np.array([
        x @ Q @ x + u @ R @ u
        for x, u in zip(x_grid, u_grid)
    ])

    running_cost_integral = np.trapezoid(running_cost_values, t_grid)

    x_T = x_grid[-1]
    terminal_cost = x_T @ Q_T @ x_T

    J = terminal_cost + running_cost_integral

    return J, running_cost_integral, terminal_cost


def pmp_diagnostics(data, riccati_result, state_result):
    """
    Compute basic PMP consistency diagnostics.

    Diagnostics
    -----------
    1. Terminal costate condition:
           ||p(T) - 2 Q_T x(T)||
    2. Stationarity condition:
           max_t ||u(t) + (1/2) R^{-1} B^T p(t)||
    3. Riccati symmetry consistency already handled separately.

    Parameters
    ----------
    data : dict
        Output of build_lqr_example_1().
    riccati_result : dict
        Output of solve_riccati_equation().
    state_result : dict
        Output of solve_state_equation().

    Returns
    -------
    diagnostics : dict
        Dictionary with PMP diagnostic quantities.
    """
    B = data["B"]
    Q_T = data["Q_T"]
    R_inv = data["R_inv"]

    t_grid = state_result["t_grid"]
    x_grid = state_result["x_grid"]
    u_grid = state_result["u_grid"]
    p_grid = state_result["p_grid"]

    terminal_costate_error = np.linalg.norm(
        p_grid[-1] - 2.0 * Q_T @ x_grid[-1]
    )

    stationarity_errors = np.array([
        np.linalg.norm(u + 0.5 * (R_inv @ B.T @ p))
        for u, p in zip(u_grid, p_grid)
    ])

    max_stationarity_error = np.max(stationarity_errors)
    time_of_max_stationarity_error = t_grid[np.argmax(stationarity_errors)]

    return {
        "terminal_costate_error": terminal_costate_error,
        "max_stationarity_error": max_stationarity_error,
        "time_of_max_stationarity_error": time_of_max_stationarity_error,
    }

def run_lqr_riccati_benchmark(
    riccati_num_eval_points=1000,
    riccati_rtol=1e-10,
    riccati_atol=1e-12,
    state_num_eval_points=1000,
    state_rtol=1e-10,
    state_atol=1e-12,
):
    """
    Run the full Riccati-based benchmark for LQR Example 1.

    Workflow
    --------
    1. Build problem data.
    2. Solve the Riccati equation backward in time.
    3. Compute Riccati diagnostics.
    4. Solve the closed-loop state equation forward in time.
    5. Compute the LQR cost.
    6. Compute PMP diagnostics.

    Parameters
    ----------
    riccati_num_eval_points : int, optional
        Number of stored time points for the Riccati solution.
    riccati_rtol : float, optional
        Relative tolerance for the Riccati ODE solver.
    riccati_atol : float, optional
        Absolute tolerance for the Riccati ODE solver.
    state_num_eval_points : int, optional
        Number of stored time points for the state solution.
    state_rtol : float, optional
        Relative tolerance for the state ODE solver.
    state_atol : float, optional
        Absolute tolerance for the state ODE solver.

    Returns
    -------
    result : dict
        Dictionary containing:
        - "data"
        - "riccati_result"
        - "riccati_diagnostics"
        - "state_result"
        - "cost_info"
        - "pmp_diagnostics"
    """
    data = build_lqr_example_1()

    riccati_result = solve_riccati_equation(
        data,
        num_eval_points=riccati_num_eval_points,
        rtol=riccati_rtol,
        atol=riccati_atol,
    )

    riccati_info = riccati_diagnostics(data, riccati_result)

    state_result = solve_state_equation(
        data,
        riccati_result,
        num_eval_points=state_num_eval_points,
        rtol=state_rtol,
        atol=state_atol,
    )

    J, running_cost_integral, terminal_cost = compute_lqr_cost(data, state_result)

    cost_info = {
        "J": J,
        "running_cost_integral": running_cost_integral,
        "terminal_cost": terminal_cost,
    }

    pmp_info = pmp_diagnostics(data, riccati_result, state_result)

    return {
        "data": data,
        "riccati_result": riccati_result,
        "riccati_diagnostics": riccati_info,
        "state_result": state_result,
        "cost_info": cost_info,
        "pmp_diagnostics": pmp_info,
    }

def save_plot(fig, stem, fig_dir, ext="pdf"):
    """
    Save one figure to disk and close it.
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def keep_plot(fig, stem=None):
    """
    Keep figure open in interactive mode (no save).
    """
    pass

def plot_lqr_trajectories(
    result,
    out_prefix="ex1_test",
    save_plots=False,
    plot_ext="pdf",
    fig_dir=None,
):
    """
    Plot (or save) the state, control, and costate trajectories of the LQR benchmark.

    Parameters
    ----------
    result : dict
        Output of run_lqr_riccati_benchmark().
    out_prefix : str, optional
        Prefix used for saved figure filenames.
    save_plots : bool, optional
        If True, save figures to disk. If False, display with plt.show().
    plot_ext : str, optional
        File extension for saved figures (e.g., "pdf", "png").
    fig_dir : str or pathlib.Path, optional
        Output directory for saved figures. Defaults to "<this file>/figures".
    """
    state_result = result["state_result"]

    t_grid = state_result["t_grid"]
    x_grid = state_result["x_grid"]
    u_grid = state_result["u_grid"]
    p_grid = state_result["p_grid"]

    if fig_dir is None:
        fig_dir = Path(__file__).resolve().parent / "figures"
    else:
        fig_dir = Path(fig_dir)

    plot_action = partial(save_plot, fig_dir=fig_dir, ext=plot_ext) if save_plots else keep_plot
    render_plots = (lambda: None) if save_plots else plt.show

    fig = plt.figure(figsize=(8, 5))
    plt.plot(t_grid, x_grid[:, 0], label="x1(t)")
    plt.plot(t_grid, x_grid[:, 1], label="x2(t)")
    plt.xlabel("t")
    plt.ylabel("state")
    plt.title("State trajectories")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_action(fig, f"{out_prefix}_state_x")

    fig = plt.figure(figsize=(8, 5))
    plt.plot(t_grid, u_grid[:, 0], label="u(t)")
    plt.xlabel("t")
    plt.ylabel("control")
    plt.title("Control trajectory")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_action(fig, f"{out_prefix}_control_u")

    fig = plt.figure(figsize=(8, 5))
    plt.plot(t_grid, p_grid[:, 0], label="p1(t)")
    plt.plot(t_grid, p_grid[:, 1], label="p2(t)")
    plt.xlabel("t")
    plt.ylabel("costate")
    plt.title("Costate trajectories")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_action(fig, f"{out_prefix}_costate_p")

    render_plots()

def main():
    """
    Entry point for the standalone Riccati benchmark script.
    """
    # Toggle this flag:
    # - False: interactive display via plt.show()
    # - True : save figures to disk (PDF by default)
    save_plots = True

    result = run_lqr_riccati_benchmark()

    print("=== LQR Riccati Benchmark: Example 1 ===")
    print(f"J = {result['cost_info']['J']:.12e}")
    print(f"Running cost integral = {result['cost_info']['running_cost_integral']:.12e}")
    print(f"Terminal cost = {result['cost_info']['terminal_cost']:.12e}")
    print()

    print("Riccati diagnostics")
    print(f"  Terminal condition error = {result['riccati_diagnostics']['terminal_condition_error']:.12e}")
    print(f"  Max symmetry error = {result['riccati_diagnostics']['max_symmetry_error']:.12e}")
    print(f"  Time of max symmetry error = {result['riccati_diagnostics']['time_of_max_symmetry_error']:.12e}")
    print()

    print("PMP diagnostics")
    print(f"  Terminal costate error = {result['pmp_diagnostics']['terminal_costate_error']:.12e}")
    print(f"  Max stationarity error = {result['pmp_diagnostics']['max_stationarity_error']:.12e}")
    print(f"  Time of max stationarity error = {result['pmp_diagnostics']['time_of_max_stationarity_error']:.12e}")

    plot_lqr_trajectories(
        result,
        out_prefix="example1_test",
        save_plots=save_plots,
        plot_ext="pdf",
    )
    return result


if __name__ == "__main__":
    main()