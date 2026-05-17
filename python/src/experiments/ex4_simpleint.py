"""
Example 4: 1D integrator with bounded control (PDF Example 7).

Minimize   ∫_0^T 0.5 * x(t)^2 dt
subject to x'(t) = u(t),  u(t) in [-1, 1],  x(0)=x0,  g(x(T))=0.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from scipy.sparse import issparse
from core.problem import OCPProblem
from core.adaptivity import solve_optimal_control
from core.integrators import pack_unknowns
from core.shooting import shooting_residual, shooting_jacobian


def run_example():
    # ============================================================
    # 0) Problem definition (matches PDF Example 7)
    # ============================================================
    x0 = np.array([0.5])   # scalar state, stored as shape (1,)
    T = 1.0

    def dynamics(x, u, t):
        # x, u are numpy arrays of shape (1,)
        return np.array([u[0]])

    def stage_cost(x, u, t):
        return 0.5 * float(x[0] ** 2)

    def terminal_cost(x):
        return 0.0

    u_min = np.array([-1.0])
    u_max = np.array([1.0])

    prob = OCPProblem(
        dynamics=dynamics,
        stage_cost=stage_cost,
        terminal_cost=terminal_cost,
        x0=x0,
        T=T,
        control_bounds=(u_min, u_max),
        state_bounds=None,
    )

    # ============================================================
    # 1) Solve with the repo's adaptive outer loop
    # ============================================================
    t_nodes = np.linspace(0.0, T, 30)  # dt = 0.02 (fast initial mesh)
    dt0 = t_nodes[1] - t_nodes[0]

    result = solve_optimal_control(
        prob,
        t_nodes,
        tol_time=1e-3,   # relaxed -> fewer refinements
        tol_PA=1e-4,
        tol_delta=2e-3,
        max_iters=8,
        delta0=0.02,
    )

    print("\nExample 4 (Simple Integrator / PDF Ex7)")
    print("len(log) =", len(result["log"]))
    print("last outer iter =", result["log"][-1]["iteration"])
    print("len(t_nodes) =", len(result["t_nodes"]))
    print("X.shape =", result["X"].shape)
    print("P.shape =", result["P"].shape)
    print("final delta =", result["delta"])
    print("last log entry =", result["log"][-1])

    # ============================================================
    # 2) Sanity check vs exact solution from the PDF (for x0>0)
    #    Exact:
    #      x(t) = max(x0 - t, 0)
    #      p(t) = 0.5*(t - x0)^2   for t <= x0, else 0
    # ============================================================
    t = np.asarray(result["t_nodes"])
    X = np.asarray(result["X"])[:, 0]
    P = np.asarray(result["P"])[:, 0]
    x0_scalar = float(x0[0])

    X_exact = np.maximum(x0_scalar - t, 0.0)
    P_exact = np.where(t <= x0_scalar, 0.5 * (t - x0_scalar) ** 2, 0.0)

    err_X_inf = float(np.max(np.abs(X - X_exact)))
    err_P_inf = float(np.max(np.abs(P - P_exact)))
    print("||X - X_exact||_inf =", err_X_inf)
    print("||P - P_exact||_inf =", err_P_inf)

    # Plot X and P (two windows at once)
    fig1 = plt.figure()
    plt.plot(t, X, label="X (solver)")
    plt.plot(t, X_exact, "--", label="X exact")
    plt.xlabel("t")
    plt.ylabel("X")
    plt.title("Example 4: State X(t)")
    plt.legend()
    plt.grid(True)
    plt.savefig("example4_state_X.pdf", format="pdf", bbox_inches="tight")


    fig2 = plt.figure()
    plt.plot(t, P, label="P (solver)")
    plt.plot(t, P_exact, "--", label="P exact")
    plt.xlabel("t")
    plt.ylabel("P")
    plt.title("Example 4: Costate P(t)")
    plt.legend()
    plt.grid(True)
    plt.savefig("example4_costate_P.pdf", format="pdf", bbox_inches="tight")


   

    # Show all figures at once
    plt.show()

    # Close figs (optional)
    plt.close(fig1)
    plt.close(fig2)

    

    return result


if __name__ == "__main__":
    run_example()
