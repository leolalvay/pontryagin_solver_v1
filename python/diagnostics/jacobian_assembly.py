"""
This file uses example 4 from experiments to test the assemble_jacobian function, 
more specifically we test in this file:

1. Assembly method by block for the jacobian matrix, which improves the original 
implementation. For a more detailed description look at the file integrators.md in 
docs/core/integrators.md

2.This file also plots the sparsity pattern of the jacobian. The first plot shows
the original plot and the second plot shows the pattern after a permutation.

3. This file also uses the modified solver, which now is sparse solver and previously
was a standard dense solver. This change shows a significant improvement in 
computational time as the results of this file shows.


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
    x0 = np.array([0.8])   # scalar state, stored as shape (1,)
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
    t_nodes = np.linspace(0.0, T, 3)  # dt = 0.02 (fast initial mesh)
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


    # ============================================================
    # 3) Jacobian diagnostics (this is our current task)
    #
    # Key point:
    #   - F (rows) is assembled interleaved by time: (r_x^0, r_p^0, r_x^1, r_p^1, ...)
    #   - z (columns) is ordered by variable: z = (x1..xN, p0..pN)
    #
    # Therefore:
    #   - spy(J) shows separated diagonals (same sparsity, but "opened" bandwidth)
    #   - if we ONLY permute columns to a time-interleaved unknown order,
    #     the pattern becomes tridiagonal (here n=1), matching the PDF intuition.
    # ============================================================
    z = pack_unknowns(result["X"], result["P"])
    delta = float(result["delta"])

    # Time the expensive pieces: building J vs solving the linear system
    t0 = time.perf_counter()
    F = shooting_residual(prob, result["t_nodes"], z, result["bundle"], delta)
    t1 = time.perf_counter()
    J = shooting_jacobian(prob, result["t_nodes"], z, result["bundle"], delta)
    t2 = time.perf_counter()

    rhs = -F
    # ---- Dense solve timing ----
    t3 = time.perf_counter()
    dz_dense = np.linalg.solve(J.toarray() if hasattr(J, "toarray") else J, rhs)
    t4 = time.perf_counter()

    # ---- Sparse solve timing (same J, converted to CSC) ----
    t5 = time.perf_counter()
    lu = splu(csc_matrix(J), permc_spec="COLAMD")
    t6 = time.perf_counter()
    dz_sparse = lu.solve(rhs)
    t7 = time.perf_counter()

    print("\nTiming (at converged iterate)")
    print("Residual eval time   =", (t1 - t0), "sec")
    print("Jacobian build time  =", (t2 - t1), "sec")
    print("Dense solve time     =", (t4 - t3), "sec")
    print("Sparse LU factor     =", (t6 - t5), "sec")
    print("Sparse solve time    =", (t7 - t6), "sec")
    print("||dz_dense - dz_sparse||_inf =", float(np.max(np.abs(dz_dense - dz_sparse))))    

    # Sparsity statistics
    tol_nz = 1e-12

    if issparse(J):
        nnz = int((np.abs(J) > tol_nz).nnz)
        total = int(J.shape[0] * J.shape[1])
    else:
        nnz = int(np.sum(np.abs(J) > tol_nz))
        total = int(J.size)

    density = nnz / total

    print("\nJacobian stats")
    print("Jacobian shape:", J.shape)
    print("Jacobian nnz (>|{:.0e}|):".format(tol_nz), nnz, "/", total)
    print("Jacobian density:", density)

    # Figure: sparsity with current z-order (x-block then p-block)
    fig3 = plt.figure()
    plt.spy(np.abs(J) > tol_nz, markersize=1)
    plt.title("Jacobian pattern (current z-order: x-block then p-block)")
    plt.savefig("example4_jacobian_pattern_current_order.pdf", format="pdf", bbox_inches="tight")


    # ------------------------------------------------------------
    # Column permutation to time-interleaved unknown order:
    #   Original z: [x1..xN, p0..pN]  (x0 is fixed, not in z)
    #   Desired:    [p0, x1, p1, x2, ..., p_{N-1}, xN, pN]
    #
    # IMPORTANT: we permute ONLY columns because the residual rows are
    # already ordered by time (interleaved).
    # ------------------------------------------------------------
    N_steps = result["X"].shape[0] - 1  # N
    n = result["X"].shape[1]           # state dimension (here 1)

    def block(start, n_):
        return list(range(start, start + n_))

    perm_col = []
    for i in range(N_steps):
        # p_i starts at index N*n + i*n in z
        perm_col += block(N_steps * n + i * n, n)
        # x_{i+1} starts at index i*n in z (since x1..xN are first)
        perm_col += block(i * n, n)
    # append p_N
    perm_col += block(N_steps * n + N_steps * n, n)

    perm_col = np.array(perm_col, dtype=int)
    J_col = J[:, perm_col]

    # Bandwidth estimate after column interleaving
    rows, cols = (np.abs(J_col) > tol_nz).nonzero()
    lower_bw = int(np.max(rows - cols))
    upper_bw = int(np.max(cols - rows))
    print("\nEstimated bandwidth after COLUMN interleaving: lower =", lower_bw, ", upper =", upper_bw)

    fig4 = plt.figure()
    plt.spy(np.abs(J_col) > tol_nz, markersize=1)
    plt.title("Jacobian pattern after COLUMN interleaving (time-ordered unknowns)")
    plt.savefig("example4_jacobian_pattern_column_interleaving.pdf", format="pdf", bbox_inches="tight")


    # Show all figures at once
    plt.show()

    # Close figs (optional)
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)

    # ============================================================
    # 4) Local O(N) Jacobian build (n=1) in time-interleaved ordering
    #    w = [p0, x1, p1, x2, ..., p_{N-1}, xN, pN]
    #
    # Idea: each residual row only depends on (at most) three neighboring
    # unknowns in this ordering -> tridiagonal. We approximate ONLY those
    # entries via finite differences, recomputing ONLY the local row.
    # ============================================================
    from core.smoothing import eval_H_smooth

    t_nodes_local = np.asarray(result["t_nodes"])
    X_base = np.asarray(result["X"])
    P_base = np.asarray(result["P"])
    N = t_nodes_local.size - 1
    m = 2 * N + 1  # number of unknowns in time-interleaved ordering

    def residual_row_time_order(k, X_arr, P_arr):
        # Rows are already time-interleaved: (r_x^0, r_p^0, r_x^1, r_p^1, ..., r_bc)
        if k == 2 * N:
            # terminal bc: p_N + grad g(x_N) = 0  (here g=0, but keep generic FD grad)
            xN = X_arr[-1].copy()
            pN = P_arr[-1].copy()
            g_grad = np.zeros_like(pN)
            epsg = 1e-6
            for jj in range(xN.size):
                xp = xN.copy(); xm = xN.copy()
                xp[jj] += epsg
                xm[jj] -= epsg
                g_grad[jj] = (prob.g(xp) - prob.g(xm)) / (2 * epsg)
            return float((pN + g_grad)[0])

        i = k // 2
        dt = float(t_nodes_local[i + 1] - t_nodes_local[i])

        if k % 2 == 0:
            # r_x^i = x_{i+1} - x_i - dt * dH/dp(p_{i+1}, x_i, t_i)
            x_i = X_arr[i].copy()
            x_ip1 = X_arr[i + 1].copy()
            p_ip1 = P_arr[i + 1].copy()
            _, grad_p, _ = eval_H_smooth(prob, result["bundle"], p_ip1, x_i, float(t_nodes_local[i]), delta)
            r = x_ip1 - x_i - dt * grad_p
            return float(r[0])
        else:
            # r_p^i = p_i - p_{i+1} - dt * dH/dx(p_{i+1}, x_i, t_i)
            x_i = X_arr[i].copy()
            p_i = P_arr[i].copy()
            p_ip1 = P_arr[i + 1].copy()
            _, _, grad_x = eval_H_smooth(prob, result["bundle"], p_ip1, x_i, float(t_nodes_local[i]), delta)
            r = p_i - p_ip1 - dt * grad_x
            return float(r[0])


    def perturb_time_unknown(j, X_arr, P_arr, eps):
        # time-interleaved unknown index j:
        # even j -> p_{j/2}, odd j -> x_{(j+1)/2}
        if j % 2 == 0:
            P_arr[j // 2, 0] += eps
        else:
            X_arr[(j + 1) // 2, 0] += eps

    def build_pentadiag_local_fd(eps=1e-7):
        lower2 = np.zeros(m)
        lower1 = np.zeros(m)
        diag   = np.zeros(m)
        upper1 = np.zeros(m)
        upper2 = np.zeros(m)

        for k in range(m):
            for j in (k - 2, k - 1, k, k + 1, k + 2):
                if j < 0 or j >= m:
                    continue

                Xp = X_base.copy(); Pp = P_base.copy()
                Xm = X_base.copy(); Pm = P_base.copy()

                perturb_time_unknown(j, Xp, Pp, +eps)
                perturb_time_unknown(j, Xm, Pm, -eps)

                fp = residual_row_time_order(k, Xp, Pp)
                fm = residual_row_time_order(k, Xm, Pm)
                val = (fp - fm) / (2 * eps)

                if j == k - 2:
                    lower2[k] = val
                elif j == k - 1:
                    lower1[k] = val
                elif j == k:
                    diag[k] = val
                elif j == k + 1:
                    upper1[k] = val
                elif j == k + 2:
                    upper2[k] = val

        return lower2, lower1, diag, upper1, upper2


    lower2, lower1, diag, upper1, upper2 = build_pentadiag_local_fd(eps=1e-7)

    J_penta = (
        np.diag(diag) +
        np.diag(upper1[:-1], 1) + np.diag(lower1[1:], -1) +
        np.diag(upper2[:-2], 2) + np.diag(lower2[2:], -2)
    )

    max_diff = float(np.max(np.abs(J_penta - J_col)))
    print("max|J_penta - J_col| =", max_diff)



    return result


if __name__ == "__main__":
    run_example()
