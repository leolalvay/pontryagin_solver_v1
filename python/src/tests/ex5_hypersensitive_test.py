"""
Example 5: Hypersensitive optimal control.

Minimize   ∫_0^25 (x(t)^2+alpha(t)^2)dt + gamma(x(25)-1)^2
subject to x'(t) = -x(t)^3 + alpha(t),  x(0)=1,  g(x(25))=gamma*(x(25)-1)^2.
"""
import json
from pathlib import Path
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
    x0 = np.array([1.0])   # scalar state, stored as shape (1,)
    T = 25.0
    gamma = 1e6

    def dynamics(x, u, t):
        y = x[0]; a = u[0]
        return np.array([-(y**3) + a])

    def stage_cost(x, u, t):
        y = x[0]; a = u[0]
        return float(y**2 + a**2)

    def terminal_cost(x):
        yT = x[0]
        return float(gamma * ((yT - 1.0)**2))
    
# --- Explicit oracle for Example 3.1 (paper) ---
    def u_star_fn(x, p, t):
        # alpha*(x,p,t) = -p/2  (here p is shape (1,))
        return -0.5 * p

    def hamiltonian_true(x, p, t):
        # H(x,p) = -p*x^3 - p^2/4 + x^2  (scalar case)
        y = x[0]
        lam = p[0]
        return float(-lam * (y**3) - (lam**2) / 4.0 + y**2)
    
    def hamiltonian_grad_fn(x, p, t):
        y = x[0]
        lam = p[0]
        grad_p = np.array([-(y**3) - 0.5 * lam])         # dH/dp
        grad_x = np.array([-3.0 * lam * (y**2) + 2.0*y]) # dH/dx
        return grad_p, grad_x


    u_min = np.array([-1.0])
    u_max = np.array([3.0])

   
    prob = OCPProblem(
        dynamics=dynamics,
        stage_cost=stage_cost,
        terminal_cost=terminal_cost,
        x0=x0,
        T=T,
        control_bounds=(u_min, u_max),
        state_bounds=None,
        hamiltonian_true=hamiltonian_true,
        u_star_fn=u_star_fn,
        hamiltonian_grad_fn=hamiltonian_grad_fn,
    )

    # -------------------------
    # Sanity check: explicit u* and H_true are consistent
    # -------------------------
    x_test = np.array([1.0])
    p_test = np.array([2.0])
    t_test = 0.0

    u_test, ok_u = prob.u_star(x_test, p_test, t_test, restricted=False)
    H_method, u_used, ok_H = prob.hamiltonian_true(x_test, p_test, t_test, restricted=False)

    H_closed = hamiltonian_true(x_test, p_test, t_test)
    H_pf_l = float(p_test @ dynamics(x_test, u_test, t_test) + stage_cost(x_test, u_test, t_test))

    DO_SANITY = False
    if DO_SANITY:
        print(f"[sanity] u_star={u_test}, ok_u={ok_u}")
        print(f"[sanity] H_method={H_method:.8e}, H_closed={H_closed:.8e}, H_pf+l={H_pf_l:.8e}, ok_H={ok_H}")
        return


    use_oracle_bootstrap = False   # o True
    use_oracle_PA = True
    use_explicit_hamiltonian_gradients = True
    # ============================================================
    # 1) Solve with the repo's adaptive outer loop
    # ============================================================
    t_nodes = np.linspace(0.0, T, 30)  # dt = 0.02 (fast initial mesh)

    t0 = time.perf_counter()
    result = solve_optimal_control(
        prob,
        t_nodes,
        tol_time=1e-2,   # relaxed -> fewer refinements
        tol_PA=1e-2,
        tol_delta=1e-2,
        max_iters=25,
        delta0=0.02,
        use_oracle_bootstrap=use_oracle_bootstrap,
        use_oracle_PA=use_oracle_PA,
        use_explicit_hamiltonian_gradients=use_explicit_hamiltonian_gradients,
    )
    t1 = time.perf_counter()
    wall_time = t1 - t0
    print(f"[benchmark] wall_time_sec = {wall_time:.3f}")



    print("\nExample 5 (Hypersensitive optimal control)")
    print("len(log) =", len(result["log"]))
    print("last outer iter =", result["log"][-1]["iteration"])
    print("len(t_nodes) =", len(result["t_nodes"]))
    print("X.shape =", result["X"].shape)
    print("P.shape =", result["P"].shape)
    print("final delta =", result["delta"])
    print("last log entry =", result["log"][-1])

    # -------------------------
    # Save benchmark summary (overwritten each run)
    # -------------------------
    Path("benchmarks").mkdir(parents=True, exist_ok=True)

    last = result["log"][-1]
    bench = {
        "example": "ex5_hypersensitive",
        "wall_time_sec": float(wall_time),
        "max_iters": int(last["iteration"]),
        "N_final": int(last["N"]),
        "M_final": int(last["M"]),
        "delta_final": float(result["delta"]),
        "eta_time": float(last["eta_time"]),
        "eta_PA": float(last["eta_PA"]),
        "eta_delta": float(last["eta_delta"]),
        "len_t_nodes": int(len(result["t_nodes"])),
    }

    # these two variables should match what you pass into solve_optimal_control(...)
    # e.g. define them near the solve call: use_oracle_bootstrap = True/False, use_oracle_PA = True/False
    bench["use_oracle_bootstrap"] = bool(use_oracle_bootstrap)
    bench["use_oracle_PA"] = bool(use_oracle_PA)

    tag_parts = []
    if use_oracle_bootstrap:
        tag_parts.append("oracleBoot")
    if use_oracle_PA:
        tag_parts.append("oraclePA")
    tag = "_".join(tag_parts) if tag_parts else "baseline"

    out_path = f"benchmarks/ex5_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(bench, f, indent=2)

    print(f"[benchmark] wrote {out_path}")
    print(f"[benchmark-debug] use_oracle_bootstrap={use_oracle_bootstrap}, use_oracle_PA={use_oracle_PA}, out_path={out_path}")


    # ============================================================
    #    
    #     
    #      
    # ============================================================
    t = np.asarray(result["t_nodes"])
    dt = np.diff(t)

    fig_dir = fig_dir = Path(__file__).resolve().parent / "figures"

    fig1 = plt.figure()
    #plt.figure()
    plt.step(t[:-1], dt, where="post")   # Δt constant in [t_n, t_{n+1})
    plt.yscale("log")                   #log scale in y
    plt.xlabel("t")
    plt.ylabel("Δt")
    plt.title("Time mesh: Δt(t) (step plot)")
    plt.grid(True, which="both")
    plt.savefig(fig_dir /"example5_test_tvsdt.pdf", format="pdf", bbox_inches="tight")
    #plt.show()

    X = np.asarray(result["X"])[:, 0]
    P = np.asarray(result["P"])[:, 0]
    x0_scalar = float(x0[0])


    # Plot X and P (two windows at once)
    fig1 = plt.figure()
    plt.plot(t, X, label="X (solver)")
    plt.xlabel("t")
    plt.ylabel("X")
    plt.title("Example 5: State X(t)")
    plt.legend()
    plt.grid(True)
    plt.savefig(fig_dir /"example5_test_state_X.pdf", format="pdf", bbox_inches="tight")


    fig2 = plt.figure()
    plt.plot(t, P, label="P (solver)")
    plt.xlabel("t")
    plt.ylabel("P")
    plt.title("Example 5: Costate P(t)")
    plt.legend()
    plt.grid(True)
    plt.savefig(fig_dir /"example5_test_costate_P.pdf", format="pdf", bbox_inches="tight")

    # --- arrays ---
    rho_bar = np.asarray(result["rhobar"])   # length N
    r_bar   = np.asarray(result["rbar"])     # length N
    t_mid   = 0.5*(t[:-1] + t[1:])           # length N

    # Plot rho_bar
    fig3 = plt.figure()
    plt.step(t[:-1], np.abs(rho_bar), where="post", label=r"$|\bar{\rho}_n|$")
    plt.yscale("log")
    plt.xlabel("t")
    plt.ylabel(r"$|\bar{\rho}|$")
    plt.title(r"Example 5: density-like term $\bar{\rho}_n$")   # <-- TEXTO PRIMERO
    plt.grid(True)
    plt.legend()
    plt.savefig(fig_dir /"example5_test_rho_bar.pdf", format="pdf", bbox_inches="tight")


    # Plot r_bar
    fig4 = plt.figure()
    plt.step(t[:-1], r_bar, where="post", label=r"$\bar{r}_n$")
    plt.yscale("log")
    plt.xlabel("t")
    plt.ylabel(r"$\bar{r}$")
    plt.title(r"Example 5: time error indicator $\bar{r}_n = |\bar{\rho}_n|\,\Delta t_n^2$")  # <-- TEXTO PRIMERO
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig(fig_dir /"example5_test_r_bar.pdf", format="pdf", bbox_inches="tight")
   

    # Show all figures at once
    #plt.show()

    # Close figs (optional)
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)
    

    return result


if __name__ == "__main__":
    run_example()
