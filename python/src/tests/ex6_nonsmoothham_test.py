import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from pathlib import Path
from functools import partial

def init_ex6():
    T = 1.0
    x0 = 0.5
    delta = 1e-10
    N0 = 20
    tol_time = 1e-6
    max_refine = 20

    # adaptivity params (article style)
    s_mark = 0.25
    M_sub = 2
    K_time = 1e-6

    if T <= 0:
        raise ValueError("T must be > 0")
    if delta <= 0:
        raise ValueError("delta must be > 0")
    if N0 < 1:
        raise ValueError("N0 must be >= 1")

    t = np.linspace(0.0, T, N0 + 1)

    # Model
    def f(x, a, t_):
        return a

    def L(x, a, t_):
        return x**10

    def g(xT):
        return 0.0

    def H_delta(x, lam):
        return x**10 - np.sqrt(lam**2 + delta**2)

    def dH_dlam(x, lam):
        return -lam / np.sqrt(lam**2 + delta**2)

    def dH_dx(x, lam):
        return 10.0 * x**9

    # Exact/reference
    def x_star(tt):
        return np.maximum(x0 - tt, 0.0)

    def a_star(tt):
        return np.where(tt < x0, -1.0, 0.0)

    def p_star(tt):
        return np.where(tt <= x0, (x0 - tt)**10, 0.0)

    params = {
        "T": T,
        "x0": x0,
        "delta": delta,
        "N0": N0,
        "tol_time": tol_time,
        "max_refine": max_refine,
        "s_mark": s_mark,
        "M_sub": M_sub,
        "K_time": K_time,
        "t": t,
    }

    model = {
        "f": f,
        "L": L,
        "g": g,
        "H_delta": H_delta,
        "dH_dlam": dH_dlam,
        "dH_dx": dH_dx,
    }

    ref = {
        "x_star": x_star,
        "a_star": a_star,
        "p_star": p_star,
        "J_star": x0**11 / 11.0,
    }

    return params, model, ref


def pack_z(x, lam):
    # z = [x1..xN, lam0..lamN]
    return np.concatenate([x[1:], lam])


def unpack_z(z, x0):
    total = z.size  # = 2N+1
    N = (total - 1) // 2

    x = np.empty(N + 1)
    lam = np.empty(N + 1)

    x[0] = x0
    x[1:] = z[:N]
    lam[:] = z[N:]
    return x, lam


def residual_symplectic_euler(z, params, model):
    t = params["t"]
    x0 = params["x0"]

    x, lam = unpack_z(z, x0)

    N = len(t) - 1
    dt = np.diff(t)

    r = np.zeros(2 * N + 1, dtype=float)

    dH_dlam = model["dH_dlam"]
    dH_dx = model["dH_dx"]

    for i in range(N):
        gp = dH_dlam(x[i], lam[i + 1])  # ∂H/∂λ at (x_i, λ_{i+1})
        gx = dH_dx(x[i], lam[i + 1])    # ∂H/∂x at (x_i, λ_{i+1})

        r[2 * i] = x[i + 1] - x[i] - dt[i] * gp
        r[2 * i + 1] = lam[i] - lam[i + 1] - dt[i] * gx

    # terminal BC: lam_N + g_x(x_N) = 0, and g=0 => lam_N=0
    r[-1] = lam[-1]
    return r


def jacobian_symplectic_euler(z, params, model):
    t = params["t"]
    x0 = params["x0"]
    delta = params["delta"]

    x, lam = unpack_z(z, x0)
    N = len(t) - 1
    dt = np.diff(t)

    m = 2 * N + 1
    J = np.zeros((m, m), dtype=float)

    def ix(k):   # x_k, k=1..N
        return k - 1

    def il(j):   # lam_j, j=0..N
        return N + j

    for i in range(N):
        rx = 2 * i
        rl = 2 * i + 1

        xi = x[i]
        lip1 = lam[i + 1]
        den = lip1 * lip1 + delta * delta

        # second derivatives
        d2H_dlam2 = -(delta * delta) / (den ** 1.5)
        d2H_dx2 = 90.0 * (xi ** 8)

        # r_x^i
        if i >= 1:
            J[rx, ix(i)] += -1.0
        J[rx, ix(i + 1)] += +1.0
        J[rx, il(i + 1)] += -dt[i] * d2H_dlam2

        # r_lam^i
        if i >= 1:
            J[rl, ix(i)] += -dt[i] * d2H_dx2
        J[rl, il(i)] += +1.0
        J[rl, il(i + 1)] += -1.0

    # r_bc = lam_N
    J[-1, il(N)] = 1.0
    return J

def initial_guess(params):
    t = params["t"]
    x0 = params["x0"]
    N = len(t) - 1

    # state guess: move left with slope -1 until hitting zero
    x_guess = np.maximum(x0 - t, 0.0)

    # costate guess: backward Euler from lambda(T)=0 using lambda' = 10 x^9
    lam_guess = np.zeros(N + 1)
    for i in range(N - 1, -1, -1):
        dt = t[i + 1] - t[i]
        lam_guess[i] = lam_guess[i + 1] + dt * 10.0 * x_guess[i]**9

    return pack_z(x_guess, lam_guess)

'''def initial_guess(params):
    t = params["t"]
    x0 = params["x0"]
    N = len(t) - 1

    x_guess = np.linspace(x0, 0.0, N + 1)
    lam_guess = np.zeros(N + 1)

    return pack_z(x_guess, lam_guess)'''

def initial_guess_from_reference(params, ref):
    t = params["t"]
    x_guess = np.asarray(ref["x_star"](t), dtype=float)
    lam_guess = np.asarray(ref["p_star"](t), dtype=float)
    return pack_z(x_guess, lam_guess)


def solve_on_mesh(params, model, z0=None, tol=1e-10, maxfev=20000):
    if z0 is None:
        z0 = initial_guess(params)

    sol = root(
        fun=residual_symplectic_euler,
        x0=z0,
        args=(params, model),
        jac=jacobian_symplectic_euler,
        method="hybr",
        tol=tol,
        options={"maxfev": maxfev},
    )

    res_inf = float(np.linalg.norm(sol.fun, ord=np.inf))
    accept_res = 1e-10
    success_flag = bool(sol.success) or (res_inf <= accept_res)

    if sol.success:
        message = sol.message
    elif res_inf <= accept_res:
        message = f"Accepted by residual criterion: res_inf={res_inf:.3e}"
    else:
        message = sol.message

    x, lam = unpack_z(sol.x, params["x0"])
    t = params["t"]
    N = len(t) - 1
    a = np.array([model["dH_dlam"](x[i], lam[i + 1]) for i in range(N)])
    J_cost = float(np.sum(np.diff(t) * (x[:-1] ** 10)))

    return {
        "success": bool(success_flag),
        "message": message,
        "nfev": int(sol.nfev),
        "njev": int(getattr(sol, "njev", -1)),
        "res_inf": res_inf,
        "x": x,
        "lam": lam,
        "a": a,
        "J": J_cost,
        "z": sol.x,
        "solver_obj": sol,
    }


def compute_time_indicator(params, model, x, lam, K_time=1.0):
    t = np.asarray(params["t"], dtype=float)
    x = np.asarray(x, dtype=float)
    lam = np.asarray(lam, dtype=float)

    N = t.size - 1
    if N <= 0:
        return {
            "dt": np.array([]),
            "rho": np.array([]),
            "rho_bar": np.array([]),
            "r_bar": np.array([]),
            "eta_time_max": 0.0,
            "eta_time_sum": 0.0,
            "tol_star": float(params["tol_time"]),
            "mark_thr": 0.0,
            "floor": 0.0,
        }

    dt = np.diff(t)
    dt_max = float(np.max(dt))
    floor = float(K_time * np.sqrt(dt_max))

    rho = np.zeros(N, dtype=float)
    rho_bar = np.zeros(N, dtype=float)
    r_bar = np.zeros(N, dtype=float)

    dH_dlam = model["dH_dlam"]
    dH_dx = model["dH_dx"]

    for i in range(N):
        Hp = float(dH_dlam(x[i], lam[i + 1]))
        Hx = float(dH_dx(x[i], lam[i + 1]))
        rho_i = -0.5 * Hp * Hx
        rho[i] = rho_i

        mag = max(abs(rho_i), floor)
        rho_bar_i = mag
        #rho_bar_i = np.sign(rho_i) * mag
        rho_bar[i] = rho_bar_i

        r_bar[i] = abs(rho_bar_i) * (dt[i] ** 2)

    tol_star = float(params["tol_time"] / N)
    mark_thr = float(params["s_mark"] * params["tol_time"] / N)

    return {
        "dt": dt,
        "rho": rho,
        "rho_bar": rho_bar,
        "r_bar": r_bar,
        "eta_time_max": float(np.max(r_bar)),
        "eta_time_sum": float(np.sum(r_bar)),
        "tol_star": tol_star,
        "mark_thr": mark_thr,
        "floor": floor,
    }


def refine_mesh_article(t_old, r_bar, tol_time, s_mark=0.8, M_sub=2):
    t_old = np.asarray(t_old, dtype=float)
    N = len(t_old) - 1
    if N <= 0:
        return t_old.copy(), np.array([], dtype=bool)

    thr = float(s_mark * tol_time / N)
    marked = np.asarray(r_bar) > thr

    new_nodes = [t_old[0]]
    for i in range(N):
        a, b = t_old[i], t_old[i + 1]
        if marked[i]:
            mids = np.linspace(a, b, M_sub + 1)[1:]  # exclude left endpoint
            new_nodes.extend(mids.tolist())
        else:
            new_nodes.append(b)

    t_new = np.asarray(new_nodes, dtype=float)
    return t_new, marked


def prolongate_guess(t_old, x_old, lam_old, t_new):
    x_new = np.interp(t_new, t_old, x_old)
    lam_new = np.interp(t_new, t_old, lam_old)
    return x_new, lam_new


def run_adaptivity_test(params, model, ref=None, verbose=True, maxit=None, z0_init=None):
    tol_time = float(params["tol_time"])
    s_mark = float(params["s_mark"])
    M_sub = int(params["M_sub"])
    K_time = float(params["K_time"])

    if maxit is None:
        maxit = int(params.get("max_refine", 20))
    else:
        maxit = int(maxit)

    t = np.asarray(params["t"], dtype=float).copy()
    z0 = z0_init
    log = []

    iters_used = 0
    pending_update = False
    converged = False
    stop_reason = "unknown"

    while True:
        if (iters_used >= maxit) and (not pending_update):
            stop_reason = "maxit_reached"
            break

        p = dict(params)
        p["t"] = t

        sol = solve_on_mesh(p, model, z0=z0)
        if not sol["success"]:
            stop_reason = "nonlinear_solve_failed"
            log.append({
                "iter": iters_used,
                "N": len(t) - 1,
                "success": False,
                "message": sol["message"],
                "res_inf": sol["res_inf"],
            })
            break

        x = sol["x"]
        lam = sol["lam"]

        ind = compute_time_indicator(p, model, x, lam, K_time=K_time)

        N = len(t) - 1
        eta_time = float(ind["eta_time_max"])
        tol_star = float(ind["tol_star"])
        mark_thr = float(ind["mark_thr"])
        n_marked = int(np.sum(ind["r_bar"] > mark_thr)) if N > 0 else 0

        entry = {
            "iter": iters_used,
            "N": N,
            "success": True,
            "res_inf": float(sol["res_inf"]),
            "nfev": int(sol["nfev"]),
            "njev": int(sol["njev"]),
            "J": float(sol["J"]),
            "eta_time_max": eta_time,
            "eta_time_sum": float(ind["eta_time_sum"]),
            "tol_star": tol_star,
            "mark_thr": mark_thr,
            "n_marked": n_marked,
            "dt_min": float(np.min(ind["dt"])) if N > 0 else 0.0,
            "dt_max": float(np.max(ind["dt"])) if N > 0 else 0.0,
            "floor": float(ind["floor"]),
            "rho": ind["rho"].copy(),
            "rho_bar": ind["rho_bar"].copy(),
            "r_bar": ind["r_bar"].copy(),
        }

        if ref is not None:
            x_star = ref["x_star"](t)
            p_star = ref["p_star"](t)
            entry["err_x_inf"] = float(np.max(np.abs(x - x_star)))
            entry["err_p_inf"] = float(np.max(np.abs(lam - p_star)))
            entry["err_J_abs"] = float(abs(sol["J"] - ref["J_star"]))

        log.append(entry)

        if verbose:
            print(
                f"[adapt {iters_used:02d}] N={N:4d} "
                f"res={sol['res_inf']:.2e} "
                f"eta={eta_time:.2e}/{tol_star:.2e} "
                f"marked={n_marked}"
            )

        iters_used += 1
        pending_update = False

        if eta_time < tol_star:
            converged = True
            stop_reason = "time_tolerance_reached"
            z0 = sol["z"]
            break

        t_new, marked = refine_mesh_article(
            t_old=t,
            r_bar=ind["r_bar"],
            tol_time=tol_time,
            s_mark=s_mark,
            M_sub=M_sub,
        )

        if not np.any(marked):
            stop_reason = "no_intervals_marked"
            z0 = sol["z"]
            break

        xg, lg = prolongate_guess(t, x, lam, t_new)
        z0 = pack_z(xg, lg)

        t = t_new
        pending_update = True

    result = {
        "converged": converged,
        "stop_reason": stop_reason,
        "iterations": iters_used,
        "maxit": maxit,
        "t": t,
        "log": log,
    }

    # ensure final output corresponds to returned mesh
    p_final = dict(params)
    p_final["t"] = t
    sol_final = solve_on_mesh(p_final, model, z0=z0)

    result.update({
        "success": bool(sol_final["success"]),
        "message": sol_final["message"],
        "res_inf": float(sol_final["res_inf"]),
        "x": sol_final["x"],
        "lam": sol_final["lam"],
        "a": sol_final["a"],
        "J": float(sol_final["J"]),
        "z": sol_final["z"],
    })

    return result

def save_plot(fig, stem, fig_dir, ext="pdf"):
    """
    Save one figure to disk and close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to save.
    stem : str
        Base filename without extension.
    fig_dir : pathlib.Path
        Directory where figures are stored.
    ext : str, default="pdf"
        File extension / output format.
    """
    fig.savefig(fig_dir / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)

def keep_plot(fig, stem=None):
    """
    Do nothing to the figure.

    This is used in interactive mode:
    figures remain open, and at the end we call plt.show()
    once to display all of them together.
    """
    pass

#Compact summary fucntions
def summarize_array(name, arr):
    arr = np.asarray(arr, dtype=float)
    print(
        f"{name:10s}: len={len(arr):4d}  "
        f"min={arr.min():.3e}  max={arr.max():.3e}  "
        f"first={arr[0]:.3e}  last={arr[-1]:.3e}"
    )

def print_log_summary(log_entry):
    print("\n=== LAST LOG SUMMARY ===")
    for key in [
        "iter", "N", "success", "res_inf", "nfev", "njev", "J",
        "eta_time_max", "eta_time_sum", "tol_star", "mark_thr",
        "n_marked", "dt_min", "dt_max", "floor",
        "err_x_inf", "err_p_inf", "err_J_abs"
    ]:
        if key in log_entry:
            print(f"{key:12s}: {log_entry[key]}")

    for key in ["rho", "rho_bar", "r_bar"]:
        if key in log_entry:
            summarize_array(key, log_entry[key])



def plot_ex6_results(result, ref, out_prefix="ex6_test", save_plots=False, plot_ext="pdf", fig_dir=None):
    if ("log" not in result) or (len(result["log"]) == 0):
        print("[plot] No iteration log available; skipping plots.")
        return
    
    if fig_dir is None:
        fig_dir = Path(__file__).resolve().parent / "figures"
    else:
        fig_dir = Path(fig_dir)

    plot_action = partial(save_plot, fig_dir=fig_dir, ext=plot_ext) if save_plots else keep_plot
    render_plots = (lambda: None) if save_plots else plt.show

    t = np.asarray(result["t"], dtype=float)
    x = np.asarray(result["x"], dtype=float)
    p = np.asarray(result["lam"], dtype=float)
    a = np.asarray(result["a"], dtype=float)

    dt = np.diff(t)
    t_int = t[:-1]

    x_star = ref["x_star"](t)
    p_star = ref["p_star"](t)
    a_star = ref["a_star"](t_int)

    last = result["log"][-1]
    rho = np.asarray(last["rho"], dtype=float)
    rho_bar = np.asarray(last["rho_bar"], dtype=float)
    r_bar = np.asarray(last["r_bar"], dtype=float)

    fig = plt.figure(figsize=(6, 4))
    plt.step(t_int, dt, where="post", label=r"$\Delta t_n$")
    plt.yscale("log")
    plt.xlabel("t")
    plt.ylabel(r"$\Delta t$")
    plt.title(r"Time mesh: $\Delta t(t)$")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plot_action(fig, f"{out_prefix}_t_vs_dt")

    fig = plt.figure(figsize=(6, 4))
    plt.plot(t, x, label="x (computed)", linewidth=2.0)
    plt.plot(t, x_star, "--", label="x* (exact)", linewidth=2.0)
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("State trajectory")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_action(fig, f"{out_prefix}_state_x")

    fig = plt.figure(figsize=(6, 4))
    plt.plot(t, p, label="p (computed)", linewidth=2.0)
    plt.plot(t, p_star, "--", label="p* (exact)", linewidth=2.0)
    plt.xlabel("t")
    plt.ylabel("p")
    plt.title("Costate trajectory")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_action(fig, f"{out_prefix}_costate_p")

    fig = plt.figure(figsize=(6, 4))
    plt.step(t_int, a, where="post", label="a (computed)", linewidth=2.0)
    plt.step(t_int, a_star, where="post", linestyle="--", label="a* (exact)", linewidth=2.0)
    plt.xlabel("t")
    plt.ylabel("a")
    plt.title("Control trajectory")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_action(fig, f"{out_prefix}_control_a")

    fig = plt.figure(figsize=(6, 4))
    plt.step(t_int, rho, where="post", label=r"$\rho_n$")
    plt.step(t_int, rho_bar, where="post", label=r"$\bar{\rho}_n$")
    plt.xlabel("t")
    plt.ylabel(r"$\rho$")
    plt.title(r"Error density: $\rho_n,\ \bar{\rho}_n$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_action(fig, f"{out_prefix}_rho_density")

    fig = plt.figure(figsize=(6, 4))
    plt.step(t_int, r_bar, where="post", label=r"$\bar r_n = |\bar\rho_n|\Delta t_n^2$")
    plt.yscale("log")
    plt.xlabel("t")
    plt.ylabel(r"$\bar r_n$")
    plt.title(r"Time error indicator $\bar r_n$")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plot_action(fig, f"{out_prefix}_r_indicator")

    render_plots()

if __name__ == "__main__":
    params, model, ref = init_ex6()

    z0_init = initial_guess_from_reference(params, ref)
    result = run_adaptivity_test(
        params,
        model,
        ref=ref,
        verbose=True,
        maxit=20,
        z0_init=z0_init,
    )
    plot_ex6_results(result, ref, out_prefix="example6_test")
    print("\n=== FINAL SUMMARY ===")
    print("converged   :", result["converged"])
    print("stop_reason :", result["stop_reason"])
    print("success     :", result["success"])
    print("message     :", result["message"])
    print("res_inf     :", result["res_inf"])
    print_log_summary(result["log"][-1])
    '''print("\n=== FINAL SUMMARY ===")
    print("converged   :", result["converged"])
    print("stop_reason :", result["stop_reason"])
    print("success     :", result["success"])
    print("message     :", result["message"])
    print("res_inf     :", result["res_inf"])
    print("N_final     :", len(result["t"]) - 1)'''
    '''result = run_adaptivity_test(params, model, ref=ref, verbose=True, maxit=20)

    if not result["success"]:
        print("\n[ERROR] Final solve failed:", result["message"])
    else:
        plot_ex6_results(result, ref, out_prefix="example6_test")

        t = result["t"]
        x = result["x"]
        p = result["lam"]
        a = result["a"]

        x_star = ref["x_star"](t)
        p_star = ref["p_star"](t)
        a_star = ref["a_star"](t[:-1])

        print("\n=== FINAL SUMMARY ===")
        print("converged   :", result["converged"], "| reason:", result["stop_reason"])
        print("iters_used  :", result["iterations"], "/", result["maxit"])
        print("N_final     :", len(t) - 1)
        print("res_inf     :", result["res_inf"])
        print("J_hat       :", result["J"])
        print("J_star      :", ref["J_star"])
        print("||x-x*||inf :", np.max(np.abs(x - x_star)))
        print("||p-p*||inf :", np.max(np.abs(p - p_star)))
        print("||a-a*||inf :", np.max(np.abs(a - a_star)))'''