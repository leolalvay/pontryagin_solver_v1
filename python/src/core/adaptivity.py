import numpy as np
import atexit
from pathlib import Path
from .pa_bundle import PABundle
from .smoothing import eval_H_smooth
from .hamiltonian import compute_H
from .newton import solve_tpbvp

def _local_dt_at_node(t_nodes: np.ndarray, idx: int):
    t_nodes = np.asarray(t_nodes, dtype=float)
    if t_nodes.size <= 1:
        return None
    if idx < t_nodes.size - 1:
        return float(t_nodes[idx + 1] - t_nodes[idx])
    return float(t_nodes[-1] - t_nodes[-2])


def choose_adaptive_action(
    eta_time,
    tol_time_star,
    eta_PA,
    tol_PA,
    eta_delta,
    tol_delta,
    n_marked,
    *,
    explicit_mode=False,
    time_balance_ratio=0.1,
):
    if explicit_mode:
        return "STOP" if eta_time <= tol_time_star else f"refine_time(marked={n_marked})"

    if eta_time <= tol_time_star and eta_PA <= tol_PA and eta_delta <= tol_delta:
        return "STOP"

    max_non_time = max(float(eta_PA), float(eta_delta))
    time_refinement_allowed = True
    if max_non_time > 0.0 and float(eta_time) <= float(time_balance_ratio) * max_non_time:
        time_refinement_allowed = False

    if eta_time > tol_time_star and time_refinement_allowed:
        return f"refine_time(marked={n_marked})"
    if not time_refinement_allowed:
        if float(eta_delta) >= float(eta_PA) and float(eta_delta) > 0.0:
            return "delta*=0.5"
        if float(eta_PA) > 0.0:
            return "add_plane"
    if eta_PA > tol_PA:
        return "add_plane"
    if eta_delta > tol_delta:
        return "delta*=0.5"
    if eta_time > tol_time_star:
        return f"refine_time(marked={n_marked})"
    return "continue"


def validate_indicator_tolerances(
    tol_time: float,
    tol_PA: float,
    tol_delta: float,
    *,
    max_relative_factor: float = 2.0,
):
    tol_time = float(tol_time)
    tol_PA = float(tol_PA)
    tol_delta = float(tol_delta)
    max_relative_factor = float(max_relative_factor)

    if tol_time <= 0.0 or tol_PA <= 0.0 or tol_delta <= 0.0:
        raise ValueError("All indicator tolerances must be strictly positive.")
    if max_relative_factor < 1.0:
        raise ValueError("max_relative_factor must be at least 1.0.")
    if tol_PA > max_relative_factor * tol_time:
        raise ValueError(
            f"tol_PA={tol_PA:g} exceeds {max_relative_factor:g} * tol_time={tol_time:g}; "
            "keep PA and time tolerances comparable."
        )
    if tol_delta > max_relative_factor * tol_time:
        raise ValueError(
            f"tol_delta={tol_delta:g} exceeds {max_relative_factor:g} * tol_time={tol_time:g}; "
            "keep delta and time tolerances comparable."
        )

def _grads_for_indicators(problem, bundle, p, x, t, delta, dt=None, use_explicit_hamiltonian_gradients=False):
    if use_explicit_hamiltonian_gradients and problem.hamiltonian_grad_fn is not None:
        Hp, Hx = problem.hamiltonian_gradients(x, p, t)
        return None, Hp, Hx
    return eval_H_smooth(problem, bundle, p, x, t, delta, dt=dt)


def _compute_node_controls(problem, bundle, X, P, t_nodes, restricted=True, use_oracle=False):
    controls = []
    for i, t_i in enumerate(t_nodes):
        dt_i = _local_dt_at_node(t_nodes, i)
        _, u_star = compute_H(
            problem,
            P[i],
            X[i],
            t_i,
            bundle.controls,
            restricted=restricted,
            use_oracle=use_oracle,
            dt=dt_i,
        )
        if u_star is None:
            raise RuntimeError(f"No admissible control found at node {i} and time {t_i}.")
        controls.append(np.asarray(u_star, dtype=float).reshape(-1))
    return np.vstack(controls) if controls else np.zeros((0, problem.m or 0))


def _supports_need_dt_refresh(problem) -> bool:
    return getattr(problem, "step_feasible_control_fn", None) is not None or getattr(problem, "u_star_local_fn", None) is not None


def _refresh_bundle_support_controls(problem, t_nodes, X_guess, P_guess, support_points, use_oracle=False):
    if not _supports_need_dt_refresh(problem):
        return None
    refreshed_bundle = PABundle()
    bounds = problem.control_bounds_tuple()
    m = problem.m
    if m is None and bounds is not None:
        m = bounds[0].size
    if m is not None:
        if bounds is not None:
            u_min, u_max = bounds
            refreshed_bundle.add_control(0.5 * (u_min + u_max))
            refreshed_bundle.add_control(u_min)
            refreshed_bundle.add_control(u_max)
        else:
            refreshed_bundle.add_control(np.zeros(m))

    if X_guess is None or P_guess is None:
        return refreshed_bundle

    t_nodes = np.asarray(t_nodes, dtype=float)
    X_guess = np.asarray(X_guess, dtype=float)
    P_guess = np.asarray(P_guess, dtype=float)
    for point in support_points:
        target_time = float(point.get("time"))
        idx = int(np.argmin(np.abs(t_nodes - target_time)))
        dt_i = _local_dt_at_node(t_nodes, idx)
        _, u_star = compute_H(
            problem,
            P_guess[idx],
            X_guess[idx],
            float(t_nodes[idx]),
            refreshed_bundle.controls,
            restricted=True,
            use_oracle=use_oracle,
            dt=dt_i,
        )
        if u_star is None:
            continue
        refreshed_bundle.add_control(u_star)
        point["control"] = np.asarray(u_star, dtype=float).copy()
        point["time"] = float(t_nodes[idx])
        point["node_index"] = int(idx)
        point["state"] = np.asarray(X_guess[idx], dtype=float).copy()
        point["costate"] = np.asarray(P_guess[idx], dtype=float).copy()
        point["local_dt"] = float(dt_i) if dt_i is not None else None
        point["bundle_size_after"] = int(refreshed_bundle.num_planes())
    return refreshed_bundle


def _mesh_objective(problem, t_nodes, X, controls):
    obj = float(problem.g(X[-1]))
    for i in range(len(t_nodes) - 1):
        dt = float(t_nodes[i + 1] - t_nodes[i])
        obj += float(problem.l(X[i], controls[i], t_nodes[i])) * dt
    return obj

def _local_control_scale(problem) -> float:
    bounds = problem.control_bounds_tuple()
    if bounds is None:
        m = problem.m if problem.m is not None else 1
        return float(max(np.sqrt(float(m)), 1.0))
    u_min, u_max = bounds
    scale = float(np.linalg.norm(np.asarray(u_max, dtype=float) - np.asarray(u_min, dtype=float)))
    return scale if scale > 0.0 else 1.0


def _find_feasibility_refinement_intervals(
    problem,
    bundle,
    X,
    P,
    t_nodes,
    use_oracle=False,
    *,
    probe_ratio: float = 0.5,
    control_sensitivity: float = 0.05,
):
    marked = set()
    issues = []
    control_scale = _local_control_scale(problem)
    for i, t_i in enumerate(t_nodes):
        dt_i = _local_dt_at_node(t_nodes, i)
        _, u_star = compute_H(
            problem,
            P[i],
            X[i],
            float(t_i),
            bundle.controls,
            restricted=True,
            use_oracle=use_oracle,
            dt=dt_i,
        )
        if u_star is None:
            interval_idx = min(i, max(len(t_nodes) - 2, 0))
            marked.add(int(interval_idx))
            issues.append(
                {
                    "node_index": int(i),
                    "interval_index": int(interval_idx),
                    "time": float(t_i),
                    "local_dt": float(dt_i) if dt_i is not None else None,
                    "state": np.asarray(X[i], dtype=float).copy(),
                    "costate": np.asarray(P[i], dtype=float).copy(),
                    "reason": "empty_admissible_set",
                }
            )
            continue

        if getattr(problem, "feasibility_refinement_fn", None) is not None and dt_i is not None and dt_i > 0.0:
            issue = problem.feasibility_refinement_fn(
                np.asarray(X[i], dtype=float),
                np.asarray(P[i], dtype=float),
                float(t_i),
                float(dt_i),
                1e-8,
            )
            if issue is not None:
                interval_idx = min(i, max(len(t_nodes) - 2, 0))
                marked.add(int(interval_idx))
                issue_record = {
                    "node_index": int(i),
                    "interval_index": int(interval_idx),
                    "time": float(t_i),
                    "local_dt": float(dt_i),
                    "state": np.asarray(X[i], dtype=float).copy(),
                    "costate": np.asarray(P[i], dtype=float).copy(),
                    "reason": "problem_feasibility_refinement",
                }
                if isinstance(issue, dict):
                    issue_record.update(issue)
                issues.append(issue_record)
                continue

        if (
            dt_i is None
            or dt_i <= 0.0
            or probe_ratio <= 0.0
            or probe_ratio >= 1.0
            or control_sensitivity <= 0.0
        ):
            continue

        dt_probe = float(probe_ratio) * float(dt_i)
        _, u_probe = compute_H(
            problem,
            P[i],
            X[i],
            float(t_i),
            bundle.controls,
            restricted=True,
            use_oracle=use_oracle,
            dt=dt_probe,
        )
        if u_probe is None:
            interval_idx = min(i, max(len(t_nodes) - 2, 0))
            marked.add(int(interval_idx))
            issues.append(
                {
                    "node_index": int(i),
                    "interval_index": int(interval_idx),
                    "time": float(t_i),
                    "local_dt": float(dt_i),
                    "probe_dt": float(dt_probe),
                    "state": np.asarray(X[i], dtype=float).copy(),
                    "costate": np.asarray(P[i], dtype=float).copy(),
                    "reason": "empty_probe_admissible_set",
                }
            )
            continue

        rel_change = float(
            np.linalg.norm(np.asarray(u_star, dtype=float) - np.asarray(u_probe, dtype=float)) / control_scale
        )
        if rel_change > float(control_sensitivity):
            interval_idx = min(i, max(len(t_nodes) - 2, 0))
            marked.add(int(interval_idx))
            issues.append(
                {
                    "node_index": int(i),
                    "interval_index": int(interval_idx),
                    "time": float(t_i),
                    "local_dt": float(dt_i),
                    "probe_dt": float(dt_probe),
                    "state": np.asarray(X[i], dtype=float).copy(),
                    "costate": np.asarray(P[i], dtype=float).copy(),
                    "reason": "oracle_dt_sensitivity",
                    "relative_control_change": rel_change,
                    "control_dt": np.asarray(u_star, dtype=float).copy(),
                    "control_probe": np.asarray(u_probe, dtype=float).copy(),
                }
            )
    return sorted(marked), issues


def _refine_selected_intervals(t_nodes, X, P, marked_intervals):
    t_nodes = np.asarray(t_nodes, dtype=float)
    X = np.asarray(X, dtype=float)
    P = np.asarray(P, dtype=float)
    marked_set = {int(i) for i in marked_intervals}
    new_nodes = [t_nodes[0]]
    X_new = [X[0]]
    P_new = [P[0]]
    for i in range(len(t_nodes) - 1):
        dt = t_nodes[i + 1] - t_nodes[i]
        if i in marked_set:
            t_mid = 0.5 * (t_nodes[i] + t_nodes[i + 1])
            alpha = (t_mid - t_nodes[i]) / dt
            x_mid = (1 - alpha) * X[i] + alpha * X[i + 1]
            p_mid = (1 - alpha) * P[i] + alpha * P[i + 1]
            new_nodes.append(t_mid)
            X_new.append(x_mid)
            P_new.append(p_mid)
        new_nodes.append(t_nodes[i + 1])
        X_new.append(X[i + 1])
        P_new.append(P[i + 1])
    return np.array(new_nodes, dtype=float), np.array(X_new), np.array(P_new)


def _local_time_radius(t_nodes: np.ndarray, idx: int, separation_factor: float) -> float:
    if len(t_nodes) <= 1:
        return 0.0
    if idx <= 0:
        local_scale = float(t_nodes[1] - t_nodes[0])
    elif idx >= len(t_nodes) - 1:
        local_scale = float(t_nodes[-1] - t_nodes[-2])
    else:
        local_scale = max(
            float(t_nodes[idx] - t_nodes[idx - 1]),
            float(t_nodes[idx + 1] - t_nodes[idx]),
        )
    return float(separation_factor) * max(local_scale, 0.0)


def compute_pa_ranking_scores(
    t_nodes: np.ndarray,
    pa_gaps: np.ndarray,
):
    t_nodes = np.asarray(t_nodes, dtype=float)
    pa_gaps = np.asarray(pa_gaps, dtype=float)
    if t_nodes.ndim != 1 or pa_gaps.ndim != 1 or len(t_nodes) != len(pa_gaps):
        raise ValueError("t_nodes and pa_gaps must be one-dimensional arrays of equal length.")

    if len(t_nodes) == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    if len(t_nodes) == 1:
        return np.zeros(1, dtype=float), np.zeros(1, dtype=float)

    dt = np.diff(t_nodes)
    node_dt = np.empty(len(t_nodes), dtype=float)
    node_dt[:-1] = dt
    node_dt[-1] = dt[-1]
    pa_scores = pa_gaps * node_dt
    return node_dt, pa_scores


def select_pa_enrichment_candidates(
    t_nodes: np.ndarray,
    ranking_scores: np.ndarray,
    target_count: int,
    *,
    separation_factor: float = 5.0,
    gap_floor_ratio: float = 0.2,
):
    t_nodes = np.asarray(t_nodes, dtype=float)
    ranking_scores = np.asarray(ranking_scores, dtype=float)
    if t_nodes.ndim != 1 or ranking_scores.ndim != 1 or len(t_nodes) != len(ranking_scores):
        raise ValueError("t_nodes and ranking_scores must be one-dimensional arrays of equal length.")

    if len(t_nodes) == 0 or target_count <= 0:
        return [], {
            "target_count": int(max(target_count, 0)),
            "max_score": 0.0,
            "score_floor": 0.0,
            "rejected_by_time_separation": 0,
            "rejected_below_score_floor": 0,
        }

    max_score = float(np.max(ranking_scores))
    if not np.isfinite(max_score) or max_score <= 0.0:
        return [], {
            "target_count": int(target_count),
            "max_score": float(max_score if np.isfinite(max_score) else 0.0),
            "score_floor": float("inf"),
            "rejected_by_time_separation": 0,
            "rejected_below_score_floor": int(len(ranking_scores)),
        }

    score_floor = float(gap_floor_ratio) * max_score
    sorted_indices = list(np.argsort(-ranking_scores))
    selected = []
    rejected_by_time_separation = 0
    rejected_below_score_floor = 0

    for idx in sorted_indices:
        score = float(ranking_scores[idx])
        if score < score_floor:
            rejected_below_score_floor += 1
            continue

        radius_i = _local_time_radius(t_nodes, int(idx), separation_factor)
        too_close = False
        for chosen_idx in selected:
            radius_j = _local_time_radius(t_nodes, int(chosen_idx), separation_factor)
            if abs(float(t_nodes[idx]) - float(t_nodes[chosen_idx])) <= max(radius_i, radius_j):
                too_close = True
                rejected_by_time_separation += 1
                break
        if too_close:
            continue

        selected.append(int(idx))
        if len(selected) >= int(target_count):
            break

    return selected, {
        "target_count": int(target_count),
        "max_score": max_score,
        "score_floor": score_floor,
        "rejected_by_time_separation": int(rejected_by_time_separation),
        "rejected_below_score_floor": int(rejected_below_score_floor),
    }


def bootstrap_bundle_from_trajectory(
    problem,
    t_nodes: np.ndarray,
    X: np.ndarray,
    P: np.ndarray,
    bundle,
    restricted: bool = True,
    num_support_nodes: int = 20,
    grid_size: int = 3,
    use_oracle: bool = False,
    support_log=None,
    iteration: int | None = None,
) -> int:
    """
    Bootstrap for PA bundle:
    - If an explicit oracle u_star is available, use it to add candidate controls.
    - Otherwise (or if oracle is not feasible under `restricted`), fall back to a cheap 1D grid search
      (only for scalar control with bounds).
    Returns the number of *new* controls added.
    """

    # detect whether an oracle exists (your OCPProblem.u_star returns (u, ok) or (None, False))
    has_oracle = hasattr(problem, "u_star")

    bounds = problem.control_bounds_tuple()
    u_grid = None

    # grid search is only possible if bounds exist and control is scalar
    if bounds is not None:
        u_min, u_max = bounds
        m = int(u_min.size)
        if m == 1:
            u_grid = np.linspace(float(u_min[0]), float(u_max[0]), int(grid_size))

    N = len(t_nodes) - 1
    if N <= 0:
        return 0

    # pick representative node indices (including endpoints)
    k = min(num_support_nodes, N + 1)
    idx = np.unique(np.round(np.linspace(0, N, k)).astype(int))

    added = 0

    for i in idx:
        x_i = X[i]
        p_i = P[i]
        t_i = float(t_nodes[i])
        dt_i = _local_dt_at_node(t_nodes, int(i))

        # ------------------------------------------------------------
        # 1) Try oracle u_star first (does projection to bounds inside u_star)
        # ------------------------------------------------------------
        if use_oracle and has_oracle:
            u_oracle, ok = problem.u_star(x_i, p_i, t_i, restricted=restricted, dt=dt_i)
            if (u_oracle is not None) and (not restricted or ok):
                before = bundle.num_planes()
                bundle.add_control(u_oracle)
                if bundle.num_planes() > before:
                    added += 1
                    if support_log is not None:
                        support_log.append({
                            "iteration": iteration,
                            "kind": "bootstrap",
                            "node_index": int(i),
                            "time": float(t_i),
                            "state": np.asarray(x_i, dtype=float).copy(),
                            "costate": np.asarray(p_i, dtype=float).copy(),
                            "control": np.asarray(u_oracle, dtype=float).copy(),
                            "local_dt": float(dt_i) if dt_i is not None else None,
                            "bundle_size_after": int(bundle.num_planes()),
                        })
                # oracle succeeded -> no need for grid search at this node
                continue

        # ------------------------------------------------------------
        # 2) Fallback: grid search (only if available)
        # ------------------------------------------------------------
        if u_grid is None:
            continue

        best_val = np.inf
        best_u = None

        for a in u_grid:
            u = np.array([a], dtype=float)

            if not problem.local_control_feasible(x_i, u, t_i, restricted=restricted, dt=dt_i):
                continue

            val = float(np.dot(p_i, problem.f(x_i, u, t_i)) + problem.l(x_i, u, t_i))
            if val < best_val:
                best_val = val
                best_u = u

        if best_u is not None:
            before = bundle.num_planes()
            bundle.add_control(best_u)
            if bundle.num_planes() > before:
                added += 1
                if support_log is not None:
                    support_log.append({
                        "iteration": iteration,
                        "kind": "bootstrap",
                        "node_index": int(i),
                        "time": float(t_i),
                        "state": np.asarray(x_i, dtype=float).copy(),
                        "costate": np.asarray(p_i, dtype=float).copy(),
                        "control": np.asarray(best_u, dtype=float).copy(),
                        "local_dt": float(dt_i) if dt_i is not None else None,
                        "bundle_size_after": int(bundle.num_planes()),
                    })

    return added



def solve_optimal_control(
    problem,
    initial_mesh: np.ndarray,
    tol_time: float = 1e-3,
    tol_PA: float = 1e-3,
    tol_delta: float = 1e-3,
    max_iters: int = 10,
    delta0: float = 0.1,
    s_time: float = 0.5,
    K_time: float = 1e-6,
    newton_tol: float = 1e-10,
    newton_max_iter: int = 50,
    verbose: bool =  True,
    print_every: int=1,
    log_path: str = "logs/last_run.txt",
    use_oracle_bootstrap: bool = False,
    use_oracle_PA: bool = False,
    use_explicit_hamiltonian_gradients: bool = False,
    store_iterates: bool = False,
    fallback_solver: str | None = "least_squares",
    time_balance_ratio: float = 0.1,
    pa_add_fraction: float = 0.1,
    pa_time_separation_factor: float = 5.0,
    pa_gap_floor_ratio: float = 0.2,
    feasibility_probe_ratio: float = 0.5,
    feasibility_control_sensitivity: float = 0.05,
    initial_X_guess: np.ndarray | None = None,
    initial_P_guess: np.ndarray | None = None,
    initial_guess_label: str = "default",
):
    """
    Solve an optimal control problem adaptively by refining time mesh,
    adding new control planes, and reducing the smoothing parameter.

    Parameters
    ----------
    problem : OCPProblem
        Problem definition.
    initial_mesh : np.ndarray
        Initial time grid (including 0 and T).  Should be sorted.
    tol_time : float
        Tolerance for the time discretisation error indicator.
    tol_PA : float
        Tolerance for the PA surrogate error indicator.
    tol_delta : float
        Tolerance for the smoothing error indicator.
    max_iters : int
        Maximum number of outer adaptivity iterations.
    delta0 : float
        Initial smoothing parameter.

    Returns
    -------
    dict
        Dictionary with solution, mesh, bundle, delta, and log information.
    """
# -------------------------
    validate_indicator_tolerances(tol_time, tol_PA, tol_delta)

    # -------------------------
    # Persistent run log file (overwritten each run)
    # -------------------------
    log_f = None
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log_f = open(log_path, "w", buffering=1)  # overwrite, line-buffered
        atexit.register(log_f.close)

    def _log(msg: str):
        if verbose:
            print(msg, flush=True)
        if log_f is not None:
            print(msg, file=log_f, flush=True)

    # copy mesh
    t_nodes = np.asarray(initial_mesh, dtype=float).copy()
    # initialize PA bundle with zero control if dimension known, otherwise empty
    bundle = PABundle()
    # try to add zero control (or mean of bounds) if possible
    bounds = problem.control_bounds_tuple()
    m = problem.m
    if m is None and bounds is not None:
        m = bounds[0].size
    if m is not None:
        if bounds is not None:
            u_min, u_max = bounds
            u_mid = 0.5 * (u_min + u_max)
            bundle.add_control(u_mid)
            bundle.add_control(u_min)
            bundle.add_control(u_max)
        else:
            u0 = np.zeros(m)
            bundle.add_control(u0)
    delta = delta0
    log = []
    bundle_support_points = []
    # initial guesses for X and P: None (will be set in Newton)
    X_guess = None if initial_X_guess is None else np.asarray(initial_X_guess, dtype=float).copy()
    P_guess = None if initial_P_guess is None else np.asarray(initial_P_guess, dtype=float).copy()
    # Hard-disable policy:
    # In explicit Hamiltonian-gradient mode, PA/delta criteria are not used
    # to drive adaptivity decisions (time indicator still drives refinement).
    explicit_mode = bool(use_explicit_hamiltonian_gradients)

    #==================================== Outer Loop =====================================================
    for k in range(max_iters):
        # solve TPBVP on current mesh with current bundle and delta
        X, P, info = solve_tpbvp(
            problem,
            t_nodes,
            bundle,
            delta,
            X_guess,
            P_guess,
            tol=newton_tol,
            max_iter=newton_max_iter,
            use_explicit_hamiltonian_gradients=use_explicit_hamiltonian_gradients,
            fallback_solver=fallback_solver,
        )
        # --- bootstrap PA bundle after first coarse solve (minimal change) ---
        if k == 0 and (not use_explicit_hamiltonian_gradients):
            M_before = bundle.num_planes()
            added = bootstrap_bundle_from_trajectory(
                problem,
                t_nodes=t_nodes,
                X=X,
                P=P,
                bundle=bundle,
                restricted=True,
                num_support_nodes=12,
                grid_size=51,
                use_oracle=use_oracle_bootstrap,
                support_log=bundle_support_points,
                iteration=k,
            )
            _log(f"[bootstrap] M_before={M_before}, added={added}, M_after={bundle.num_planes()}")
            if added > 0:
                # re-solve once with improved bundle (same mesh, same delta)
                X_guess, P_guess = X, P
                X, P, info = solve_tpbvp(
                    problem,
                    t_nodes,
                    bundle,
                    delta,
                    X_guess,
                    P_guess,
                    tol=newton_tol,
                    max_iter=newton_max_iter,
                    use_explicit_hamiltonian_gradients=use_explicit_hamiltonian_gradients,
                    fallback_solver=fallback_solver,
                )

        if (not explicit_mode) and getattr(problem, "step_feasible_control_fn", None) is not None:
            marked_feasibility, feasibility_issues = _find_feasibility_refinement_intervals(
                problem,
                bundle,
                X,
                P,
                t_nodes,
                use_oracle=use_oracle_PA,
                probe_ratio=feasibility_probe_ratio,
                control_sensitivity=feasibility_control_sensitivity,
            )
            if marked_feasibility:
                action = f"refine_feasibility(marked={len(marked_feasibility)})"
                entry = {
                    'iteration': k,
                    'N': len(t_nodes) - 1,
                    'M': bundle.num_planes(),
                    'delta': delta,
                    'eta_time': np.nan,
                    'eta_time_sum': np.nan,
                    'eta_PA': np.nan,
                    'eta_delta': np.nan,
                    'rho': np.zeros(max(len(t_nodes) - 1, 0)),
                    'rho_bar': np.zeros(max(len(t_nodes) - 1, 0)),
                    'r_bar': np.zeros(max(len(t_nodes) - 1, 0)),
                    'eta_PA_local': np.zeros(max(len(t_nodes) - 1, 0)),
                    'eta_delta_local': np.zeros(max(len(t_nodes) - 1, 0)),
                    'pa_gap_nodes': np.zeros(len(t_nodes)),
                    'delta_gap_nodes': np.zeros(len(t_nodes)),
                    'active_plane_idx_nodes': np.zeros(len(t_nodes), dtype=int),
                    'tol_time_star': np.nan,
                    'mark_thr': np.nan,
                    'time_balance_ratio': float(time_balance_ratio),
                    'dominant_non_time_indicator': np.nan,
                    'time_refinement_suppressed': False,
                    'pa_add_fraction': float(pa_add_fraction),
                    'pa_time_separation_factor': float(pa_time_separation_factor),
                    'pa_gap_floor_ratio': float(pa_gap_floor_ratio),
                    'initial_guess_label': initial_guess_label,
                    'pa_addition_plan': {"target_count": 0},
                    't_nodes_iter': t_nodes.copy(),
                    'newton_iter': info['iterations'],
                    'newton_residual': info['residual_norm'],
                    'solver_phase': info.get('solver_phase', 'newton'),
                    'fallback_used': bool(info.get('fallback_used', False)),
                    'objective_mesh_approx': np.nan,
                    'all_indicators_within_tolerance': False,
                    'action': action,
                    'note': 'feasibility_refinement',
                    'feasibility_issues': feasibility_issues,
                }
                if store_iterates:
                    entry['X_iter'] = X.copy()
                    entry['P_iter'] = P.copy()
                    entry['U_iter'] = _compute_node_controls(problem, bundle, X, P, t_nodes, restricted=True, use_oracle=use_oracle_PA)
                    entry['objective_mesh_approx'] = _mesh_objective(problem, t_nodes, X, entry['U_iter'])
                    entry['bundle_support_points_so_far'] = [
                        {
                            'iteration': point.get('iteration'),
                            'kind': point.get('kind'),
                            'node_index': point.get('node_index'),
                            'time': float(point.get('time')),
                            'state': np.asarray(point.get('state'), dtype=float).copy(),
                            'costate': np.asarray(point.get('costate'), dtype=float).copy(),
                            'control': np.asarray(point.get('control'), dtype=float).copy(),
                            'local_dt': point.get('local_dt'),
                            'bundle_size_after': int(point.get('bundle_size_after')),
                        }
                        for point in bundle_support_points
                    ]
                log.append(entry)
                _log(
                    f"[adapt {k:02d}] feasibility refinement triggered at "
                    f"{len(marked_feasibility)} interval(s); warm restart on refined mesh."
                )
                t_nodes, X_guess, P_guess = _refine_selected_intervals(t_nodes, X, P, marked_feasibility)
                refreshed_bundle = _refresh_bundle_support_controls(problem, t_nodes, X_guess, P_guess, bundle_support_points, use_oracle=use_oracle_PA)
                if refreshed_bundle is not None:
                    bundle = refreshed_bundle
                continue
        delta_solved = delta
        # compute error indicators
        #=======================================================================
        # TIME DISCRETIZATION ERROR
        #=======================================================================
        N = len(t_nodes) - 1
        eta_time_local = np.zeros(N)   # will store r_bar_n
        
        if N > 0:
            dt = np.diff(t_nodes)
            dt_max = float(np.max(dt))
            floor = K_time * np.sqrt(dt_max)

            rho_arr = np.zeros(N)
            rho_bar_arr = np.zeros(N)
            for i in range(N):
                # evaluate at symplectic-Euler point (p_{i+1}, x_i, t_i)
                _, Hp, Hx = _grads_for_indicators(
                    problem,
                    bundle,
                    P[i + 1],
                    X[i],
                    t_nodes[i],
                    delta,
                    dt=dt[i],
                    use_explicit_hamiltonian_gradients=use_explicit_hamiltonian_gradients,
                )

                rho_arr[i] = -0.5 * float(np.dot(Hp, Hx))
                rho_bar_arr[i] = max(abs(rho_arr[i]), floor)
                #rho_bar_arr[i] = np.sign(rho_arr[i]) * max(abs(rho_arr[i]), floor)

                eta_time_local[i] = abs(rho_bar_arr[i]) * (dt[i] ** 2)  # r_bar_i

            eta_time = float(np.max(eta_time_local))          # r*
            eta_time_sum = float(np.sum(eta_time_local))      # total estimated time error
            tol_time_star = float(tol_time / N)               # TOL/N
            mark_thr = float(s_time * tol_time / N)           # s*TOL/N
        else:
            eta_time = 0.0
            eta_time_sum = 0.0
            tol_time_star = tol_time
            mark_thr = 0.0
        
    
        if explicit_mode:
            eta_PA = 0.0
            pa_gap_nodes = np.zeros(N + 1)
            eta_PA_local = np.zeros(N)
            active_plane_idx_nodes = np.zeros(N + 1, dtype=int)
            eta_delta = 0.0
            delta_gap_nodes = np.zeros(N + 1)
            eta_delta_local = np.zeros(N)
        else:
            # PA error: integrate (Hbar - H)
            eta_PA = 0.0
            pa_gap_nodes = np.zeros(N + 1)
            eta_PA_local = np.zeros(N)
            active_plane_idx_nodes = np.zeros(N + 1, dtype=int)
            for i in range(N):
                # at node i and i+1, compute gap
                dt_i = _local_dt_at_node(t_nodes, i)
                dt_ip1 = _local_dt_at_node(t_nodes, i + 1)
                Hbar_i, idx_i = bundle.evaluate(problem, P[i], X[i], t_nodes[i], dt=dt_i, restricted=True, fallback_unrestricted=False)
                Hbar_ip1, idx_ip1 = bundle.evaluate(problem, P[i + 1], X[i + 1], t_nodes[i + 1], dt=dt_ip1, restricted=True, fallback_unrestricted=False)
                # compute true H (restricted) at i and i+1
                H_i, _ = compute_H(problem, P[i], X[i], t_nodes[i], bundle.controls, restricted=True, use_oracle=use_oracle_PA, dt=dt_i)
                H_ip1, _ = compute_H(problem, P[i + 1], X[i + 1], t_nodes[i + 1], bundle.controls, restricted=True, use_oracle=use_oracle_PA, dt=dt_ip1)
                gap_i = Hbar_i - H_i
                gap_ip1 = Hbar_ip1 - H_ip1
                pa_gap_nodes[i] = gap_i
                pa_gap_nodes[i + 1] = gap_ip1
                active_plane_idx_nodes[i] = idx_i
                active_plane_idx_nodes[i + 1] = idx_ip1
                dt = t_nodes[i + 1] - t_nodes[i]
                eta_PA_local[i] = 0.5 * (gap_i + gap_ip1) * dt
                eta_PA += eta_PA_local[i]
            # smoothing error: integrate (H_delta - Hbar)
            eta_delta = 0.0
            delta_gap_nodes = np.zeros(N + 1)
            eta_delta_local = np.zeros(N)
            for i in range(N):
                dt_i = _local_dt_at_node(t_nodes, i)
                dt_ip1 = _local_dt_at_node(t_nodes, i + 1)
                Hdelta_i, _, _ = eval_H_smooth(problem, bundle, P[i], X[i], t_nodes[i], delta, dt=dt_i)
                Hdelta_ip1, _, _ = eval_H_smooth(problem, bundle, P[i + 1], X[i + 1], t_nodes[i + 1], delta, dt=dt_ip1)
                Hbar_i, _ = bundle.evaluate(problem, P[i], X[i], t_nodes[i], dt=dt_i)
                Hbar_ip1, _ = bundle.evaluate(problem, P[i + 1], X[i + 1], t_nodes[i + 1], dt=dt_ip1)
                diff_i = Hbar_i - Hdelta_i 
                diff_ip1 = Hbar_ip1 - Hdelta_ip1
                delta_gap_nodes[i] = diff_i
                delta_gap_nodes[i + 1] = diff_ip1
                dt = t_nodes[i + 1] - t_nodes[i]
                eta_delta_local[i] = 0.5 * (diff_i + diff_ip1) * dt
                eta_delta += eta_delta_local[i]
        n_mark = int(np.sum(eta_time_local > mark_thr)) if N > 0 else 0
        dominant_non_time_indicator = float(max(eta_PA, eta_delta)) if not explicit_mode else 0.0
        time_refinement_suppressed = (
            (not explicit_mode)
            and dominant_non_time_indicator > 0.0
            and eta_time <= time_balance_ratio * dominant_non_time_indicator
        )
        action = choose_adaptive_action(
            eta_time,
            tol_time_star,
            eta_PA,
            tol_PA,
            eta_delta,
            tol_delta,
            n_mark,
            explicit_mode=explicit_mode,
            time_balance_ratio=time_balance_ratio,
        )
        pa_addition_plan = {
            "target_count": 0,
            "selected_node_indices": [],
            "selected_times": [],
            "selected_pa_gaps": [],
            "selected_time_steps": [],
            "selected_pa_scores": [],
            "selected_added_to_bundle": [],
            "selected_bundle_size_after": [],
            "rejected_by_time_separation": 0,
            "rejected_as_duplicate_control": 0,
            "rejected_below_score_floor": 0,
            "max_gap": 0.0,
            "max_score": 0.0,
            "score_floor": 0.0,
        }
        selected_pa_controls = []
        if action == "add_plane":
            target_count = max(1, int(np.ceil(float(pa_add_fraction) * max(bundle.num_planes(), 1))))
            pa_candidate_controls = []
            for i in range(N + 1):
                dt_i = _local_dt_at_node(t_nodes, i)
                _, u_star = compute_H(
                    problem,
                    P[i],
                    X[i],
                    t_nodes[i],
                    bundle.controls,
                    restricted=True,
                    use_oracle=use_oracle_PA,
                    dt=dt_i,
                )
                pa_candidate_controls.append(u_star)
            pa_node_dt, pa_scores = compute_pa_ranking_scores(t_nodes, pa_gap_nodes)
            selected_indices, selection_meta = select_pa_enrichment_candidates(
                t_nodes,
                pa_scores,
                target_count,
                separation_factor=pa_time_separation_factor,
                gap_floor_ratio=pa_gap_floor_ratio,
            )
            selected_pa_controls = [pa_candidate_controls[idx] for idx in selected_indices]
            pa_addition_plan.update(
                {
                    "target_count": int(target_count),
                    "selected_node_indices": [int(idx) for idx in selected_indices],
                    "selected_times": [float(t_nodes[idx]) for idx in selected_indices],
                    "selected_pa_gaps": [float(pa_gap_nodes[idx]) for idx in selected_indices],
                    "selected_time_steps": [float(pa_node_dt[idx]) for idx in selected_indices],
                    "selected_pa_scores": [float(pa_scores[idx]) for idx in selected_indices],
                    "rejected_by_time_separation": int(selection_meta["rejected_by_time_separation"]),
                    "rejected_below_score_floor": int(selection_meta["rejected_below_score_floor"]),
                    "max_gap": float(np.max(pa_gap_nodes) if len(pa_gap_nodes) > 0 else 0.0),
                    "max_score": float(selection_meta["max_score"]),
                    "score_floor": float(selection_meta["score_floor"]),
                }
            )
        all_indicators_within_tolerance = (
            eta_time <= tol_time_star
            and (explicit_mode or eta_PA <= tol_PA)
            and (explicit_mode or eta_delta <= tol_delta)
        )
        U = _compute_node_controls(problem, bundle, X, P, t_nodes, restricted=True, use_oracle=use_oracle_PA)
        entry = {
            'iteration': k,
            'N': N,
            'M': bundle.num_planes(),
            'delta': delta,
            'eta_time': eta_time,
            'eta_time_sum': eta_time_sum,
            'eta_PA': eta_PA,
            'eta_delta': eta_delta,
            'rho': rho_arr.copy(),
            'rho_bar': rho_bar_arr.copy(),
            'r_bar': eta_time_local.copy(),
            'eta_PA_local': eta_PA_local.copy(),
            'eta_delta_local': eta_delta_local.copy(),
            'pa_gap_nodes': pa_gap_nodes.copy(),
            'delta_gap_nodes': delta_gap_nodes.copy(),
            'active_plane_idx_nodes': active_plane_idx_nodes.copy(),
            'tol_time_star': tol_time_star,
            'mark_thr': mark_thr,
            'time_balance_ratio': float(time_balance_ratio),
            'dominant_non_time_indicator': dominant_non_time_indicator,
            'time_refinement_suppressed': bool(time_refinement_suppressed),
            'pa_add_fraction': float(pa_add_fraction),
            'pa_time_separation_factor': float(pa_time_separation_factor),
            'pa_gap_floor_ratio': float(pa_gap_floor_ratio),
            'initial_guess_label': initial_guess_label,
            'pa_addition_plan': pa_addition_plan,
            't_nodes_iter': t_nodes.copy(),
            'newton_iter': info['iterations'],
            'newton_residual': info['residual_norm'],
            'solver_phase': info.get('solver_phase', 'newton'),
            'fallback_used': bool(info.get('fallback_used', False)),
            'objective_mesh_approx': _mesh_objective(problem, t_nodes, X, U),
            'all_indicators_within_tolerance': bool(all_indicators_within_tolerance),
            'action': action,
            'note': '',
        }
        if store_iterates:
            entry['X_iter'] = X.copy()
            entry['P_iter'] = P.copy()
            entry['U_iter'] = U.copy()
            entry['bundle_support_points_so_far'] = [
                {
                    'iteration': point.get('iteration'),
                    'kind': point.get('kind'),
                    'node_index': point.get('node_index'),
                    'time': float(point.get('time')),
                    'state': np.asarray(point.get('state'), dtype=float).copy(),
                    'costate': np.asarray(point.get('costate'), dtype=float).copy(),
                    'control': np.asarray(point.get('control'), dtype=float).copy(),
                    'local_dt': point.get('local_dt'),
                    'bundle_size_after': int(point.get('bundle_size_after')),
                }
                for point in bundle_support_points
            ]
        log.append(entry)
# ------------------------------------------------------------
        # Per-iteration concise print (1 line): progress + next action
        # ------------------------------------------------------------
        if (k % max(int(print_every), 1)) == 0:
            dt_all = np.diff(t_nodes)
            dt_min = float(np.min(dt_all)) if dt_all.size else 0.0
            dt_max_ = float(np.max(dt_all)) if dt_all.size else 0.0

            _log(
                f"[adapt {k:02d}] "
                f"N={N:4d} M={bundle.num_planes():3d} dt=[{dt_min:.2e},{dt_max_:.2e}] delta={delta:.2e} | "
                f"Newton it={info['iterations']:2d} res={info['residual_norm']:.2e} | "
                f"eta_time={eta_time:.2e}/{tol_time_star:.2e} "
                f"eta_PA={eta_PA:.2e}/{tol_PA:.2e} "
                f"eta_delta={eta_delta:.2e}/{tol_delta:.2e} -> {action}"
            )


        # check convergence
        if action == "STOP":
            break
        # priority: refine time first, then PA planes, then reduce delta

        if action.startswith("refine_time"):
            # refine time mesh: subdivide intervals with high local error
            new_nodes = [t_nodes[0]]
            X_new = [X[0]]
            P_new = [P[0]]
            for i in range(N):
                dt = t_nodes[i + 1] - t_nodes[i]
                # compute midpoint and error indicator
                err = eta_time_local[i]
                if err > mark_thr:
                #if err > tol_time:
                    # insert midpoint
                    t_mid = 0.5 * (t_nodes[i] + t_nodes[i + 1])
                    # linear interpolate X and P
                    alpha = (t_mid - t_nodes[i]) / dt
                    x_mid = (1 - alpha) * X[i] + alpha * X[i + 1]
                    p_mid = (1 - alpha) * P[i] + alpha * P[i + 1]
                    new_nodes.extend([t_mid])
                    X_new.extend([x_mid])
                    P_new.extend([p_mid])
                new_nodes.append(t_nodes[i + 1])
                X_new.append(X[i + 1])
                P_new.append(P[i + 1])
            t_nodes = np.array(new_nodes, dtype=float)
            X_guess = np.array(X_new)
            P_guess = np.array(P_new)
            refreshed_bundle = _refresh_bundle_support_controls(
                problem, t_nodes, X_guess, P_guess, bundle_support_points, use_oracle=use_oracle_PA
            )
            if refreshed_bundle is not None:
                bundle = refreshed_bundle
            continue
        
        if action == "add_plane":
            duplicate_rejections = 0
            for idx, candidate_u in zip(pa_addition_plan["selected_node_indices"], selected_pa_controls):
                if candidate_u is None:
                    pa_addition_plan["selected_added_to_bundle"].append(False)
                    pa_addition_plan["selected_bundle_size_after"].append(int(bundle.num_planes()))
                    continue
                before = bundle.num_planes()
                bundle.add_control(candidate_u)
                if bundle.num_planes() > before:
                    local_dt = _local_dt_at_node(t_nodes, int(idx))
                    pa_addition_plan["selected_added_to_bundle"].append(True)
                    pa_addition_plan["selected_bundle_size_after"].append(int(bundle.num_planes()))
                    bundle_support_points.append({
                        "iteration": k,
                        "kind": "add_plane",
                        "node_index": int(idx),
                        "time": float(t_nodes[idx]),
                        "state": np.asarray(X[idx], dtype=float).copy(),
                        "costate": np.asarray(P[idx], dtype=float).copy(),
                        "control": np.asarray(candidate_u, dtype=float).copy(),
                        "local_dt": float(local_dt) if local_dt is not None else None,
                        "bundle_size_after": int(bundle.num_planes()),
                        "pa_gap_at_point": float(pa_gap_nodes[idx]),
                        "pa_time_step_at_point": float(pa_node_dt[idx]),
                        "pa_score_at_point": float(pa_scores[idx]),
                    })
                else:
                    duplicate_rejections += 1
                    pa_addition_plan["selected_added_to_bundle"].append(False)
                    pa_addition_plan["selected_bundle_size_after"].append(int(bundle.num_planes()))
            pa_addition_plan["rejected_as_duplicate_control"] = int(duplicate_rejections)
            if store_iterates:
                log[-1]['bundle_support_points_so_far'] = [
                    {
                        'iteration': point.get('iteration'),
                        'kind': point.get('kind'),
                        'node_index': point.get('node_index'),
                        'time': float(point.get('time')),
                        'state': np.asarray(point.get('state'), dtype=float).copy(),
                        'costate': np.asarray(point.get('costate'), dtype=float).copy(),
                        'control': np.asarray(point.get('control'), dtype=float).copy(),
                        'local_dt': point.get('local_dt'),
                        'bundle_size_after': int(point.get('bundle_size_after')),
                        'pa_gap_at_point': float(point.get('pa_gap_at_point', 0.0)),
                        'pa_time_step_at_point': float(point.get('pa_time_step_at_point', 0.0)),
                        'pa_score_at_point': float(point.get('pa_score_at_point', 0.0)),
                    }
                    for point in bundle_support_points
                ]
            X_guess = X
            P_guess = P
            continue
        # else reduce delta
        if action == "delta*=0.5":
            delta = delta * 0.5
            # do not change mesh or bundle
            X_guess = X
            P_guess = P
            continue
    # return final solution and log
    if (X is None) or (len(t_nodes) != X.shape[0]) or (len(t_nodes) != P.shape[0]) or (delta_solved != delta):
        X, P, info = solve_tpbvp(
            problem,
            t_nodes,
            bundle,
            delta,
            X_guess,
            P_guess,
            tol=newton_tol,
            max_iter=newton_max_iter,
            use_explicit_hamiltonian_gradients=use_explicit_hamiltonian_gradients,
            fallback_solver=fallback_solver,
        )
        #(final_resolve): At this point we re-solve TPBVP so that (X,P) match the returned `delta`.
        # However, the error indicators (eta_time, eta_PA, eta_delta) below are NOT recomputed at this final delta;
        # they may correspond to the previous outer-iteration values. Recompute them here later if needed.
        N = len(t_nodes) - 1
        eta_time_local = np.zeros(N)

        dt = np.diff(t_nodes) if N > 0 else np.zeros(0)
        dt_max = float(np.max(dt)) if N > 0 else 0.0
        floor = K_time * np.sqrt(dt_max) if N > 0 else 0.0

        rho_arr = np.zeros(N)
        rho_bar_arr = np.zeros(N)

        for i in range(N):
            _, Hp, Hx = _grads_for_indicators(
                problem,
                bundle,
                P[i + 1],
                X[i],
                t_nodes[i],
                delta,
                dt=dt[i],
                use_explicit_hamiltonian_gradients=use_explicit_hamiltonian_gradients,
            )
            rho_arr[i] = -0.5 * float(np.dot(Hp, Hx))
            rho_bar_arr[i] = max(abs(rho_arr[i]), floor)
            eta_time_local[i] = abs(rho_bar_arr[i]) * (dt[i] ** 2)

        eta_time = float(np.max(eta_time_local)) if N > 0 else 0.0
        eta_time_sum = float(np.sum(eta_time_local)) if N > 0 else 0.0
        tol_time_star = float(tol_time / N) if N > 0 else tol_time
        mark_thr = float(s_time * tol_time / N) if N > 0 else 0.0

        if explicit_mode:
            eta_PA = 0.0
            eta_delta = 0.0
            pa_gap_nodes = np.zeros(N + 1)
            eta_PA_local = np.zeros(N)
            active_plane_idx_nodes = np.zeros(N + 1, dtype=int)
            delta_gap_nodes = np.zeros(N + 1)
            eta_delta_local = np.zeros(N)
        else:
            eta_PA = 0.0
            pa_gap_nodes = np.zeros(N + 1)
            eta_PA_local = np.zeros(N)
            active_plane_idx_nodes = np.zeros(N + 1, dtype=int)
            for i in range(N):
                dt_i = _local_dt_at_node(t_nodes, i)
                dt_ip1 = _local_dt_at_node(t_nodes, i + 1)
                Hbar_i, idx_i = bundle.evaluate(problem, P[i], X[i], t_nodes[i], dt=dt_i, restricted=True, fallback_unrestricted=False)
                Hbar_ip1, idx_ip1 = bundle.evaluate(problem, P[i + 1], X[i + 1], t_nodes[i + 1], dt=dt_ip1, restricted=True, fallback_unrestricted=False)
                H_i, _ = compute_H(problem, P[i], X[i], t_nodes[i], bundle.controls, restricted=True, use_oracle=use_oracle_PA, dt=dt_i)
                H_ip1, _ = compute_H(problem, P[i + 1], X[i + 1], t_nodes[i + 1], bundle.controls, restricted=True, use_oracle=use_oracle_PA, dt=dt_ip1)
                gap_i = Hbar_i - H_i
                gap_ip1 = Hbar_ip1 - H_ip1
                pa_gap_nodes[i] = gap_i
                pa_gap_nodes[i + 1] = gap_ip1
                active_plane_idx_nodes[i] = idx_i
                active_plane_idx_nodes[i + 1] = idx_ip1
                dt_i = t_nodes[i + 1] - t_nodes[i]
                eta_PA_local[i] = 0.5 * (gap_i + gap_ip1) * dt_i
                eta_PA += eta_PA_local[i]

            eta_delta = 0.0
            delta_gap_nodes = np.zeros(N + 1)
            eta_delta_local = np.zeros(N)
            for i in range(N):
                dt_i = _local_dt_at_node(t_nodes, i)
                dt_ip1 = _local_dt_at_node(t_nodes, i + 1)
                Hdelta_i, _, _ = eval_H_smooth(problem, bundle, P[i], X[i], t_nodes[i], delta, dt=dt_i)
                Hdelta_ip1, _, _ = eval_H_smooth(problem, bundle, P[i + 1], X[i + 1], t_nodes[i + 1], delta, dt=dt_ip1)
                Hbar_i, _ = bundle.evaluate(problem, P[i], X[i], t_nodes[i], dt=dt_i)
                Hbar_ip1, _ = bundle.evaluate(problem, P[i + 1], X[i + 1], t_nodes[i + 1], dt=dt_ip1)
                diff_i = Hbar_i - Hdelta_i
                diff_ip1 = Hbar_ip1 - Hdelta_ip1
                delta_gap_nodes[i] = diff_i
                delta_gap_nodes[i + 1] = diff_ip1
                dt_i = t_nodes[i + 1] - t_nodes[i]
                eta_delta_local[i] = 0.5 * (diff_i + diff_ip1) * dt_i
                eta_delta += eta_delta_local[i]

        all_indicators_within_tolerance = (
            eta_time <= tol_time_star
            and (explicit_mode or eta_PA <= tol_PA)
            and (explicit_mode or eta_delta <= tol_delta)
        )
        U = _compute_node_controls(problem, bundle, X, P, t_nodes, restricted=True, use_oracle=use_oracle_PA)

        final_action = 'STOP' if all_indicators_within_tolerance else 'final_resolve'
        final_note = '' if all_indicators_within_tolerance else 'final_resolve'

        if len(log) > 0:
            entry = {
                'iteration': log[-1]['iteration'] + 1,
                'N': len(t_nodes) - 1,
                'M': bundle.num_planes(),
                'delta': delta,
                'eta_time': eta_time,
                'eta_time_sum': eta_time_sum,
                'eta_PA': eta_PA,
                'eta_delta': eta_delta,
                #'eta_time': log[-1]['eta_time'],   # (opcional) si quieres exactitud, luego lo recalculamos
                #'eta_PA': log[-1]['eta_PA'],
                #'eta_delta': log[-1]['eta_delta'],
                'newton_iter': info['iterations'],
                'newton_residual': info['residual_norm'],
                'solver_phase': info.get('solver_phase', 'newton'),
                'fallback_used': bool(info.get('fallback_used', False)),
                'objective_mesh_approx': _mesh_objective(problem, t_nodes, X, U),
                'all_indicators_within_tolerance': bool(all_indicators_within_tolerance),
                'note': final_note,
                'action': final_action,
                'rho': rho_arr.copy(),
                'rho_bar': rho_bar_arr.copy(),
                'r_bar': eta_time_local.copy(),
                'eta_PA_local': eta_PA_local.copy(),
                'eta_delta_local': eta_delta_local.copy(),
                'pa_gap_nodes': pa_gap_nodes.copy(),
                'delta_gap_nodes': delta_gap_nodes.copy(),
                'active_plane_idx_nodes': active_plane_idx_nodes.copy(),
                'tol_time_star': tol_time_star,
                'mark_thr': mark_thr,
                't_nodes_iter': t_nodes.copy(),
            }
            if store_iterates:
                entry['X_iter'] = X.copy()
                entry['P_iter'] = P.copy()
                entry['U_iter'] = U.copy()
                entry['bundle_support_points_so_far'] = [
                    {
                        'iteration': point.get('iteration'),
                        'kind': point.get('kind'),
                        'node_index': point.get('node_index'),
                        'time': float(point.get('time')),
                        'state': np.asarray(point.get('state'), dtype=float).copy(),
                        'costate': np.asarray(point.get('costate'), dtype=float).copy(),
                        'control': np.asarray(point.get('control'), dtype=float).copy(),
                        'local_dt': point.get('local_dt'),
                        'bundle_size_after': int(point.get('bundle_size_after')),
                        'pa_gap_at_point': float(point.get('pa_gap_at_point', 0.0)),
                    }
                    for point in bundle_support_points
                ]
            log.append(entry)
    return {
        't_nodes': t_nodes,
        'X': X,
        'P': P,
        'bundle': bundle,
        'rhobar'  : rho_bar_arr,
        'rbar'  : eta_time_local,
        'delta': delta,
        'log': log,
        'info': info,
        'problem': problem,
        'bundle_support_points': bundle_support_points,
        'settings': {
            'tol_time': tol_time,
            'tol_PA': tol_PA,
            'tol_delta': tol_delta,
            'max_iters': max_iters,
            'delta0': delta0,
            's_time': s_time,
            'K_time': K_time,
            'newton_tol': newton_tol,
            'newton_max_iter': newton_max_iter,
            'use_oracle_bootstrap': bool(use_oracle_bootstrap),
            'use_oracle_PA': bool(use_oracle_PA),
            'use_explicit_hamiltonian_gradients': bool(use_explicit_hamiltonian_gradients),
            'store_iterates': bool(store_iterates),
            'fallback_solver': fallback_solver,
            'time_balance_ratio': float(time_balance_ratio),
            'pa_add_fraction': float(pa_add_fraction),
            'pa_time_separation_factor': float(pa_time_separation_factor),
            'pa_gap_floor_ratio': float(pa_gap_floor_ratio),
            'initial_guess_label': initial_guess_label,
        },
    }

  
