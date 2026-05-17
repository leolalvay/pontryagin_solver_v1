# `core/adaptivity.py` — Adaptive outer loop for the smoothed Pontryagin solver

This file implements the **outer adaptive loop** of the solver. The inner solver,
implemented in `core/newton.py`, solves a discretized smoothed PMP two-point
boundary value problem (TPBVP) on a fixed mesh, with a fixed PA bundle and fixed
smoothing parameter. The outer loop then inspects a posteriori indicators and
updates the discretization/model before solving again.

At a high level, each outer iteration may update one of the following objects:

1. the **time mesh** $t_0 < t_1 < \cdots < t_N$,
2. the **piecewise-affine bundle** $U_{\mathrm{bundle}}$,
3. the **smoothing parameter** $\delta$.

The main entry point is:

```python
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
    verbose: bool = True,
    print_every: int = 1,
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
    ...
```

The output is a dictionary containing, among other fields,

```python
{
    "t_nodes": t_nodes,
    "X": X,
    "P": P,
    "bundle": bundle,
    "rhobar": rho_bar_arr,
    "rbar": eta_time_local,
    "delta": delta,
    "log": log,
    "info": info,
    "problem": problem,
    "bundle_support_points": bundle_support_points,
    "settings": settings,
}
```

where `X` and `P` are the final state and costate arrays, `bundle` is the final
PA bundle, `delta` is the final smoothing parameter, and `log` stores the
per-iteration diagnostic history.

---

## 1) Mathematical objects controlled by this module

Let the optimal control problem be written in Bolza form as

$$
\begin{equation}
\begin{aligned}
\min_{u(\cdot)} \quad
& g(x(T)) + \int_0^T \ell(x(t),u(t),t)\,dt, \\
\dot x(t) \quad
&= f(x(t),u(t),t), \\
x(0) \quad
&= x_0.
\end{aligned}
\end{equation}
$$

The Hamiltonian convention used in this repository is the **minimum convention**:

$$
\begin{equation}
H(p,x,t)
=
\min_{u \in A}
\left\{
p^\top f(x,u,t) + \ell(x,u,t)
\right\}.
\end{equation}
$$

If state constraints or local viability restrictions are active, the code may
instead evaluate a restricted Hamiltonian, where admissible controls are further
filtered by local feasibility tests.

The PA-bundle surrogate restricts the minimization to a finite set of controls
stored in the bundle:

$$
\begin{equation}
\bar H(p,x,t)
=
\min_{u \in U_{\mathrm{bundle}}}
\left\{
p^\top f(x,u,t) + \ell(x,u,t)
\right\}.
\end{equation}
$$

Since $U_{\mathrm{bundle}} \subset A$, the PA surrogate satisfies

$$
\begin{equation}
\bar H(p,x,t) \ge H(p,x,t)
\end{equation}
$$

whenever both Hamiltonians use the same feasibility convention.

The smoothed Hamiltonian $H_\delta$ is evaluated through
`core/smoothing.py`. For a bundle with controls $u_1,\dots,u_M$, write

$$
\begin{equation}
h_j(p,x,t)
=
p^\top f(x,u_j,t) + \ell(x,u_j,t).
\end{equation}
$$

The smoothed soft-min surrogate is

$$
\begin{equation}
H_\delta(p,x,t)
=
-\delta
\log
\left(
\sum_{j=1}^M \exp\left(-\frac{h_j(p,x,t)}{\delta}\right)
\right).
\end{equation}
$$

This is the differentiable Hamiltonian used by the inner TPBVP solve unless the
problem supplies explicit Hamiltonian-gradient callbacks.

---

## 2) Discrete variables and mesh notation

The time mesh is stored as

```python
t_nodes = np.array([t_0, t_1, ..., t_N])
```

with

$$
\begin{equation}
\Delta t_i = t_{i+1} - t_i,
\qquad
i = 0,\dots,N-1.
\end{equation}
$$

The inner TPBVP solve returns arrays

```python
X.shape == (N + 1, n)
P.shape == (N + 1, n)
```

where

$$
\begin{equation}
X_i \approx x(t_i),
\qquad
P_i \approx p(t_i).
\end{equation}
$$

For nodewise Hamiltonian evaluations that require a local step size, the helper

```python
_local_dt_at_node(t_nodes, idx)
```

returns

$$
\begin{equation}
\Delta t_i
=
t_{i+1}-t_i
\end{equation}
$$

for non-terminal nodes, and the final interval length $t_N-t_{N-1}$ at the last
node.

---

## 3) Outer-loop structure

The adaptive solve proceeds as follows.

```python
t_nodes = initial_mesh.copy()
bundle = PABundle()
delta = delta0

for k in range(max_iters):
    X, P, info = solve_tpbvp(...)

    if k == 0 and not use_explicit_hamiltonian_gradients:
        bootstrap_bundle_from_trajectory(...)
        if new controls were added:
            X, P, info = solve_tpbvp(...)

    compute time indicator
    compute PA indicator, unless explicit Hamiltonian-gradient mode is active
    compute smoothing indicator, unless explicit Hamiltonian-gradient mode is active

    action = choose_adaptive_action(...)

    if action == "STOP":
        break
    elif action starts with "refine_time":
        refine marked intervals
    elif action == "add_plane":
        add selected controls to the PA bundle
    elif action == "delta*=0.5":
        delta *= 0.5
```

There is also a feasibility-refinement branch for problems that provide local
step-feasibility callbacks. This branch can refine intervals before the usual
indicator-based refinement logic is applied.

---

## 4) Indicator tolerances and consistency check

Before the adaptive loop starts, the routine calls

```python
validate_indicator_tolerances(tol_time, tol_PA, tol_delta)
```

The current implementation enforces:

1. all three tolerances must be strictly positive,
2. `max_relative_factor` must be at least one,
3. `tol_PA` cannot be too large relative to `tol_time`,
4. `tol_delta` cannot be too large relative to `tol_time`.

With the default `max_relative_factor = 2.0`, the admissibility conditions are

$$
\begin{equation}
\mathrm{tol}_{PA}
\le
2\,\mathrm{tol}_{time},
\qquad
\mathrm{tol}_{\delta}
\le
2\,\mathrm{tol}_{time}.
\end{equation}
$$

This is a practical safeguard: if one tolerance is much looser than the others,
then the adaptive decision logic may stop improving one source of error long
before the other indicators have become meaningfully balanced.

---

## 5) Time discretization indicator

On each interval $[t_i,t_{i+1}]$, the code evaluates Hamiltonian gradients at
the symplectic-Euler point

$$
\begin{equation}
(P_{i+1}, X_i, t_i).
\end{equation}
$$

Let

$$
\begin{equation}
H_p
=
\frac{\partial H_\delta}{\partial p}(P_{i+1},X_i,t_i),
\qquad
H_x
=
\frac{\partial H_\delta}{\partial x}(P_{i+1},X_i,t_i).
\end{equation}
$$

The raw local density used by the code is

$$
\begin{equation}
\rho_i
=
-\frac{1}{2} H_p^\top H_x.
\end{equation}
$$

To avoid vanishing indicators on large intervals, the implementation introduces
a floor depending on the largest current time step:

$$
\begin{equation}
\mathrm{floor}
=
K_{\mathrm{time}} \sqrt{\max_i \Delta t_i}.
\end{equation}
$$

Then

$$
\begin{equation}
\bar \rho_i
=
\max\left(|\rho_i|,\mathrm{floor}\right).
\end{equation}
$$

and the local time indicator is

$$
\begin{equation}
\bar r_i
=
|\bar \rho_i|(\Delta t_i)^2.
\end{equation}
$$

The scalar time indicator used for stopping and refinement decisions is

$$
\begin{equation}
\eta_{\mathrm{time}}
=
\max_i \bar r_i.
\end{equation}
$$

The code also records the summed indicator

$$
\begin{equation}
\eta_{\mathrm{time,sum}}
=
\sum_i \bar r_i.
\end{equation}
$$

The per-interval stopping target is

$$
\begin{equation}
\mathrm{tol}_{time}^{*}
=
\frac{\mathrm{tol}_{time}}{N}.
\end{equation}
$$

An interval is marked for time refinement if

$$
\begin{equation}
\bar r_i
>
s_{\mathrm{time}}
\frac{\mathrm{tol}_{time}}{N}.
\end{equation}
$$

Here `s_time` is a marking parameter. The default is `s_time = 0.5`.

---

## 6) PA-bundle indicator

Unless explicit Hamiltonian-gradient mode is enabled, the code estimates the PA
surrogate error by comparing the finite-bundle Hamiltonian $\bar H$ with the
richer Hamiltonian evaluation $H$ computed by `compute_H`.

At a node $t_i$, define

$$
\begin{equation}
e^{PA}_i
=
\bar H(P_i,X_i,t_i)
-
H(P_i,X_i,t_i).
\end{equation}
$$

The local interval contribution is computed with a trapezoidal rule:

$$
\begin{equation}
\eta^{PA}_i
=
\frac{1}{2}
\left(
e^{PA}_i + e^{PA}_{i+1}
\right)
\Delta t_i.
\end{equation}
$$

The global PA indicator is

$$
\begin{equation}
\eta_{PA}
=
\sum_{i=0}^{N-1} \eta^{PA}_i.
\end{equation}
$$

When the PA indicator is too large, the adaptive action may be `"add_plane"`.
In that case, the code selects a small number of high-priority nodes and adds
the corresponding locally minimizing controls to the bundle.

---

## 7) Smoothing indicator

Unless explicit Hamiltonian-gradient mode is enabled, the code also estimates
the smoothing error by comparing the PA Hamiltonian $\bar H$ and the smoothed
Hamiltonian $H_\delta$.

At a node $t_i$, define

$$
\begin{equation}
e^\delta_i
=
\bar H(P_i,X_i,t_i)
-
H_\delta(P_i,X_i,t_i).
\end{equation}
$$

The local smoothing contribution is

$$
\begin{equation}
\eta^\delta_i
=
\frac{1}{2}
\left(
e^\delta_i + e^\delta_{i+1}
\right)
\Delta t_i.
\end{equation}
$$

The global smoothing indicator is

$$
\begin{equation}
\eta_\delta
=
\sum_{i=0}^{N-1} \eta^\delta_i.
\end{equation}
$$

If this indicator is too large, and if the action policy chooses smoothing
continuation, the code updates

$$
\begin{equation}
\delta \leftarrow \frac{1}{2}\delta.
\end{equation}
$$

---

## 8) Explicit Hamiltonian-gradient mode

If

```python
use_explicit_hamiltonian_gradients=True
```

then the adaptive logic enters an explicit-gradient mode. In this mode:

1. the time indicator is still computed,
2. PA-bundle and smoothing indicators are not used to drive decisions,
3. the adaptive action is either `"STOP"` or time refinement.

This mode is useful for examples where the problem object supplies direct
Hamiltonian-gradient formulas and the PA/smoothing machinery is not meant to be
the dominant approximation model.

---

## 9) Adaptive action policy

The action selection is centralized in

```python
choose_adaptive_action(
    eta_time,
    tol_time_star,
    eta_PA,
    tol_PA,
    eta_delta,
    tol_delta,
    n_marked,
    explicit_mode=False,
    time_balance_ratio=0.1,
)
```

In ordinary PA/smoothing mode, the stopping condition is

$$
\begin{equation}
\eta_{\mathrm{time}}
\le
\mathrm{tol}_{time}^{*},
\qquad
\eta_{PA}
\le
\mathrm{tol}_{PA},
\qquad
\eta_{\delta}
\le
\mathrm{tol}_{\delta}.
\end{equation}
$$

If all three inequalities hold, the action is `"STOP"`.

Otherwise, the code compares the time indicator with the dominant non-time
indicator

$$
\begin{equation}
\eta_{\mathrm{non-time}}
=
\max(\eta_{PA},\eta_{\delta}).
\end{equation}
$$

Time refinement may be suppressed when

$$
\begin{equation}
\eta_{\mathrm{time}}
\le
\mathrm{time\_balance\_ratio}
\,
\eta_{\mathrm{non-time}}.
\end{equation}
$$

This prevents repeated time refinement when the dominant error source is really
the PA approximation or smoothing approximation.

The action priority is:

1. refine time, if the time indicator is too large and time refinement is not suppressed;
2. if time refinement is suppressed, reduce $\delta$ when the smoothing indicator dominates;
3. if time refinement is suppressed, add a PA plane when the PA indicator dominates;
4. otherwise, add a PA plane if $\eta_{PA}$ is too large;
5. otherwise, halve $\delta$ if $\eta_\delta$ is too large;
6. otherwise, refine time if the time indicator is still too large.

In explicit-gradient mode, the policy is simpler:

$$
\begin{equation}
\text{action}
=
\begin{cases}
\texttt{"STOP"}, & \eta_{\mathrm{time}} \le \mathrm{tol}_{time}^{*}, \\
\texttt{"refine\_time"}, & \eta_{\mathrm{time}} > \mathrm{tol}_{time}^{*}.
\end{cases}
\end{equation}
$$

---

## 10) Per-iteration log entries

Each outer iteration appends a dictionary to `log`. The most important fields
are:

```python
{
    "iteration": k,
    "N": N,
    "M": bundle.num_planes(),
    "delta": delta,
    "eta_time": eta_time,
    "eta_time_sum": eta_time_sum,
    "eta_PA": eta_PA,
    "eta_delta": eta_delta,
    "rho": rho_arr.copy(),
    "rho_bar": rho_bar_arr.copy(),
    "r_bar": eta_time_local.copy(),
    "eta_PA_local": eta_PA_local.copy(),
    "eta_delta_local": eta_delta_local.copy(),
    "pa_gap_nodes": pa_gap_nodes.copy(),
    "delta_gap_nodes": delta_gap_nodes.copy(),
    "active_plane_idx_nodes": active_plane_idx_nodes.copy(),
    "tol_time_star": tol_time_star,
    "mark_thr": mark_thr,
    "newton_iter": info["iterations"],
    "newton_residual": info["residual_norm"],
    "solver_phase": info.get("solver_phase", "newton"),
    "fallback_used": bool(info.get("fallback_used", False)),
    "objective_mesh_approx": objective_mesh_approx,
    "all_indicators_within_tolerance": all_indicators_within_tolerance,
    "action": action,
}
```

If `store_iterates=True`, the log also stores copies of the current iterates:

```python
entry["X_iter"] = X.copy()
entry["P_iter"] = P.copy()
entry["U_iter"] = U.copy()
```

and a history of PA-bundle support points accumulated so far.

---

## 11) Main helper routines in this file

The current file contains the following helper routines:

| Routine | Purpose |
|---|---|
| `_local_dt_at_node` | Returns a local time-step length associated with a node. |
| `choose_adaptive_action` | Chooses `"STOP"`, time refinement, PA enrichment, or smoothing continuation. |
| `validate_indicator_tolerances` | Checks positivity and relative consistency of the three indicator tolerances. |
| `_grads_for_indicators` | Returns Hamiltonian gradients either from explicit callbacks or from the smoothed Hamiltonian. |
| `_compute_node_controls` | Computes nodewise minimizing controls using `compute_H`. |
| `_supports_need_dt_refresh` | Detects whether controls depend on local time-step information and therefore need bundle refresh after mesh changes. |
| `_refresh_bundle_support_controls` | Rebuilds support controls after a mesh change for problems with local step-feasibility or local oracle controls. |
| `_mesh_objective` | Computes a mesh-based objective approximation. |
| `_find_feasibility_refinement_intervals` | Marks intervals that require refinement for feasibility reasons. |
| `_refine_selected_intervals` | Refines a selected set of intervals by midpoint insertion. |
| `compute_pa_ranking_scores` | Converts nodewise PA gaps into time-weighted PA enrichment scores. |
| `select_pa_enrichment_candidates` | Selects separated high-score nodes for PA enrichment. |
| `bootstrap_bundle_from_trajectory` | Adds initial active controls to the bundle after the first coarse solve. |
| `solve_optimal_control` | Runs the full adaptive solve. |

---

## 12) Internal helper routines in `adaptivity.py`

The adaptive loop in `solve_optimal_control` is supported by several helper
routines defined in the same file. These helpers are part of the adaptive
algorithm itself, so we document them here in detail.

The external routines called by these helpers, such as `compute_H`,
`eval_H_smooth`, `PABundle.evaluate`, and `solve_tpbvp`, are documented in their
own files.

---

### 12.1 `_local_dt_at_node`

```python
def _local_dt_at_node(t_nodes: np.ndarray, idx: int):
    ...
```

This helper assigns a representative local time step to a mesh node. It is used
whenever a nodal Hamiltonian evaluation needs a local step size, for example in
step-feasibility checks or local oracle controls.

For an interior or non-terminal node $i<N$, the routine returns the forward
step

$$
\begin{equation}
\Delta t_i^{node}
=
t_{i+1}-t_i.
\end{equation}
$$

For the terminal node $i=N$, there is no forward step, so the routine returns
the last interval size

$$
\begin{equation}
\Delta t_N^{node}
=
t_N-t_{N-1}.
\end{equation}
$$

If the mesh has one node or fewer, the routine returns `None`.

This convention lets nodal routines use a consistent local step size without
having to treat the terminal node separately.

---

### 12.2 `choose_adaptive_action`

```python
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
    ...
```

This helper decides the next outer-loop action from the three adaptive
indicators:

- time-discretization indicator $\eta_{\mathrm{time}}$,
- PA-bundle indicator $\eta_{PA}$,
- smoothing indicator $\eta_\delta$.

The time indicator is compared against the per-interval target

$$
\begin{equation}
\mathrm{tol}_{time}^{*}
=
\frac{\mathrm{tol}_{time}}{N}.
\end{equation}
$$

The PA and smoothing indicators are compared against their global tolerances
$\mathrm{tol}_{PA}$ and $\mathrm{tol}_\delta$.

---

#### Explicit-gradient mode

If `explicit_mode=True`, the PA and smoothing indicators are ignored by the
action policy. The action is

$$
\begin{equation}
\mathrm{action}
=
\begin{cases}
\texttt{"STOP"},
&
\eta_{\mathrm{time}}
\le
\mathrm{tol}_{time}^{*},
\\
\texttt{"refine\_time(marked=n\_marked)"},
&
\eta_{\mathrm{time}}
>
\mathrm{tol}_{time}^{*}.
\end{cases}
\end{equation}
$$

This mode is used when the problem supplies explicit Hamiltonian gradients and
the PA/smoothing approximation is not intended to control the adaptive decision.

---

#### Full PA/smoothing mode

In the standard mode, the stopping criterion is

$$
\begin{equation}
\eta_{\mathrm{time}}
\le
\mathrm{tol}_{time}^{*},
\qquad
\eta_{PA}
\le
\mathrm{tol}_{PA},
\qquad
\eta_{\delta}
\le
\mathrm{tol}_{\delta}.
\end{equation}
$$

If all three inequalities hold, the action is `"STOP"`.

If not, the routine computes the dominant non-time indicator

$$
\begin{equation}
\eta_{\mathrm{non-time}}
=
\max(\eta_{PA},\eta_\delta).
\end{equation}
$$

Time refinement is suppressed when

$$
\begin{equation}
\eta_{\mathrm{time}}
\le
\mathrm{time\_balance\_ratio}
\,
\eta_{\mathrm{non-time}}.
\end{equation}
$$

This rule prevents the algorithm from repeatedly refining the mesh when the
main unresolved error source is the PA approximation or smoothing error.

The decision priority is:

1. if time error is too large and not suppressed, refine the marked intervals;
2. if time refinement is suppressed and $\eta_\delta \ge \eta_{PA}$, halve
   $\delta$;
3. if time refinement is suppressed and $\eta_{PA}$ dominates, add a PA plane;
4. otherwise, if $\eta_{PA}$ is too large, add a PA plane;
5. otherwise, if $\eta_\delta$ is too large, halve $\delta$;
6. otherwise, if time error is still too large, refine time.

The output is a string such as

```python
"STOP"
"refine_time(marked=...)"
"add_plane"
"delta*=0.5"
```

which is interpreted later in the main adaptive loop.

---

### 12.3 `validate_indicator_tolerances`

```python
def validate_indicator_tolerances(
    tol_time: float,
    tol_PA: float,
    tol_delta: float,
    *,
    max_relative_factor: float = 2.0,
):
    ...
```

This helper checks that the user-supplied indicator tolerances are compatible
with the adaptive logic.

First, all three tolerances must be positive:

$$
\begin{equation}
\mathrm{tol}_{time}>0,
\qquad
\mathrm{tol}_{PA}>0,
\qquad
\mathrm{tol}_{\delta}>0.
\end{equation}
$$

Second, `max_relative_factor` must satisfy

$$
\begin{equation}
\mathrm{max\_relative\_factor} \ge 1.
\end{equation}
$$

Third, the PA and smoothing tolerances are not allowed to be too large relative
to the time tolerance:

$$
\begin{equation}
\mathrm{tol}_{PA}
\le
\mathrm{max\_relative\_factor}\,\mathrm{tol}_{time},
\qquad
\mathrm{tol}_{\delta}
\le
\mathrm{max\_relative\_factor}\,\mathrm{tol}_{time}.
\end{equation}
$$

With the default value `max_relative_factor=2.0`, this becomes

$$
\begin{equation}
\mathrm{tol}_{PA}
\le
2\,\mathrm{tol}_{time},
\qquad
\mathrm{tol}_{\delta}
\le
2\,\mathrm{tol}_{time}.
\end{equation}
$$

This check is a practical safeguard. If one error source is assigned a much
looser tolerance than the others, the adaptive loop may stop improving that
source too early, producing an unbalanced approximation.

---

### 12.4 `_grads_for_indicators`

```python
def _grads_for_indicators(
    problem,
    bundle,
    p,
    x,
    t,
    delta,
    dt=None,
    use_explicit_hamiltonian_gradients=False,
):
    ...
```

This helper returns the Hamiltonian derivatives used by the time-discretization
indicator. It abstracts over two possible ways of obtaining $H_p$ and $H_x$.

---

#### Smoothed PA mode

By default, the gradients are computed from the smoothed PA Hamiltonian:

$$
\begin{equation}
(H_\delta,H_p,H_x)
=
\mathrm{eval\_H\_smooth}
(
problem,bundle,p,x,t,\delta
).
\end{equation}
$$

Thus

$$
\begin{equation}
H_p
=
\frac{\partial H_\delta}{\partial p}(p,x,t),
\qquad
H_x
=
\frac{\partial H_\delta}{\partial x}(p,x,t).
\end{equation}
$$

The returned tuple is

```python
H_delta, Hp, Hx
```

where `H_delta` is the smoothed Hamiltonian value.

---

#### Explicit-gradient mode

If `use_explicit_hamiltonian_gradients=True` and the problem provides
`problem.hamiltonian_grad_fn`, the routine calls

```python
Hp, Hx = problem.hamiltonian_gradients(x, p, t)
```

and returns

```python
None, Hp, Hx
```

In this mode the Hamiltonian value is not needed for the time indicator, so the
first returned value is `None`.

This is useful when the exact Hamiltonian gradients are known analytically. In
that case, the time indicator can be computed without using the PA/smoothing
gradient approximation.

---

### 12.5 `_compute_node_controls`

```python
def _compute_node_controls(
    problem,
    bundle,
    X,
    P,
    t_nodes,
    restricted=True,
    use_oracle=False,
):
    ...
```

This helper reconstructs a nodal control trajectory from the computed state and
costate arrays.

At each node $t_i$, it calls `compute_H` to solve the local Hamiltonian
minimization

$$
\begin{equation}
u_i^\star
\in
\arg\min_u
\left\{
P_i^\top f(X_i,u,t_i)
+
\ell(X_i,u,t_i)
\right\}.
\end{equation}
$$

The list `bundle.controls` is passed to `compute_H` as a source of additional
candidate controls. If `restricted=True`, only controls that pass the local
feasibility checks are considered.

The helper also passes a local step size

$$
\begin{equation}
\Delta t_i^{node}
=
\_local\_dt\_at\_node(t\_nodes,i)
\end{equation}
$$

because some problem-specific admissibility tests depend on the time step.

The output is an array

```python
U.shape == (N + 1, m)
```

when controls exist. If the mesh is empty, it returns an empty array with
compatible control dimension.

If no admissible control is found at a node, the routine raises a
`RuntimeError`. This is appropriate because the solver cannot report a
consistent nodal control trajectory if the local Hamiltonian minimization is
empty.

---

### 12.6 `_supports_need_dt_refresh`

```python
def _supports_need_dt_refresh(problem) -> bool:
    ...
```

This helper detects whether the PA bundle may need to be refreshed after mesh
changes.

It returns `True` when the problem has either of the following callbacks:

```python
problem.step_feasible_control_fn
problem.u_star_local_fn
```

The reason is that these callbacks may depend on the local step size
$\Delta t_i$. After mesh refinement, the old support controls may no longer be
consistent with the new local step sizes. In that case, the bundle should be
rebuilt or refreshed using the new mesh.

---

### 12.7 `_refresh_bundle_support_controls`

```python
def _refresh_bundle_support_controls(
    problem,
    t_nodes,
    X_guess,
    P_guess,
    support_points,
    use_oracle=False,
):
    ...
```

This helper rebuilds the PA bundle after a mesh change for problems whose
controls may depend on the local time step.

If `_supports_need_dt_refresh(problem)` is false, the routine returns `None`.
This tells the caller that no refresh is needed.

If a refresh is needed, the routine first creates a new `PABundle`. It seeds the
bundle with basic controls:

- if bounds are available, it adds the midpoint, lower-bound, and upper-bound
  controls;
- if bounds are absent but the control dimension is known, it adds the zero
  control.

When bounds are available, the initial seed controls are

$$
\begin{equation}
u_{\mathrm{mid}}
=
\frac{1}{2}(u_{\min}+u_{\max}),
\qquad
u_{\min},
\qquad
u_{\max}.
\end{equation}
$$

Then the routine loops over the stored `support_points`. Each support point
contains information such as the old support time, state, costate, and control.
The old support time is projected to the closest node of the new mesh:

$$
\begin{equation}
i^\star
=
\arg\min_i
|t_i - t_{\mathrm{support}}|.
\end{equation}
$$

At the new node, the routine recomputes a local Hamiltonian minimizer:

$$
\begin{equation}
u_{i^\star}^\star
\in
\arg\min_u
\left\{
P_{i^\star}^\top f(X_{i^\star},u,t_{i^\star})
+
\ell(X_{i^\star},u,t_{i^\star})
\right\}.
\end{equation}
$$

The recomputed control is added to the refreshed bundle, and the corresponding
support-point record is updated with the new node index, time, state, costate,
local step size, control, and bundle size.

This mechanism is important for problems where the feasible set depends on
$\Delta t_i$, because mesh refinement changes the local feasibility geometry.

---

### 12.8 `_mesh_objective`

```python
def _mesh_objective(problem, t_nodes, X, controls):
    ...
```

This helper computes a first-order approximation of the Bolza objective on the
current mesh.

It starts with the terminal cost

$$
\begin{equation}
g(X_N),
\end{equation}
$$

and adds a left-endpoint quadrature approximation of the running cost:

$$
\begin{equation}
J_h
=
g(X_N)
+
\sum_{i=0}^{N-1}
\ell(X_i,U_i,t_i)\Delta t_i.
\end{equation}
$$

The value is stored in the iteration log as `objective_mesh_approx`. It is a
diagnostic quantity and is not used directly as the adaptive stopping criterion.

---

### 12.9 `_local_control_scale`

```python
def _local_control_scale(problem) -> float:
    ...
```

This helper defines a normalization scale for comparing controls in the
feasibility-refinement logic.

If control bounds are available, the scale is the diameter of the control box:

$$
\begin{equation}
\mathrm{control\_scale}
=
\|u_{\max}-u_{\min}\|.
\end{equation}
$$

If this norm is zero, the function returns $1$ to avoid division by zero.

If no bounds are available, the routine uses the control dimension $m$ and
returns

$$
\begin{equation}
\mathrm{control\_scale}
=
\max(\sqrt{m},1).
\end{equation}
$$

If the control dimension is unknown, it defaults to $m=1$.

This scale is used to convert absolute control changes into relative changes.

---

### 12.10 `_find_feasibility_refinement_intervals`

```python
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
    ...
```

This helper detects intervals that should be refined for feasibility reasons,
independently of the usual time, PA, and smoothing indicators.

It returns

```python
marked_intervals, issues
```

where:

- `marked_intervals` is a sorted list of interval indices,
- `issues` is a list of diagnostic dictionaries explaining why each interval
  was marked.

The routine loops over every node $t_i$ and performs the checks below.

---

#### 12.10.1 Empty admissible set at the current step size

At node $i$, the routine first computes the local restricted Hamiltonian
minimizer using the current local time step $\Delta t_i^{node}$:

$$
\begin{equation}
u_i^\star(\Delta t_i)
\in
\arg\min_u
\left\{
P_i^\top f(X_i,u,t_i)
+
\ell(X_i,u,t_i)
\right\}.
\end{equation}
$$

If `compute_H` returns no admissible control, the corresponding interval is
marked. The issue record receives the reason

```python
"empty_admissible_set"
```

For a terminal node, the code marks the last available interval.

---

#### 12.10.2 Problem-specific feasibility-refinement callback

If the problem provides

```python
problem.feasibility_refinement_fn
```

and the local step size is positive, the routine calls this callback with the
current state, costate, time, local step size, and a small tolerance.

If the callback returns a non-`None` object, the corresponding interval is
marked. The issue record receives the reason

```python
"problem_feasibility_refinement"
```

If the callback returns a dictionary, its contents are merged into the issue
record. This allows problem-specific diagnostics to be propagated into the
adaptive log.

---

#### 12.10.3 Probe-step admissibility

If the previous checks pass, the routine probes the local Hamiltonian
minimization with a shorter step

$$
\begin{equation}
\Delta t_i^{probe}
=
\mathrm{probe\_ratio}\,\Delta t_i.
\end{equation}
$$

It computes another local minimizer

$$
\begin{equation}
u_i^\star(\Delta t_i^{probe})
\in
\arg\min_u
\left\{
P_i^\top f(X_i,u,t_i)
+
\ell(X_i,u,t_i)
\right\}.
\end{equation}
$$

If no admissible control is found for the probe step, the interval is marked
with reason

```python
"empty_probe_admissible_set"
```

---

#### 12.10.4 Step-size sensitivity of the local minimizer

If both the current-step and probe-step controls exist, the routine computes

$$
\begin{equation}
\mathrm{rel\_change}
=
\frac{
\left\|
u_i^\star(\Delta t_i)
-
u_i^\star(\Delta t_i^{probe})
\right\|
}{
\mathrm{control\_scale}
}.
\end{equation}
$$

If

$$
\begin{equation}
\mathrm{rel\_change}
>
\mathrm{control\_sensitivity},
\end{equation}
$$

then the interval is marked with reason

```python
"oracle_dt_sensitivity"
```

This detects regions where the locally minimizing feasible control changes
significantly when the time step is reduced. Such behavior suggests that the
mesh is too coarse for the local feasibility structure.

---

### 12.11 `_refine_selected_intervals`

```python
def _refine_selected_intervals(t_nodes, X, P, marked_intervals):
    ...
```

This helper refines a selected set of intervals by midpoint insertion.

For a marked interval $[t_i,t_{i+1}]$, the inserted node is

$$
\begin{equation}
t_{i+1/2}
=
\frac{1}{2}(t_i+t_{i+1}).
\end{equation}
$$

The state and costate guesses at the midpoint are obtained by linear
interpolation. With

$$
\begin{equation}
\alpha
=
\frac{t_{i+1/2}-t_i}{t_{i+1}-t_i},
\end{equation}
$$

the midpoint guesses are

$$
\begin{equation}
X_{i+1/2}
=
(1-\alpha)X_i+\alpha X_{i+1},
\qquad
P_{i+1/2}
=
(1-\alpha)P_i+\alpha P_{i+1}.
\end{equation}
$$

For midpoint refinement, $\alpha=1/2$. The returned arrays are used as warm
starts for the next TPBVP solve.

---

### 12.12 `_local_time_radius`

```python
def _local_time_radius(t_nodes: np.ndarray, idx: int, separation_factor: float) -> float:
    ...
```

This helper defines a local exclusion radius around a mesh node. It is used
when selecting PA-enrichment points, so that the algorithm does not choose many
nearly identical support points in the same small time neighborhood.

For the first node, the local time scale is the first interval length:

$$
\begin{equation}
\Delta t_0^{local}
=
t_1-t_0.
\end{equation}
$$

For the terminal node, it is the last interval length:

$$
\begin{equation}
\Delta t_N^{local}
=
t_N-t_{N-1}.
\end{equation}
$$

For an interior node, it is the larger of the adjacent interval lengths:

$$
\begin{equation}
\Delta t_i^{local}
=
\max(t_i-t_{i-1},\,t_{i+1}-t_i).
\end{equation}
$$

The exclusion radius is then

$$
\begin{equation}
R_i
=
\mathrm{separation\_factor}
\,
\max(\Delta t_i^{local},0).
\end{equation}
$$

If the mesh has one node or fewer, the radius is zero.

---

### 12.13 `compute_pa_ranking_scores`

```python
def compute_pa_ranking_scores(
    t_nodes: np.ndarray,
    pa_gaps: np.ndarray,
):
    ...
```

This helper converts nodewise PA Hamiltonian gaps into time-weighted scores used
for PA-bundle enrichment.

The input `pa_gaps` stores values of the form

$$
\begin{equation}
e_i^{PA}
=
\bar H(P_i,X_i,t_i)
-
H(P_i,X_i,t_i).
\end{equation}
$$

The routine first assigns a representative node time step. For non-terminal
nodes,

$$
\begin{equation}
\Delta t_i^{node}
=
t_{i+1}-t_i,
\qquad
i=0,\dots,N-1.
\end{equation}
$$

For the terminal node,

$$
\begin{equation}
\Delta t_N^{node}
=
t_N-t_{N-1}.
\end{equation}
$$

The PA ranking score is

$$
\begin{equation}
s_i^{PA}
=
e_i^{PA}\Delta t_i^{node}.
\end{equation}
$$

The routine returns both arrays:

```python
node_dt, pa_scores
```

This weighting favors nodes where the PA Hamiltonian gap is large and where the
associated time interval is significant.

---

### 12.14 `select_pa_enrichment_candidates`

```python
def select_pa_enrichment_candidates(
    t_nodes: np.ndarray,
    ranking_scores: np.ndarray,
    target_count: int,
    *,
    separation_factor: float = 5.0,
    gap_floor_ratio: float = 0.2,
):
    ...
```

This helper selects the nodes where new controls should be added to the PA
bundle.

The input `ranking_scores` is typically the `pa_scores` array from
`compute_pa_ranking_scores`.

The routine returns

```python
selected_indices, selection_metadata
```

where `selected_indices` is a list of node indices and `selection_metadata`
contains diagnostic values such as the maximum score, score floor, and rejection
counts.

---

#### 12.14.1 Score floor

The maximum score is

$$
\begin{equation}
s_{\max}
=
\max_i s_i^{PA}.
\end{equation}
$$

The selection floor is

$$
\begin{equation}
s_{\min}
=
\mathrm{gap\_floor\_ratio}\,s_{\max}.
\end{equation}
$$

A candidate node is rejected if

$$
\begin{equation}
s_i^{PA}
<
s_{\min}.
\end{equation}
$$

This prevents the algorithm from adding PA planes at nodes whose PA contribution
is small relative to the dominant PA gap.

If all scores are nonpositive or not finite, no candidates are selected.

---

#### 12.14.2 Time separation

The remaining candidates are considered in decreasing score order. A candidate
node $i$ is compared against each already-selected node $j$.

Using `_local_time_radius`, the code computes local exclusion radii $R_i$ and
$R_j$. The candidate is rejected if

$$
\begin{equation}
|t_i-t_j|
\le
\max(R_i,R_j).
\end{equation}
$$

This keeps the selected PA-enrichment points separated in time. The goal is to
improve the PA approximation in distinct regions of the trajectory instead of
adding several nearly duplicate controls near the same time.

The selection stops once `target_count` nodes have been accepted.

---

### 12.15 `bootstrap_bundle_from_trajectory`

```python
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
    ...
```

This helper enriches the PA bundle after the first coarse TPBVP solve. Its
purpose is to insert a small number of controls that are already relevant along
the first computed trajectory.

The function selects representative support nodes by uniformly sampling indices
between $0$ and $N$:

$$
\begin{equation}
i_1,\dots,i_q
\subset
\{0,\dots,N\},
\qquad
q
=
\min(\mathrm{num\_support\_nodes},N+1).
\end{equation}
$$

At each selected node, it tries to add a control approximating the local
Hamiltonian minimizer

$$
\begin{equation}
u_i^\star
\in
\arg\min_u
\left\{
P_i^\top f(X_i,u,t_i)
+
\ell(X_i,u,t_i)
\right\}.
\end{equation}
$$

The current implementation has two stages.

---

#### 12.15.1 Oracle bootstrap

If `use_oracle=True`, the routine first calls

```python
u_oracle, ok = problem.u_star(
    x_i,
    p_i,
    t_i,
    restricted=restricted,
    dt=dt_i,
)
```

If the oracle returns a control and the control is feasible under the requested
restriction, the control is added to the PA bundle.

If the control is new, the routine increments its `added` counter and optionally
records the support point in `support_log`.

---

#### 12.15.2 Scalar grid-search fallback

If the oracle is not used, or if it does not provide a feasible control at a
node, the routine falls back to a small grid search when scalar bounds are
available.

If the control bounds are

$$
\begin{equation}
u_{\min}
\le
u
\le
u_{\max},
\end{equation}
$$

and the control is scalar, it builds a grid

$$
\begin{equation}
u^{(1)},\dots,u^{(G)}
\subset
[u_{\min},u_{\max}],
\end{equation}
$$

where $G=\mathrm{grid\_size}$.

For each grid value, the code checks local feasibility and evaluates

$$
\begin{equation}
h(u^{(j)})
=
P_i^\top f(X_i,u^{(j)},t_i)
+
\ell(X_i,u^{(j)},t_i).
\end{equation}
$$

The best feasible grid control is

$$
\begin{equation}
u_i^{grid}
=
\arg\min_{u^{(j)}}
h(u^{(j)}).
\end{equation}
$$

If such a control exists and is not already present in the bundle, it is added.

---

#### 12.15.3 Support log

When `support_log` is not `None`, every newly added bootstrap control records a
dictionary containing:

```python
{
    "iteration": iteration,
    "kind": "bootstrap",
    "node_index": i,
    "time": t_i,
    "state": x_i,
    "costate": p_i,
    "control": added_control,
    "local_dt": dt_i,
    "bundle_size_after": bundle.num_planes(),
}
```

These records are later useful when generating reports or explaining how the PA
bundle was enriched.

The routine returns the number of new controls actually added to the bundle.
Duplicate controls are ignored by `bundle.add_control`, so the return value can
be smaller than the number of sampled support nodes.

---

## 13) Main routine: `solve_optimal_control`

The main adaptive driver is

```python
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
    verbose: bool = True,
    print_every: int = 1,
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
    ...
```

This routine coordinates the full adaptive process:

1. initialize the mesh, PA bundle, smoothing parameter, and warm starts;
2. solve the smoothed discrete PMP TPBVP on the current mesh;
3. compute a posteriori indicators;
4. decide whether to stop, refine the mesh, enrich the PA bundle, or decrease
   the smoothing parameter;
5. repeat until the stopping criteria are met or `max_iters` is reached.

The routine returns a dictionary containing the final mesh, state trajectory,
costate trajectory, PA bundle, smoothing parameter, diagnostics, and run
settings.

---

### 13.1 Parameter groups

The argument list is long because the routine exposes several independent
parts of the adaptive algorithm. The parameters can be grouped as follows.

---

#### Problem and initial mesh

```python
problem
initial_mesh
```

The object `problem` is an `OCPProblem` instance. It provides dynamics, costs,
bounds, optional Hamiltonian or control oracles, and optional feasibility
callbacks.

The array `initial_mesh` contains the starting mesh

$$
\begin{equation}
0=t_0<t_1<\cdots<t_N=T.
\end{equation}
$$

The code assumes the input mesh is already sorted and uses a copy internally.

---

#### Indicator tolerances

```python
tol_time
tol_PA
tol_delta
```

These are the stopping tolerances for the three adaptive indicators:

$$
\begin{equation}
\eta_{\mathrm{time}},
\qquad
\eta_{PA},
\qquad
\eta_\delta.
\end{equation}
$$

The time tolerance is converted into the per-interval target

$$
\begin{equation}
\mathrm{tol}_{time}^{*}
=
\frac{\mathrm{tol}_{time}}{N}.
\end{equation}
$$

The PA and smoothing tolerances are used as global tolerances.

---

#### Outer-loop and smoothing parameters

```python
max_iters
delta0
```

The value `max_iters` limits the number of outer adaptive iterations.

The value `delta0` is the initial smoothing parameter:

$$
\begin{equation}
\delta^{(0)}=\delta_0.
\end{equation}
$$

Whenever the adaptive policy chooses smoothing continuation, the update is

$$
\begin{equation}
\delta
\leftarrow
\frac{1}{2}\delta.
\end{equation}
$$

---

#### Time-indicator parameters

```python
s_time
K_time
time_balance_ratio
```

The parameter `s_time` controls which intervals are marked for time refinement.
An interval is marked if

$$
\begin{equation}
\bar r_i
>
s_{\mathrm{time}}
\frac{\mathrm{tol}_{time}}{N}.
\end{equation}
$$

The parameter `K_time` defines the floor used in the time-indicator density:

$$
\begin{equation}
\mathrm{floor}
=
K_{\mathrm{time}}\sqrt{\max_i \Delta t_i}.
\end{equation}
$$

The parameter `time_balance_ratio` is used by `choose_adaptive_action` to avoid
over-refining time when the dominant error source is PA or smoothing error.

---

#### Newton solver parameters

```python
newton_tol
newton_max_iter
fallback_solver
```

These are passed to the inner TPBVP solver `solve_tpbvp`.

The inner solver tries to solve the nonlinear shooting system on the current
mesh, bundle, and smoothing parameter. The value `fallback_solver` controls
whether a fallback nonlinear least-squares phase may be used if Newton stalls.

The details of the Newton and fallback logic are documented in `newton.md`.

---

#### Logging and storage parameters

```python
verbose
print_every
log_path
store_iterates
```

If `verbose=True`, progress messages are printed during the solve.

If `log_path` is not `None`, the same messages are also written to that file.
The file is overwritten at the start of the run.

The parameter `print_every` controls how frequently one-line iteration summaries
are printed.

If `store_iterates=True`, each log entry stores copies of the state, costate,
control trajectory, and accumulated PA support-point information. This is useful
for report generation but can increase memory usage.

---

#### Oracle and explicit-gradient flags

```python
use_oracle_bootstrap
use_oracle_PA
use_explicit_hamiltonian_gradients
```

If `use_oracle_bootstrap=True`, the bootstrap routine is allowed to use
`problem.u_star(...)` when adding initial PA support controls.

If `use_oracle_PA=True`, PA enrichment and diagnostic Hamiltonian computations
are allowed to use the oracle control when available.

If `use_explicit_hamiltonian_gradients=True`, the adaptive loop uses explicit
Hamiltonian gradients supplied by the problem. In this mode, the PA and
smoothing indicators are disabled for adaptive decisions, and only the time
indicator drives refinement.

---

#### PA-enrichment parameters

```python
pa_add_fraction
pa_time_separation_factor
pa_gap_floor_ratio
```

When the selected action is `"add_plane"`, the algorithm does not necessarily
add only one plane. It sets a target number of new candidate nodes by

$$
\begin{equation}
N_{\mathrm{add}}
=
\max
\left(
1,
\left\lceil
\mathrm{pa\_add\_fraction}
\cdot
\max(M,1)
\right\rceil
\right),
\end{equation}
$$

where $M$ is the current number of bundle planes.

The parameters `pa_time_separation_factor` and `pa_gap_floor_ratio` control the
selection of separated high-score nodes for PA enrichment.

---

#### Feasibility-refinement parameters

```python
feasibility_probe_ratio
feasibility_control_sensitivity
```

These parameters are used only for problems with step-feasibility logic.

The probe ratio defines a shorter local step

$$
\begin{equation}
\Delta t_i^{probe}
=
\mathrm{feasibility\_probe\_ratio}
\,
\Delta t_i.
\end{equation}
$$

The sensitivity parameter determines when the difference between the control
computed with $\Delta t_i$ and the control computed with $\Delta t_i^{probe}$ is
large enough to trigger refinement.

---

#### Warm-start parameters

```python
initial_X_guess
initial_P_guess
initial_guess_label
```

The optional arrays `initial_X_guess` and `initial_P_guess` are used as the
initial state and costate guesses for the first TPBVP solve. If they are not
provided, the inner Newton solver constructs its own default initial guess.

The string `initial_guess_label` is stored in the log and settings for
bookkeeping.

---

### 13.2 Initialization

The routine starts by validating the three indicator tolerances:

```python
validate_indicator_tolerances(tol_time, tol_PA, tol_delta)
```

This enforces positivity and prevents PA or smoothing tolerances from being much
larger than the time tolerance.

---

#### Log file setup

If `log_path` is not `None`, the code creates the parent directory if necessary
and opens the log file in overwrite mode:

```python
Path(log_path).parent.mkdir(parents=True, exist_ok=True)
log_f = open(log_path, "w", buffering=1)
```

The local helper `_log(msg)` then sends messages to the terminal when
`verbose=True` and to the log file when logging is enabled.

---

#### Mesh copy

The initial mesh is copied into a floating-point array:

```python
t_nodes = np.asarray(initial_mesh, dtype=float).copy()
```

The adaptive loop mutates `t_nodes` through refinement, so copying prevents the
input array from being modified outside the solver.

---

#### Initial PA bundle

The routine initializes an empty PA bundle:

```python
bundle = PABundle()
```

Then it tries to add basic controls. If control bounds are available, it adds

$$
\begin{equation}
u_{\mathrm{mid}}
=
\frac{1}{2}(u_{\min}+u_{\max}),
\qquad
u_{\min},
\qquad
u_{\max}.
\end{equation}
$$

In code this corresponds to

```python
u_mid = 0.5 * (u_min + u_max)
bundle.add_control(u_mid)
bundle.add_control(u_min)
bundle.add_control(u_max)
```

If bounds are not available but the control dimension $m$ is known, it adds the
zero control

$$
\begin{equation}
u_0=0 \in \mathbb{R}^m.
\end{equation}
$$

This initial bundle gives the smoothed Hamiltonian at least a minimal set of
planes before the first TPBVP solve.

---

#### Initial smoothing parameter and logs

The smoothing parameter is initialized as

$$
\begin{equation}
\delta = \delta_0.
\end{equation}
$$

The routine also initializes

```python
log = []
bundle_support_points = []
```

The list `log` stores per-iteration diagnostics. The list
`bundle_support_points` stores information about controls added during bootstrap
or PA enrichment.

---

#### Initial guesses

The optional initial guesses are copied if provided:

```python
X_guess = None if initial_X_guess is None else np.asarray(initial_X_guess, dtype=float).copy()
P_guess = None if initial_P_guess is None else np.asarray(initial_P_guess, dtype=float).copy()
```

These guesses are passed to the first TPBVP solve. Later, after each adaptive
update, the current solution is reused or interpolated to warm-start the next
solve.

---

#### Explicit mode flag

Finally, the routine defines

```python
explicit_mode = bool(use_explicit_hamiltonian_gradients)
```

When this flag is true, PA and smoothing indicators are not used to decide
adaptive actions. The time indicator remains active.

---

### 13.3 Beginning of an outer iteration

The main loop is

```python
for k in range(max_iters):
    ...
```

At the start of each outer iteration, the code solves the discrete smoothed PMP
TPBVP on the current mesh, with the current PA bundle and current smoothing
parameter:

```python
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
```

Mathematically, this computes discrete state and costate arrays satisfying the
current smoothed shooting equations:

$$
\begin{equation}
F_h(X,P;\,t_{\mathrm{nodes}},U_{\mathrm{bundle}},\delta)
\approx
0.
\end{equation}
$$

The returned dictionary `info` contains Newton diagnostics such as the number of
iterations, residual norm, solver phase, and whether a fallback solver was used.

---

### 13.4 Bootstrap after the first coarse solve

On the first outer iteration, after the initial TPBVP solve, the code may enrich
the PA bundle using the newly computed trajectory. This happens only when

```python
k == 0 and not use_explicit_hamiltonian_gradients
```

The bootstrap call is

```python
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
```

Notice that the main routine overrides the bootstrap defaults and uses
`num_support_nodes=12` and `grid_size=51` in this call.

The purpose of this step is to add controls that are active or nearly active
along the first computed trajectory. If the bundle initially contains only
basic controls, the first smoothed Hamiltonian approximation may be crude. The
bootstrap improves the local PA approximation before the algorithm starts
interpreting PA-error indicators.

If at least one new control is added, the TPBVP is solved again on the same mesh
and with the same smoothing parameter:

```python
if added > 0:
    X_guess, P_guess = X, P
    X, P, info = solve_tpbvp(...)
```

This gives a trajectory consistent with the improved bundle.

---

### 13.5 Feasibility-driven refinement branch

After the TPBVP solve and possible bootstrap, the code checks whether a
feasibility-driven refinement pass is needed.

This branch is active only when the problem provides

```python
problem.step_feasible_control_fn
```

and the solve is not in explicit-gradient mode.

The code calls

```python
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
```

If `marked_feasibility` is nonempty, then the algorithm records a special log
entry with action

```python
"refine_feasibility(marked=...)"
```

The usual time, PA, and smoothing indicators are not computed for that
iteration. Instead, the log stores placeholder arrays and the diagnostic list
`feasibility_issues`.

The mesh and warm-start trajectories are refined by midpoint insertion:

```python
t_nodes, X_guess, P_guess = _refine_selected_intervals(
    t_nodes,
    X,
    P,
    marked_feasibility,
)
```

If the problem requires time-step-dependent bundle refresh, the bundle is then
rebuilt on the refined mesh:

```python
refreshed_bundle = _refresh_bundle_support_controls(
    problem,
    t_nodes,
    X_guess,
    P_guess,
    bundle_support_points,
    use_oracle=use_oracle_PA,
)
if refreshed_bundle is not None:
    bundle = refreshed_bundle
```

The loop then immediately continues to the next outer iteration. In other
words, feasibility refinement has priority over the standard indicator-based
decision logic.

This priority is important for constrained problems: before trusting the usual
error indicators, the code first ensures that the local feasible-control
structure is sufficiently resolved by the mesh.

---

### 13.6 Time-indicator computation inside the loop

If no feasibility refinement is triggered, the code computes the standard
adaptive indicators.

Let

$$
\begin{equation}
N = \mathrm{len}(t_{\mathrm{nodes}})-1.
\end{equation}
$$

For each interval $[t_i,t_{i+1}]$, the time step is

$$
\begin{equation}
\Delta t_i
=
t_{i+1}-t_i.
\end{equation}
$$

The largest step is

$$
\begin{equation}
\Delta t_{\max}
=
\max_i \Delta t_i.
\end{equation}
$$

The indicator floor is

$$
\begin{equation}
\mathrm{floor}
=
K_{\mathrm{time}}\sqrt{\Delta t_{\max}}.
\end{equation}
$$

At the symplectic-Euler evaluation point $(P_{i+1},X_i,t_i)$, the code obtains

$$
\begin{equation}
H_p
=
\frac{\partial H}{\partial p}(P_{i+1},X_i,t_i),
\qquad
H_x
=
\frac{\partial H}{\partial x}(P_{i+1},X_i,t_i),
\end{equation}
$$

where $H$ means either the smoothed PA Hamiltonian $H_\delta$ or the explicit
Hamiltonian used in explicit-gradient mode.

The raw time-error density is

$$
\begin{equation}
\rho_i
=
-\frac{1}{2}H_p^\top H_x.
\end{equation}
$$

The floored density is

$$
\begin{equation}
\bar\rho_i
=
\max(|\rho_i|,\mathrm{floor}).
\end{equation}
$$

The local time indicator is

$$
\begin{equation}
\bar r_i
=
|\bar\rho_i|(\Delta t_i)^2.
\end{equation}
$$

The scalar indicators recorded by the code are

$$
\begin{equation}
\eta_{\mathrm{time}}
=
\max_i \bar r_i,
\qquad
\eta_{\mathrm{time,sum}}
=
\sum_i \bar r_i.
\end{equation}
$$

The stopping threshold and marking threshold are

$$
\begin{equation}
\mathrm{tol}_{time}^{*}
=
\frac{\mathrm{tol}_{time}}{N},
\qquad
\mathrm{mark\_thr}
=
s_{\mathrm{time}}
\frac{\mathrm{tol}_{time}}{N}.
\end{equation}
$$

Intervals satisfying

$$
\begin{equation}
\bar r_i > \mathrm{mark\_thr}
\end{equation}
$$

are considered marked for time refinement.

---

### 13.7 PA and smoothing indicators inside the loop

If `explicit_mode=True`, the code sets the PA and smoothing indicators to zero:

```python
eta_PA = 0.0
eta_delta = 0.0
```

and creates zero arrays for their local diagnostics. In this mode, only the time
indicator drives adaptivity.

If `explicit_mode=False`, the code computes both the PA-bundle indicator and
the smoothing indicator.

---

#### PA-bundle indicator

At each node, the code compares the PA surrogate $\bar H$ with the richer
Hamiltonian approximation $H$ returned by `compute_H`:

$$
\begin{equation}
e_i^{PA}
=
\bar H(P_i,X_i,t_i)
-
H(P_i,X_i,t_i).
\end{equation}
$$

On interval $[t_i,t_{i+1}]$, the local PA contribution is

$$
\begin{equation}
\eta_i^{PA}
=
\frac{1}{2}
\left(
e_i^{PA}
+
e_{i+1}^{PA}
\right)
\Delta t_i.
\end{equation}
$$

The global PA indicator is accumulated as

$$
\begin{equation}
\eta_{PA}
=
\sum_{i=0}^{N-1}
\eta_i^{PA}.
\end{equation}
$$

The code also records:

```python
pa_gap_nodes
eta_PA_local
active_plane_idx_nodes
```

These arrays are used later for logging and for choosing where to add new PA
planes.

---

#### Smoothing indicator

The smoothing indicator compares the PA Hamiltonian $\bar H$ with the smoothed
Hamiltonian $H_\delta$:

$$
\begin{equation}
e_i^\delta
=
\bar H(P_i,X_i,t_i)
-
H_\delta(P_i,X_i,t_i).
\end{equation}
$$

The interval contribution is

$$
\begin{equation}
\eta_i^\delta
=
\frac{1}{2}
\left(
e_i^\delta
+
e_{i+1}^\delta
\right)
\Delta t_i.
\end{equation}
$$

The global smoothing indicator is

$$
\begin{equation}
\eta_\delta
=
\sum_{i=0}^{N-1}
\eta_i^\delta.
\end{equation}
$$

The code also records:

```python
delta_gap_nodes
eta_delta_local
```

for diagnostics and reporting.

At this point in the loop, the solver has all information needed to select the
next adaptive action.

---

### 13.8 Adaptive action selection

After computing the time, PA, and smoothing indicators, the code counts how many
intervals are marked for time refinement:

```python
n_mark = int(np.sum(eta_time_local > mark_thr)) if N > 0 else 0
```

Mathematically,

$$
\begin{equation}
n_{\mathrm{mark}}
=
\#\left\{
i
:
\bar r_i > \mathrm{mark\_thr}
\right\}.
\end{equation}
$$

The code also computes the dominant non-time indicator

$$
\begin{equation}
\eta_{\mathrm{non-time}}
=
\max(\eta_{PA},\eta_\delta)
\end{equation}
$$

when not in explicit-gradient mode. In explicit-gradient mode this value is set
to zero.

The boolean

```python
time_refinement_suppressed
```

records whether time refinement is being suppressed because the time indicator
is small relative to the dominant non-time indicator:

$$
\begin{equation}
\eta_{\mathrm{time}}
\le
\mathrm{time\_balance\_ratio}
\,
\eta_{\mathrm{non-time}}.
\end{equation}
$$

The action is then selected by

```python
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
```

The returned action is one of the following strings:

```python
"STOP"
"refine_time(marked=...)"
"add_plane"
"delta*=0.5"
"continue"
```

In normal operation, the important actions are stop, refine time, add a PA
plane, or reduce the smoothing parameter. The fallback string `"continue"` is a
defensive return value for logically unusual cases.

---

### 13.9 PA-addition plan

If the selected action is

```python
"add_plane"
```

the code prepares a PA-enrichment plan before appending the iteration log entry.

The target number of new PA support points is

$$
\begin{equation}
N_{\mathrm{add}}
=
\max
\left(
1,
\left\lceil
\mathrm{pa\_add\_fraction}
\cdot
\max(M,1)
\right\rceil
\right),
\end{equation}
$$

where $M$ is the current number of controls in the PA bundle.

At each mesh node, the code computes a locally minimizing control using
`compute_H`:

$$
\begin{equation}
u_i^\star
\in
\arg\min_u
\left\{
P_i^\top f(X_i,u,t_i)
+
\ell(X_i,u,t_i)
\right\}.
\end{equation}
$$

These nodewise candidates are stored in `pa_candidate_controls`.

The nodewise PA gaps are converted into time-weighted scores by

```python
pa_node_dt, pa_scores = compute_pa_ranking_scores(t_nodes, pa_gap_nodes)
```

That is,

$$
\begin{equation}
s_i^{PA}
=
e_i^{PA}
\Delta t_i^{node}.
\end{equation}
$$

Then the selected node indices are computed by

```python
selected_indices, selection_meta = select_pa_enrichment_candidates(
    t_nodes,
    pa_scores,
    target_count,
    separation_factor=pa_time_separation_factor,
    gap_floor_ratio=pa_gap_floor_ratio,
)
```

This selection keeps only high-score nodes and enforces time separation between
chosen support points.

The plan is stored in the dictionary `pa_addition_plan`, including:

```python
{
    "target_count": target_count,
    "selected_node_indices": selected_indices,
    "selected_times": ...,
    "selected_pa_gaps": ...,
    "selected_time_steps": ...,
    "selected_pa_scores": ...,
    "rejected_by_time_separation": ...,
    "rejected_below_score_floor": ...,
    "max_gap": ...,
    "max_score": ...,
    "score_floor": ...,
}
```

At this stage, the controls are not yet added to the bundle. The code first
logs the current iteration, then performs the action.

---

### 13.10 Checking the stopping criteria

The code records whether all active indicators are within tolerance:

```python
all_indicators_within_tolerance = (
    eta_time <= tol_time_star
    and (explicit_mode or eta_PA <= tol_PA)
    and (explicit_mode or eta_delta <= tol_delta)
)
```

Equivalently, in full PA/smoothing mode, this is

$$
\begin{equation}
\eta_{\mathrm{time}}
\le
\mathrm{tol}_{time}^{*},
\qquad
\eta_{PA}
\le
\mathrm{tol}_{PA},
\qquad
\eta_\delta
\le
\mathrm{tol}_{\delta}.
\end{equation}
$$

In explicit-gradient mode, only the time condition is active:

$$
\begin{equation}
\eta_{\mathrm{time}}
\le
\mathrm{tol}_{time}^{*}.
\end{equation}
$$

The code also reconstructs the nodal control trajectory

```python
U = _compute_node_controls(
    problem,
    bundle,
    X,
    P,
    t_nodes,
    restricted=True,
    use_oracle=use_oracle_PA,
)
```

so that the mesh objective and, optionally, the stored iterates can include a
control trajectory.

---

### 13.11 Standard iteration log entry

After action selection, the code appends a detailed dictionary to `log`.

The main scalar fields are:

```python
{
    "iteration": k,
    "N": N,
    "M": bundle.num_planes(),
    "delta": delta,
    "eta_time": eta_time,
    "eta_time_sum": eta_time_sum,
    "eta_PA": eta_PA,
    "eta_delta": eta_delta,
    "tol_time_star": tol_time_star,
    "mark_thr": mark_thr,
    "newton_iter": info["iterations"],
    "newton_residual": info["residual_norm"],
    "solver_phase": info.get("solver_phase", "newton"),
    "fallback_used": bool(info.get("fallback_used", False)),
    "objective_mesh_approx": _mesh_objective(problem, t_nodes, X, U),
    "all_indicators_within_tolerance": all_indicators_within_tolerance,
    "action": action,
}
```

The main array fields are:

```python
{
    "rho": rho_arr.copy(),
    "rho_bar": rho_bar_arr.copy(),
    "r_bar": eta_time_local.copy(),
    "eta_PA_local": eta_PA_local.copy(),
    "eta_delta_local": eta_delta_local.copy(),
    "pa_gap_nodes": pa_gap_nodes.copy(),
    "delta_gap_nodes": delta_gap_nodes.copy(),
    "active_plane_idx_nodes": active_plane_idx_nodes.copy(),
    "t_nodes_iter": t_nodes.copy(),
}
```

The log also records action-policy and PA-selection metadata:

```python
{
    "time_balance_ratio": time_balance_ratio,
    "dominant_non_time_indicator": dominant_non_time_indicator,
    "time_refinement_suppressed": time_refinement_suppressed,
    "pa_add_fraction": pa_add_fraction,
    "pa_time_separation_factor": pa_time_separation_factor,
    "pa_gap_floor_ratio": pa_gap_floor_ratio,
    "initial_guess_label": initial_guess_label,
    "pa_addition_plan": pa_addition_plan,
}
```

If `store_iterates=True`, the current trajectories and controls are copied into
the log:

```python
entry["X_iter"] = X.copy()
entry["P_iter"] = P.copy()
entry["U_iter"] = U.copy()
```

The accumulated PA support points are also copied into

```python
entry["bundle_support_points_so_far"]
```

This extended storage is useful for generating detailed reports, but it can be
memory-intensive for large meshes or many iterations.

---

### 13.12 Progress printing

Every `print_every` iterations, the routine prints a one-line summary of the
current adaptive state. The printed information includes:

- number of intervals $N$,
- number of PA planes $M$,
- minimum and maximum time step,
- current smoothing parameter $\delta$,
- Newton iteration count and residual,
- current values of the three indicators,
- selected next action.

This summary is sent to the terminal if `verbose=True` and to `log_path` if file
logging is enabled.

---

### 13.13 Stopping

If the selected action is

```python
"STOP"
```

the loop terminates immediately:

```python
if action == "STOP":
    break
```

At this point, the current trajectories, bundle, and smoothing parameter are
considered consistent with the requested tolerances.

---

### 13.14 Executing a time-refinement action

If the selected action starts with

```python
"refine_time"
```

the code refines all intervals whose local time indicator exceeds the marking
threshold.

For each interval $[t_i,t_{i+1}]$, the local error is

$$
\begin{equation}
\bar r_i.
\end{equation}
$$

The interval is refined if

$$
\begin{equation}
\bar r_i
>
\mathrm{mark\_thr}.
\end{equation}
$$

For each marked interval, the midpoint

$$
\begin{equation}
t_{i+1/2}
=
\frac{1}{2}(t_i+t_{i+1})
\end{equation}
$$

is inserted.

The state and costate warm-start values are linearly interpolated:

$$
\begin{equation}
X_{i+1/2}
=
(1-\alpha)X_i+\alpha X_{i+1},
\qquad
P_{i+1/2}
=
(1-\alpha)P_i+\alpha P_{i+1},
\end{equation}
$$

where

$$
\begin{equation}
\alpha
=
\frac{t_{i+1/2}-t_i}{t_{i+1}-t_i}
=
\frac{1}{2}.
\end{equation}
$$

The refined arrays become the warm starts for the next outer iteration:

```python
t_nodes = np.array(new_nodes, dtype=float)
X_guess = np.array(X_new)
P_guess = np.array(P_new)
```

After time refinement, the code may refresh the PA bundle if the problem has
time-step-dependent local controls or feasibility logic:

```python
refreshed_bundle = _refresh_bundle_support_controls(...)
if refreshed_bundle is not None:
    bundle = refreshed_bundle
```

Then the loop continues to the next outer iteration.

---

### 13.15 Executing a PA-enrichment action

If the selected action is

```python
"add_plane"
```

the code loops over the selected PA-enrichment nodes and their corresponding
candidate controls.

For a selected node $i$, the candidate control is approximately

$$
\begin{equation}
u_i^\star
\in
\arg\min_u
\left\{
P_i^\top f(X_i,u,t_i)
+
\ell(X_i,u,t_i)
\right\}.
\end{equation}
$$

The control is inserted into the bundle using

```python
bundle.add_control(candidate_u)
```

The bundle itself handles duplicate rejection. If the number of bundle planes
increases, the code records a support point containing:

```python
{
    "iteration": k,
    "kind": "add_plane",
    "node_index": idx,
    "time": t_nodes[idx],
    "state": X[idx],
    "costate": P[idx],
    "control": candidate_u,
    "local_dt": local_dt,
    "bundle_size_after": bundle.num_planes(),
    "pa_gap_at_point": pa_gap_nodes[idx],
    "pa_time_step_at_point": pa_node_dt[idx],
    "pa_score_at_point": pa_scores[idx],
}
```

If the candidate is a duplicate, no new plane is added and the duplicate
rejection counter is incremented.

After processing the selected controls, the current state and costate
trajectories are reused as warm starts:

```python
X_guess = X
P_guess = P
```

The loop then continues to the next outer iteration, where the TPBVP is solved
again with the enriched PA bundle.

---

### 13.16 Executing a smoothing-continuation action

If the selected action is

```python
"delta*=0.5"
```

the smoothing parameter is halved:

$$
\begin{equation}
\delta
\leftarrow
\frac{1}{2}\delta.
\end{equation}
$$

The mesh and bundle are not changed. The current trajectories are reused as
warm starts:

```python
X_guess = X
P_guess = P
```

The next outer iteration then solves the TPBVP with the smaller smoothing
parameter.

---

### 13.17 Final consistency solve

After the outer loop exits, the code checks whether the returned trajectory is
consistent with the final mesh and final smoothing parameter. A final TPBVP
solve is performed if one of the following is true:

```python
X is None
len(t_nodes) != X.shape[0]
len(t_nodes) != P.shape[0]
delta_solved != delta
```

This can happen, for example, if the last executed action changed the mesh,
bundle, or smoothing parameter and then the loop ended because `max_iters` was
reached.

When a final solve is needed, the code calls `solve_tpbvp` one more time using
the current `t_nodes`, `bundle`, and `delta`.

After this final solve, the code recomputes the time, PA, and smoothing
indicators so the returned log reflects the final trajectory more accurately.

---

### 13.18 Final log entry

If a final consistency solve is performed and the existing log is nonempty, the
code appends one more entry. Its action is

```python
"STOP"
```

if all active indicators are now within tolerance, and

```python
"final_resolve"
```

otherwise.

Thus, the final log entry distinguishes between two cases:

1. the algorithm ended with a fully satisfactory final trajectory;
2. the algorithm performed a final consistency solve, but the final indicators
   still did not satisfy all requested tolerances.

The second case can occur if `max_iters` is reached before convergence.

---

### 13.19 Return dictionary

The routine returns a dictionary with the final solution, diagnostics, and
settings.

The main solution fields are:

```python
{
    "t_nodes": t_nodes,
    "X": X,
    "P": P,
    "bundle": bundle,
    "delta": delta,
    "problem": problem,
}
```

The final time-indicator arrays are also returned directly:

```python
{
    "rhobar": rho_bar_arr,
    "rbar": eta_time_local,
}
```

Here `rhobar` stores the final floored time-error densities $\bar\rho_i$, and
`rbar` stores the final local time indicators $\bar r_i$.

The diagnostic fields are:

```python
{
    "log": log,
    "info": info,
    "bundle_support_points": bundle_support_points,
}
```

Finally, the returned `settings` dictionary records the main solver options:

```python
{
    "tol_time": tol_time,
    "tol_PA": tol_PA,
    "tol_delta": tol_delta,
    "max_iters": max_iters,
    "delta0": delta0,
    "s_time": s_time,
    "K_time": K_time,
    "newton_tol": newton_tol,
    "newton_max_iter": newton_max_iter,
    "use_oracle_bootstrap": bool(use_oracle_bootstrap),
    "use_oracle_PA": bool(use_oracle_PA),
    "use_explicit_hamiltonian_gradients": bool(use_explicit_hamiltonian_gradients),
    "store_iterates": bool(store_iterates),
    "fallback_solver": fallback_solver,
    "time_balance_ratio": float(time_balance_ratio),
    "pa_add_fraction": float(pa_add_fraction),
    "pa_time_separation_factor": float(pa_time_separation_factor),
    "pa_gap_floor_ratio": float(pa_gap_floor_ratio),
    "initial_guess_label": initial_guess_label,
}
```

The returned dictionary is therefore sufficient for both numerical use and
post-processing: it contains the final solution, the final approximation model,
the full adaptive history, and the settings needed to interpret the run.

---

## 14) Full adaptive algorithm summary

The following pseudocode summarizes the current implementation of
`solve_optimal_control`.

```python
validate_indicator_tolerances(tol_time, tol_PA, tol_delta)

initialize log file
t_nodes = copy(initial_mesh)

bundle = PABundle()
seed bundle with midpoint/bound controls or zero control

delta = delta0
log = []
bundle_support_points = []

X_guess = initial_X_guess or None
P_guess = initial_P_guess or None

explicit_mode = bool(use_explicit_hamiltonian_gradients)

for k in range(max_iters):

    # 1. Solve fixed-mesh TPBVP
    X, P, info = solve_tpbvp(
        problem,
        t_nodes,
        bundle,
        delta,
        X_guess,
        P_guess,
        ...
    )

    # 2. Bootstrap PA bundle after first coarse solve
    if k == 0 and not explicit_mode:
        added = bootstrap_bundle_from_trajectory(...)
        if added > 0:
            X_guess, P_guess = X, P
            X, P, info = solve_tpbvp(...)

    # 3. Optional feasibility-driven refinement
    if problem has step-feasibility callback and not explicit_mode:
        marked_feasibility, feasibility_issues = (
            _find_feasibility_refinement_intervals(...)
        )

        if marked_feasibility is not empty:
            append feasibility-refinement log entry

            t_nodes, X_guess, P_guess = _refine_selected_intervals(...)

            refreshed_bundle = _refresh_bundle_support_controls(...)
            if refreshed_bundle is not None:
                bundle = refreshed_bundle

            continue

    # 4. Compute indicators
    compute eta_time, eta_time_sum, eta_time_local

    if explicit_mode:
        eta_PA = 0
        eta_delta = 0
    else:
        compute eta_PA and eta_PA_local
        compute eta_delta and eta_delta_local

    # 5. Choose adaptive action
    action = choose_adaptive_action(...)

    # 6. If action is add_plane, prepare PA-addition plan
    if action == "add_plane":
        compute nodewise candidate controls
        rank PA gaps
        select separated enrichment nodes

    # 7. Append standard log entry
    U = _compute_node_controls(...)
    append iteration diagnostics to log

    # 8. Execute selected action
    if action == "STOP":
        break

    elif action starts with "refine_time":
        refine intervals with eta_time_local > mark_thr
        refresh bundle if needed
        continue

    elif action == "add_plane":
        add selected candidate controls to bundle
        X_guess, P_guess = X, P
        continue

    elif action == "delta*=0.5":
        delta = 0.5 * delta
        X_guess, P_guess = X, P
        continue

# 9. Final consistency solve if needed
if final mesh/trajectory/delta are inconsistent:
    X, P, info = solve_tpbvp(...)
    recompute indicators
    append final log entry if possible

return result dictionary
```

The adaptive loop can therefore be viewed as alternating between a fixed-model
TPBVP solve and one controlled update of the approximation model:

$$
\begin{equation}
(t_{\mathrm{nodes}},U_{\mathrm{bundle}},\delta)
\longrightarrow
(X,P)
\longrightarrow
(\eta_{\mathrm{time}},\eta_{PA},\eta_\delta)
\longrightarrow
\text{adaptive update}.
\end{equation}
$$

---

## 15) Important implementation notes

This section records several details that are easy to miss when reading only
the high-level algorithm.

---

### 15.1 The time tolerance is used locally

The user supplies a global-looking tolerance `tol_time`, but the code compares
the maximum local time indicator against

$$
\begin{equation}
\mathrm{tol}_{time}^{*}
=
\frac{\mathrm{tol}_{time}}{N}.
\end{equation}
$$

Thus the stopping condition for the time indicator is

$$
\begin{equation}
\eta_{\mathrm{time}}
=
\max_i \bar r_i
\le
\frac{\mathrm{tol}_{time}}{N}.
\end{equation}
$$

The marking threshold is similarly local:

$$
\begin{equation}
\mathrm{mark\_thr}
=
s_{\mathrm{time}}
\frac{\mathrm{tol}_{time}}{N}.
\end{equation}
$$

---

### 15.2 Feasibility refinement has priority

If feasibility refinement is triggered, the usual time, PA, and smoothing
indicator logic is skipped for that iteration. The algorithm refines the
feasibility-marked intervals, refreshes the bundle if needed, and immediately
continues to the next TPBVP solve.

This design is intentional: for constrained problems, a poorly resolved
feasible-control structure can make the standard indicators unreliable.

---

### 15.3 Bootstrap defaults differ from function defaults

The function `bootstrap_bundle_from_trajectory` has defaults

```python
num_support_nodes=20
grid_size=3
```

but the main routine calls it with

```python
num_support_nodes=12
grid_size=51
```

during the first outer iteration.

Therefore, the effective bootstrap behavior inside `solve_optimal_control` is
determined by the values in the call site, not only by the helper's signature.

---

### 15.4 Explicit-gradient mode disables PA and smoothing decisions

When

```python
use_explicit_hamiltonian_gradients=True
```

the code sets

```python
explicit_mode = True
```

In this mode:

- PA bootstrap is skipped;
- feasibility refinement branch is skipped;
- PA and smoothing indicators are set to zero;
- `choose_adaptive_action` only uses the time indicator.

Thus the adaptive loop behaves as a time-refinement loop driven by explicit
Hamiltonian gradients.

---

### 15.5 Bundle enrichment is duplicate-safe

Whenever controls are inserted into the PA bundle, the insertion goes through

```python
bundle.add_control(...)
```

The bundle performs a tolerance-based duplicate check. Therefore, the number of
selected PA-enrichment nodes can be larger than the number of controls actually
added.

The log records duplicate rejections during the `"add_plane"` action.

---

### 15.6 PA enrichment may add multiple controls per action

Although the action string is `"add_plane"`, the current implementation can add
more than one control in a single outer iteration. The target number is

$$
\begin{equation}
N_{\mathrm{add}}
=
\max
\left(
1,
\left\lceil
\mathrm{pa\_add\_fraction}
\max(M,1)
\right\rceil
\right).
\end{equation}
$$

The selected controls are chosen from high PA-score nodes subject to time
separation.

---

### 15.7 Time refinement uses midpoint insertion and linear warm starts

Both standard time refinement and feasibility-driven refinement insert midpoint
nodes. The new state and costate guesses are obtained by linear interpolation.

This keeps the warm start simple and robust:

$$
\begin{equation}
X_{i+1/2}
=
\frac{1}{2}(X_i+X_{i+1}),
\qquad
P_{i+1/2}
=
\frac{1}{2}(P_i+P_{i+1}).
\end{equation}
$$

---

### 15.8 Final resolve can produce a final diagnostic entry

If the last adaptive action changes the mesh, bundle, or smoothing parameter,
and the loop stops because `max_iters` is reached, the stored trajectory may not
match the final adaptive objects. The code checks for this and performs a final
TPBVP solve if needed.

After this solve, it recomputes indicators and can append a final log entry
whose action is either

```python
"STOP"
```

or

```python
"final_resolve"
```

The second case means the final solve was performed, but the final indicators
still did not satisfy all requested tolerances.

---

### 15.9 The log is the main diagnostic object

The returned solution fields give the final state of the algorithm, but the
returned `log` is the main object for understanding the adaptive path. It stores
the mesh size, bundle size, smoothing parameter, Newton diagnostics, indicators,
local indicator arrays, action decisions, and PA-enrichment metadata at each
outer iteration.

For report generation, use `store_iterates=True` to preserve copies of
intermediate trajectories and controls.

---

## 16) Relation to the other core modules

The adaptive loop coordinates several other core modules. Their internal
details are documented separately.

| Module | Role in the adaptive loop |
|---|---|
| `problem.py` | Defines the OCP data: dynamics, costs, bounds, optional Hamiltonian gradients, oracles, feasibility callbacks, and projection logic. |
| `pa_bundle.py` | Stores the finite set of candidate controls used by the PA Hamiltonian surrogate. |
| `hamiltonian.py` | Computes local Hamiltonian minimizers and richer Hamiltonian values used for PA gaps and control reconstruction. |
| `smoothing.py` | Evaluates the smoothed PA Hamiltonian $H_\delta$ and its gradients. |
| `integrators.py` | Assembles the discrete symplectic-Euler shooting residual and Jacobian. |
| `shooting.py` | Wraps residual and Jacobian assembly for packed unknown vectors. |
| `newton.py` | Solves the fixed-mesh TPBVP using damped Newton and optional fallback least squares. |
| `constraints.py` | Provides basic projection and clipping utilities for constraints. |

Thus `adaptivity.py` should be understood as the **outer coordinator**. It does
not define the problem, assemble the low-level residual, or solve Newton systems
itself. Instead, it repeatedly calls the fixed-mesh solver, evaluates
indicators, and updates the mesh, PA bundle, or smoothing parameter.

---

## 17) Conceptual summary

The current implementation separates three approximation errors:

1. **time discretization error**, controlled by mesh refinement;
2. **PA-bundle approximation error**, controlled by adding new controls to the
   bundle;
3. **smoothing error**, controlled by decreasing $\delta$.

The adaptive method attempts to reduce these errors in a balanced way. The
central loop is:

$$
\begin{equation}
\boxed{
\text{solve}
\quad\longrightarrow\quad
\text{estimate}
\quad\longrightarrow\quad
\text{choose action}
\quad\longrightarrow\quad
\text{update}
}
\end{equation}
$$

where the update is one of

$$
\begin{equation}
\boxed{
\text{refine mesh}
}
\qquad
\boxed{
\text{add PA support controls}
}
\qquad
\boxed{
\delta \leftarrow \frac{1}{2}\delta
}
\end{equation}
$$

with feasibility-driven refinement taking priority when problem-specific
feasibility checks indicate that the current mesh is not adequate.

This is the role of `adaptivity.py`: it is the adaptive controller around the
fixed-mesh Pontryagin solver.