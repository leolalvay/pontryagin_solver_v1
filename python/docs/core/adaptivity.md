# `core/adaptivity.py` — Adaptive outer loop (mesh refinement + PA bundle enrichment + smoothing continuation)

This file implements the **outer adaptivity loop** that repeatedly solves a discretized smoothed PMP TPBVP and then updates **one** of:

1) the **time mesh** $t_0<\dots<t_N$ (refinement),
2) the **piecewise-affine (PA) bundle** (add a new “plane”, i.e., a new control candidate),
3) the **smoothing parameter** $\delta$ (reduce it).

The entry point is:

```python
solve_optimal_control(problem, initial_mesh, tol_time, tol_PA, tol_delta, max_iters, delta0)
```

It returns a dictionary containing the final $(t\_\text{nodes}, X, P)$, the final bundle, final $\delta$, and a per-iteration log.

---
### `bootstrap_bundle_from_trajectory`

This helper implements a **minimal bootstrap** for the PA bundle: after a first coarse solve has produced a trajectory $(X,P)$, we add a few **approximately active controls** to the bundle so that the piecewise-affine surrogate $\bar H$ becomes a better approximation of the Hamiltonian in the region of costates actually visited by the solver.

---

#### Mathematical background

For fixed $(x,t)$, define the affine function in $p$
$$
h_u(p;x,t) := p^\top f(x,u,t) + \ell(x,u,t).
$$

With the “min” convention, the Hamiltonian is
$$
H(p;x,t) = \min_{u\in A} \; h_u(p;x,t),
$$
hence $H(\cdot;x,t)$ is **concave in $p$** (a pointwise minimum of affine functions).

The PA bundle surrogate restricts the minimization to a **finite** set of controls $U_{\text{bundle}}$:
$$
\bar H(p;x,t) = \min_{u\in U_{\text{bundle}}} \; h_u(p;x,t),
$$
so $\bar H(p;x,t) \ge H(p;x,t)$. Improving $\bar H$ requires enriching $U_{\text{bundle}}$ with controls that are (approximately) optimal for representative triplets $(x_i,p_i,t_i)$.

This function enriches $U_{\text{bundle}}$ by computing, at a small set of support nodes,
$$
\hat u_i \approx \arg\min_{u\in [u_{\min},u_{\max}]} \Big(p_i^\top f(x_i,u,t_i) + \ell(x_i,u,t_i)\Big),
$$
using a cheap 1D grid search (minimal-change version: **scalar control only**, $m=1$).

---

#### Signature

```python
def bootstrap_bundle_from_trajectory(
    problem,
    t_nodes: np.ndarray,
    X: np.ndarray,
    P: np.ndarray,
    bundle,
    restricted: bool = True,
    num_support_nodes: int = 8,
    grid_size: int = 20,
) -> int:
    ...
```

---

#### What the key code blocks do

##### 1) Read bounds and enforce the “minimal” scope ($m=1$)

```python
bounds = problem.control_bounds_tuple()
if bounds is None:
    return 0

u_min, u_max = bounds
m = int(u_min.size)
if m != 1:
    return 0
```

- The bootstrap relies on a **1D grid** in the control interval. If the problem has no bounds, or if the control is not scalar, this minimal version exits early and makes no changes.

##### 2) Pick a small set of representative “support nodes” along the mesh

```python
N = len(t_nodes) - 1
k = min(num_support_nodes, N + 1)
idx = np.unique(np.round(np.linspace(0, N, k)).astype(int))
```

- The trajectory $(X,P)$ is available at nodes $t_0,\dots,t_N$.
- Instead of using every node (expensive), we select a small set of indices `idx` that are roughly **uniformly spread** across $[0,T]$ and include endpoints.
- These are the points where we will compute approximately active controls.

##### 3) Build a cheap 1D control grid in $[u_{\min},u_{\max}]$

```python
u_grid = np.linspace(float(u_min[0]), float(u_max[0]), int(grid_size))
```

- This discretizes the admissible interval. The grid resolution controls the tradeoff:
  - larger `grid_size` $\Rightarrow$ better $\hat u_i$ but more evaluations of $f$ and $\ell$;
  - smaller `grid_size` $\Rightarrow$ cheaper but coarser.

##### 4) At each support node, solve a discrete inner minimization

```python
best_val = np.inf
best_u = None

for a in u_grid:
    u = np.array([a], dtype=float)

    val = float(np.dot(p_i, problem.f(x_i, u, t_i)) + problem.l(x_i, u, t_i))
    if val < best_val:
        best_val = val
        best_u = u
```

- This is the discrete analogue of
  $$\min_{u\in [u_{\min},u_{\max}]} \; h_u(p_i;x_i,t_i).$$
- For each grid point $u=[a]$, we evaluate $h_u$ and keep the best one.
- The winner `best_u` is the bootstrap approximation $\hat u_i$.

##### 5) Optional viability / restriction checks (consistency with `restricted=True`)

```python
if not problem.admissible_control(u):
    continue

if restricted:
    if hasattr(problem, "tangent_ok") and (not problem.tangent_ok(x_i, u, t_i)):
        continue
```

- `admissible_control` enforces basic feasibility (e.g., bounds, user-defined rules).
- If `restricted=True`, we also enforce viability via `tangent_ok` when available, mirroring the “restricted Hamiltonian” setting used elsewhere.

##### 6) Add the new control to the bundle (with deduplication)

```python
before = bundle.num_planes()
bundle.add_control(best_u)
if bundle.num_planes() > before:
    added += 1
```

- `bundle.add_control` uses an $L^2$ tolerance to avoid inserting duplicates.
- We count how many **new** controls were actually added and return that number.

---

#### Return value

- Returns the number of **new** (previously absent) controls added to the bundle.

---

#### Minimal implementation (as used in this repo)

```python
def bootstrap_bundle_from_trajectory(
    problem,
    t_nodes: np.ndarray,
    X: np.ndarray,
    P: np.ndarray,
    bundle,
    restricted: bool = True,
    num_support_nodes: int = 8,
    grid_size: int = 20,
) -> int:
    """
    Minimal bootstrap for PA bundle: add approximate active controls at a few support nodes
    by doing a cheap 1D grid search over control bounds.

    Returns the number of *new* controls added.
    """
    bounds = problem.control_bounds_tuple()
    if bounds is None:
        return 0

    u_min, u_max = bounds
    m = int(u_min.size)
    if m != 1:
        # Minimal-change version: only handle scalar control for now.
        return 0

    N = len(t_nodes) - 1
    if N <= 0:
        return 0

    # pick a few representative node indices (including endpoints)
    k = min(num_support_nodes, N + 1)
    idx = np.unique(np.round(np.linspace(0, N, k)).astype(int))

    u_grid = np.linspace(float(u_min[0]), float(u_max[0]), int(grid_size))

    added = 0
    for i in idx:
        x_i = X[i]
        p_i = P[i]
        t_i = float(t_nodes[i])

        best_val = np.inf
        best_u = None

        for a in u_grid:
            u = np.array([a], dtype=float)

            if not problem.admissible_control(u):
                continue
            if restricted:
                # keep consistent with compute_H(..., restricted=True)
                if hasattr(problem, "tangent_ok") and (not problem.tangent_ok(x_i, u, t_i)):
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

    return added
```

---

#### How it fits into the solver loop

The bootstrap is intended to be called **once**, immediately after the first coarse TPBVP solve:

1. Solve once with a minimal initial bundle.
2. Call `bootstrap_bundle_from_trajectory(...)` to enrich `bundle.controls`.
3. Re-solve once with the improved bundle (same mesh, same $\delta$).
4. Proceed with the usual adaptivity loop.

This provides a low-cost, structure-preserving improvement of the initial PA bundle without redesigning the surrogate machinery.


## 0) Objects and dimensions (what is being manipulated)

Let

- state dimension: $x(t)\in\mathbb{R}^n$,
- control dimension: $u(t)\in\mathbb{R}^m$,
- time mesh: $t\_\text{nodes} = (t_0,\dots,t_N)$ with $N = \text{len}(t\_\text{nodes})-1$.

Discrete unknowns in the inner TPBVP solve:

- $X_i \approx x(t_i)\in\mathbb{R}^n$, collected as $X\in\mathbb{R}^{(N+1)\times n}$,
- $P_i \approx p(t_i)\in\mathbb{R}^n$, collected as $P\in\mathbb{R}^{(N+1)\times n}$.

The TPBVP is solved by damped Newton (`core/newton.solve_tpbvp`) on the global shooting system $F(z)=0$ where
$$
z=(X_1,\dots,X_N,\;P_0,\dots,P_N)\in\mathbb{R}^{(2N+1)n}.
$$

The **bundle** is a finite set of control candidates
$$
\{a_1,\dots,a_M\}\subset \mathbb{R}^m,
$$
stored as `bundle.controls`, where M = `bundle.num_planes()`.

---

## 1) Hamiltonians used in the outer loop

### 1.1 PA surrogate Hamiltonian $\bar H$

Given bundle controls $a_i$, the surrogate Hamiltonian is
$$
\bar H(p,x,t) \;=\; \min_{i=1,\dots,M}\Big(p\cdot f(x,a_i,t) + \ell(x,a_i,t)\Big).
$$

This is implemented by:

```python
Hbar, active_idx = bundle.evaluate(problem, p, x, t)
```

### 1.2 “True” (restricted) Hamiltonian $H$ used for the PA indicator

The code computes a discrete minimization over:
- all $2^m$ **corner controls** of the control box $[u_{\min},u_{\max}]$,
- plus controls in `bundle.controls`,
and optionally enforces viability when `restricted=True`.

Conceptually:
$$
H(p,x,t) \approx \min_{u\in\mathcal{U}_{\text{candidates}}} \Big(p\cdot f(x,u,t)+\ell(x,u,t)\Big),
$$
with an additional “tangent cone / viability” filter if `restricted=True`.

This is implemented by:

```python
H, u_star = compute_H(problem, p, x, t, bundle.controls, restricted=True)
```

**Important:** this is not an exact continuous minimization over all admissible controls; it is a finite candidate minimization (corners + bundle planes), which is consistent with a prototype implementation but should be kept in mind when interpreting $\eta_{\text{PA}}$.

### 1.3 Smoothed Hamiltonian $H_\delta$

The smoothed Hamiltonian is the log-sum-exp soft-min of the PA planes:
$$
H_\delta(p,x,t)
=
-\delta\log\!\left(\sum_{i=1}^M \exp\!\left(-\frac{g_i(p,x,t)}{\delta}\right)\right),
\qquad
g_i(p,x,t)=p\cdot f(x,a_i,t)+\ell(x,a_i,t).
$$

It is evaluated (together with gradients) by:

```python
Hdelta, grad_p, grad_x = eval_H_smooth(problem, bundle, p, x, t, delta)
```

---

## 2) Outer-loop flow (high level)

Each outer iteration $k=0,1,2,\dots$ does:

1) **Solve TPBVP** for current $(t\_\text{nodes}, \text{bundle}, \delta)$ via Newton:
   $$ (X,P) \leftarrow \texttt{solve\_tpbvp}(\cdots). $$

2) **Compute three indicators**:
   - time discretization indicator $\eta_{\text{time}}$ (also local per-interval values),
   - PA surrogate indicator $\eta_{\text{PA}}$,
   - smoothing indicator $\eta_\delta$.

3) **Check stopping**: stop if all indicators are below their tolerances.

4) **Otherwise update exactly one item**, with the priority:
   - refine time mesh if $\eta_{\text{time}}>\text{tol\_time}$,
   - else add a bundle plane if $\eta_{\text{PA}}>\text{tol\_PA}$,
   - else reduce $\delta$ if $\eta_\delta>\text{tol\_delta}$.

The code reflects this priority with `continue` after each update.

---

## 3) Step 0 in code: initializing mesh, bundle, and $\delta$

### 3.1 Mesh

```python
t_nodes = np.asarray(initial_mesh, dtype=float).copy()
```

### 3.2 Bundle initialization

The smoothed Hamiltonian requires a non-empty bundle (at least one plane). The code tries to seed the bundle with a “neutral” control:

- If bounds are known, it uses the midpoint $u_0=\tfrac12(u_{\min}+u_{\max})$,
- otherwise it uses $u_0=0$.

```python
bundle = PABundle()
bounds = problem.control_bounds_tuple()
m = problem.m
if m is None and bounds is not None:
    m = bounds[0].size
if m is not None:
    if bounds is not None:
        u0 = 0.5 * (bounds[0] + bounds[1])
    else:
        u0 = np.zeros(m)
    bundle.add_control(u0)
```

### 3.3 Smoothing parameter

```python
delta = delta0
```

---

## 4) Inner TPBVP solve (Newton) and warm-starting

Inside the outer loop:

```python
X, P, info = solve_tpbvp(problem, t_nodes, bundle, delta, X_guess, P_guess)
```

- `X_guess` and `P_guess` are carried between outer iterations to warm-start Newton.
- After mesh refinement, guesses are updated by **linear interpolation** at inserted midpoints.
- After adding a plane or reducing $\delta$, the previous $(X,P)$ is reused as the next initial guess.

---

## 5) Error indicators (math + exact code meaning)

### 5.1 Time discretization indicator $\eta_{\text{time}}$

The code builds a *local* indicator per interval by measuring the variation of the smoothed Hamiltonian gradients across the interval endpoints.

First, gradients at all nodes:

```python
grad_p_list = []
grad_x_list = []
for i in range(N + 1):
    _, grad_p_i, grad_x_i = eval_H_smooth(problem, bundle, P[i], X[i], t_nodes[i], delta)
    grad_p_list.append(grad_p_i)
    grad_x_list.append(grad_x_i)
```

Then per interval $[t_i,t_{i+1}]$:
$$
\eta_{\text{time},i}
=
\Delta t_i\Big(
\|\nabla_p H_\delta(P_{i+1},X_{i+1},t_{i+1})-\nabla_p H_\delta(P_i,X_i,t_i)\|_2
+
\|\nabla_x H_\delta(P_{i+1},X_{i+1},t_{i+1})-\nabla_x H_\delta(P_i,X_i,t_i)\|_2
\Big).
$$

This matches:

```python
for i in range(N):
    dt = t_nodes[i + 1] - t_nodes[i]
    diff_gp = grad_p_list[i + 1] - grad_p_list[i]
    diff_gx = grad_x_list[i + 1] - grad_x_list[i]
    eta_time_local[i] = dt * (np.linalg.norm(diff_gp) + np.linalg.norm(diff_gx))
```

Global time indicator is the max:
$$
\eta_{\text{time}}=\max_{i=0,\dots,N-1}\eta_{\text{time},i}.
$$

```python
eta_time = np.max(eta_time_local) if N > 0 else 0.0
```

**Interpretation:** large gradient changes indicate that the discrete solution varies significantly across that interval, so the mesh should be refined there.

---

### 5.2 PA surrogate indicator $\eta_{\text{PA}}$

This indicator estimates how far the bundle surrogate $\bar H$ is from the “true” (restricted) Hamiltonian $H$ along the computed trajectory. The code integrates the gap via trapezoidal rule:

$$
\eta_{\text{PA}}
\approx
\sum_{i=0}^{N-1}
\frac{\Delta t_i}{2}
\Big[
(\bar H_i - H_i) + (\bar H_{i+1} - H_{i+1})
\Big],
$$

with:
- $\bar H_i = \bar H(P_i,X_i,t_i)$ via `bundle.evaluate`,
- $H_i \approx H(P_i,X_i,t_i)$ via `compute_H(..., restricted=True)`.

Implementation:

```python
eta_PA = 0.0
for i in range(N):
    Hbar_i, _   = bundle.evaluate(problem, P[i],     X[i],     t_nodes[i])
    Hbar_ip1, _ = bundle.evaluate(problem, P[i + 1], X[i + 1], t_nodes[i + 1])

    H_i, _      = compute_H(problem, P[i],     X[i],     t_nodes[i],     bundle.controls, restricted=True)
    H_ip1, _    = compute_H(problem, P[i + 1], X[i + 1], t_nodes[i + 1], bundle.controls, restricted=True)

    gap_i    = Hbar_i - H_i
    gap_ip1  = Hbar_ip1 - H_ip1
    dt       = t_nodes[i + 1] - t_nodes[i]
    eta_PA  += 0.5 * (gap_i + gap_ip1) * dt
```

**Expected sign:** since $\bar H$ minimizes over a smaller set than $H$, typically $\bar H \ge H$, so $\bar H - H \ge 0$, making $\eta_{\text{PA}}$ nonnegative.

---

### 5.3 Smoothing indicator $\eta_\delta$

This aims to quantify the discrepancy between the smooth Hamiltonian $H_\delta$ and the nonsmooth surrogate $\bar H$. The natural nonnegative quantity would be:
$$
\bar H - H_\delta \ge 0
\quad \text{(since soft-min satisfies } H_\delta \le \bar H\text{)}.
$$

However, the current code integrates:
$$
\eta_\delta^{\text{code}}
\approx
\int_0^T (H_\delta - \bar H)\,dt,
$$
which is typically **non-positive**.

Implementation (exactly as in the file):

```python
eta_delta = 0.0
for i in range(N):
    Hdelta_i, _, _   = eval_H_smooth(problem, bundle, P[i],     X[i],     t_nodes[i],     delta)
    Hdelta_ip1, _, _ = eval_H_smooth(problem, bundle, P[i + 1], X[i + 1], t_nodes[i + 1], delta)

    Hbar_i, _        = bundle.evaluate(problem, P[i],     X[i],     t_nodes[i])
    Hbar_ip1, _      = bundle.evaluate(problem, P[i + 1], X[i + 1], t_nodes[i + 1])

    diff_i    = Hdelta_i    - Hbar_i
    diff_ip1  = Hdelta_ip1  - Hbar_ip1
    dt        = t_nodes[i + 1] - t_nodes[i]
    eta_delta += 0.5 * (diff_i + diff_ip1) * dt
```

**Practical note:** if you want $\eta_\delta$ to behave like a standard nonnegative “error indicator”, you usually replace this with either:
- $\int_0^T (\bar H - H_\delta)\,dt$, or
- $\int_0^T |H_\delta - \bar H|\,dt$.

This matters because the refinement logic checks `if eta_delta > tol_delta:`.

---

## 6) Convergence check and logging

After computing indicators, the code records diagnostics:

```python
log.append({
    'iteration': k,
    'N': N,
    'M': bundle.num_planes(),
    'delta': delta,
    'eta_time': eta_time,
    'eta_PA': eta_PA,
    'eta_delta': eta_delta,
    'newton_iter': info['iterations'],
    'newton_residual': info['residual_norm'],
})
```

Stop if all indicators are below tolerances:

```python
if (eta_time <= tol_time) and (eta_PA <= tol_PA) and (eta_delta <= tol_delta):
    break
```

---

## 7) Adaptation actions (implementation details)

### 7.1 Time mesh refinement (highest priority)

If $\eta_{\text{time}}>\text{tol\_time}$, the code refines **all intervals** whose local indicator exceeds the tolerance by inserting a midpoint.

Implementation outline:

```python
new_nodes = [t_nodes[0]]
X_new = [X[0]]
P_new = [P[0]]

for i in range(N):
    dt = t_nodes[i + 1] - t_nodes[i]
    err = eta_time_local[i]
    if err > tol_time:
        t_mid = 0.5 * (t_nodes[i] + t_nodes[i + 1])

        alpha = (t_mid - t_nodes[i]) / dt  # = 0.5 here
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
continue
```

**Meaning:**
- The mesh is refined where the solution appears under-resolved.
- The new initial guess is generated by linear interpolation to keep Newton stable on the refined mesh.

---

### 7.2 Add a new PA plane (second priority)

If $\eta_{\text{time}}$ is acceptable but $\eta_{\text{PA}}>\text{tol\_PA}$, the algorithm enriches the bundle.

It finds the node $i$ with the largest gap
$$
\text{gap}_i = \bar H(P_i,X_i,t_i) - H(P_i,X_i,t_i),
$$
and retrieves a minimizing control candidate $u^*$ for $H$ at that node (from `compute_H`). Then it adds this $u^*$ as a new plane to the bundle.

Implementation:

```python
max_gap = -np.inf
max_idx = 0
for i in range(N + 1):
    Hbar_i, _ = bundle.evaluate(problem, P[i], X[i], t_nodes[i])
    H_i, u_star = compute_H(problem, P[i], X[i], t_nodes[i], bundle.controls, restricted=True)
    gap = Hbar_i - H_i
    if gap > max_gap:
        max_gap = gap
        max_idx = i
        best_u = u_star

if best_u is not None:
    bundle.add_control(best_u)

X_guess = X
P_guess = P
continue
```

**Meaning:**
- The bundle surrogate $\bar H$ is missing an important control direction at the worst-gap location.
- Adding $u^*$ locally improves the global surrogate quality in subsequent iterations.

---

### 7.3 Reduce smoothing $\delta$ (third priority)

If mesh and bundle are acceptable but smoothing is not, the code performs a continuation step:
$$
\delta \leftarrow \frac{\delta}{2}.
$$

Implementation:

```python
delta = delta * 0.5
X_guess = X
P_guess = P
continue
```

**Meaning:** progressively move from a smooth surrogate (large $\delta$) toward the nonsmooth min (small $\delta$), reusing the previous solution as a warm start.

---

## 8) Final return value

At the end, the function returns:

```python
return {
    't_nodes': t_nodes,
    'X': X,
    'P': P,
    'bundle': bundle,
    'delta': delta,
    'log': log
}
```

This gives you:
- the final adapted mesh,
- the final discrete solution $(X,P)$ on that mesh,
- the final bundle and smoothing parameter,
- a full history of indicator values and Newton diagnostics.

---
