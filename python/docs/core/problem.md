# `core/problem.py` — `OCPProblem`

This module defines the `OCPProblem` class: a minimal, explicit container for **problem-specific** information in a finite-horizon optimal control problem (OCP). All solver components (Hamiltonian evaluation, smoothing, symplectic discretization, Newton, adaptivity) interact with the OCP only through this interface so that the core algorithms remain **problem-agnostic**.

---

## 1) Mathematical model (Bolza form)

We consider a deterministic finite-horizon OCP on $[0,T]$ in **Bolza form**:

$$
\min_{u(\cdot)} J[u]
\;:=\;
g(x(T)) + \int_{0}^{T} \ell(x(t),u(t),t)\,dt,
$$

subject to the controlled ODE

$$
\dot{x}(t)=f(x(t),u(t),t), \qquad x(0)=x_0.
$$

Optional **box constraints** (componentwise bounds) may be enforced:

- **Control bounds:** $u(t)\in A := [u_{\min},u_{\max}] \subset \mathbb{R}^m$.
- **State bounds:** $x(t)\in K := [x_{\min},x_{\max}] \subset \mathbb{R}^n$.

> Practical note: In this repo, the solver *expects* a compact control set $A$ (a box) to seed the PA-bundle and to generate Hamiltonian candidates. If you omit `control_bounds`, you must ensure the solver can still infer `m` and initialize the bundle (otherwise the adaptivity loop will fail when evaluating $\bar H$).

---

## 2) Example 1 (LQR) as a running example

Example 1 (`experiments/ex1_lqr.py`) defines an LQR-like problem:

- Linear dynamics:
  $$
  \dot{x}(t)=Ax(t)+Bu(t),\qquad x(0)=x_0,
  $$
- Quadratic running and terminal costs:
  $$
  \ell(x,u,t)=x^\top Qx + u^\top Ru,
  \qquad
  g(x)=x^\top Q_f x.
  $$

In the repo, the problem is instantiated as:

```python
import numpy as np
from core.problem import OCPProblem

A = np.array([[0.0, 1.0], [0.0, 0.0]])
B = np.array([[0.0], [1.0]])

Q  = np.eye(2)
R  = 1e-2 * np.eye(1)
Qf = Q

x0 = np.array([1.0, 0.0])
T  = 1.0

def dynamics(x, u, t):
    return A.dot(x) + B.dot(u)

def stage_cost(x, u, t):
    return float(x.dot(Q.dot(x)) + u.T.dot(R).dot(u))

def terminal_cost(x):
    return float(x.dot(Qf.dot(x)))

# Approximate "unconstrained control" with large bounds
u_min = np.array([-5.0])
u_max = np.array([ 5.0])

prob = OCPProblem(
    dynamics, stage_cost, terminal_cost,
    x0=x0, T=T,
    control_bounds=(u_min, u_max),
    state_bounds=None
)
```

**Why bounds in LQR?** LQR is naturally unconstrained, but this solver builds Hamiltonian candidates from the **extreme points of the control box** (plus the bundle controls). Large bounds $[-5,5]$ approximate an unconstrained optimum while keeping the solver design consistent.

---

## 3) Constructor and stored fields

### Constructor

```python
OCPProblem(
    dynamics,             # f(x,u,t) -> xdot, shape (n,)
    stage_cost,           # ℓ(x,u,t) -> float
    terminal_cost,        # g(x)     -> float
    x0,                   # shape (n,)
    T,                    # float > 0
    control_bounds=None,  # (u_min, u_max), each shape (m,)
    state_bounds=None     # (x_min, x_max), each shape (n,)
)
```

### Stored attributes (conceptually)

- `f_fn`, `l_fn`, `g_fn`: the user callables implementing $f,\ell,g$
- `x0`, `T`
- `u_min`, `u_max` (or `None`)
- `x_min`, `x_max` (or `None`)

The implementation casts everything to `float` and **copies bounds** to avoid accidental mutation from outside.

---

## 4) Implementation walkthrough (minimal mental model)

This section explains what the key code blocks in `core/problem.py` do, focusing on what you need for debugging and maintenance.

### 4.1 `__init__`: store callables + sanitize inputs

The constructor does three things:

1. **Store function handles**
   - `self.f_fn = dynamics` for $f(x,u,t)$
   - `self.l_fn = stage_cost` for $\ell(x,u,t)$
   - `self.g_fn = terminal_cost` for $g(x)$

2. **Normalize core data**
   - `self.x0 = np.asarray(x0, dtype=float)`
   - `self.T  = float(T)`

3. **Normalize and copy bounds**
   - Convert to `float` and `.copy()` to protect against external mutation:
     - `self.u_min/u_max` from `control_bounds`
     - `self.x_min/x_max` from `state_bounds`

**Why the `.copy()` matters:** if someone later modifies `u_min` outside the class, the problem definition inside the solver stays unchanged.

**Typical failure modes:**
- Mismatched shapes, e.g. `u_min.shape=(m,)` but `u` passed around as `(m,1)` → silent broadcasting in comparisons/clipping.
- `x0` not 1D (`(n,1)` instead of `(n,)`) → downstream residual assembly becomes inconsistent.

---

### 4.2 Wrappers `f`, `l`, `g`: not “logic”, just a stable API

- `f(x,u,t)` returns `self.f_fn(x,u,t)`
- `l(x,u,t)` returns `self.l_fn(x,u,t)`
- `g(x)` returns `self.g_fn(x)`

These wrappers provide a **single standardized interface** for the solver modules.

**Typical failure modes:**
- `dynamics` returns shape `(n,1)` instead of `(n,)`.
- `stage_cost` returns a numpy scalar/array (e.g. shape `(1,1)`) rather than a Python `float`.

---

### 4.3 Control bounds utilities: feasibility + projection onto a box

#### `get_control_bounds()`
Returns **copies** of `(u_min, u_max)` or `None`.

#### `admissible_control(u, x=None, t=None)`
Checks componentwise feasibility:
$$
u_{\min} \le u \le u_{\max}.
$$
`x` and `t` are accepted for interface uniformity (unused for box-only checks).

#### `project_control(u)`
Clips the control onto the box:
$$
\Pi_A(u)_i = \min\{\max\{u_i,(u_{\min})_i\},(u_{\max})_i\}.
$$

**Why this matters in the solver:** Hamiltonian minimization uses candidate controls, and clipping keeps candidates within bounds.

**Typical failure modes (shape/broadcasting):**
- `u` is `(m,1)` but bounds are `(m,)`. Keep `u` as `(m,)` across the codebase.

---

### 4.4 State bounds utilities: simple box membership

#### `get_state_bounds()` and `admissible_state(x)`
Membership check:
$$
x_{\min} \le x \le x_{\max}.
$$

**Note:** This is only *instantaneous membership*. Viability across time steps is handled by the tangent cone filter below.

---

### 4.5 Dimensions: `n`, `m`, `state_dim`, `control_dim`

- `state_dim = self.x0.size` and `n` is an alias.

- `m` is inferred **only from control bounds**:
  - If `u_min` exists, `m = u_min.size`
  - Else if `u_max` exists, `m = u_max.size`
  - Else `m = None`

- `control_dim` behaves similarly but returns `0` if no bounds exist.

#### Important implementation note (debugging relevance)
The docstring suggests that `control_dim` could be inferred “the first time a control vector is passed to `f` or `l`”, but in the current implementation there is **no actual inference logic**. Therefore, if `control_bounds=None`, then:
- `m` stays `None`, and
- `control_dim` stays `0`.

**Practical consequence:** for this repo, you should **always provide** `control_bounds`, even for “unconstrained” problems (use large bounds as in Example 1).

---

### 4.6 Viability for state constraints: tangent cone filter (box-only)

When state constraints $x(t)\in K=[x_{\min},x_{\max}]$ exist, the solver can restrict candidate controls to those whose velocity does not point outside the box when on an active boundary.

For a box $K$, a simple tangent cone test is used:

- If $x_i \approx (x_{\min})_i$, require $\dot{x}_i \ge 0$.
- If $x_i \approx (x_{\max})_i$, require $\dot{x}_i \le 0$.

This implements a practical version of
$$
f(x,u,t)\in T_K(x),
$$
where $T_K(x)$ is the tangent cone to $K$ at $x$.

#### `tangent_cone_filter(x, f_candidates, tol=1e-8)`
Input: `f_candidates = [v_1, v_2, ...]`, where each $v_j$ is a candidate velocity vector.  
Output: boolean mask selecting which $v_j$ are viable.

- If `state_bounds=None`, everything is viable (all `True`).
- `tol` decides what counts as “on the boundary”.

#### `tangent_ok(x, u, t, tol=1e-8)`
Convenience wrapper:
- compute $v=f(x,u,t)$
- call `tangent_cone_filter(x, [v], tol)`

**Typical failure modes:**
- `tol` too small: boundary not detected → the method can step outside the box.
- `tol` too large: boundary detected too early → too many candidates filtered out, possibly leaving none.

---

## 5) API reference (methods) and mathematical meaning

### 5.1 Dynamics and costs

#### `f(x, u, t) -> np.ndarray`
Wrapper for the dynamics $f(x,u,t)=\dot{x}$.

Expected shapes:
- `x`: `(n,)`
- `u`: `(m,)`
- return: `(n,)`

#### `l(x, u, t) -> float`
Wrapper for the running cost $\ell(x,u,t)$.

#### `g(x) -> float`
Wrapper for the terminal cost $g(x)$.

---

### 5.2 Convenience accessors used by the adaptivity module

#### `control_bounds_tuple()` and `state_bounds_tuple()`
Aliases returning the same as `get_control_bounds()` and `get_state_bounds()`.  
These exist for compatibility and to keep solver code clean.

---

## 6) Debugging checklist (what to verify first)

When something breaks early (dimension errors, strange residuals, exploding Newton steps), check:

1. **Shapes**
   - `x0.shape == (n,)`
   - `u_min.shape == u_max.shape == (m,)`
   - if present: `x_min.shape == x_max.shape == (n,)`

2. **Return types**
   - `dynamics(x,u,t)` returns shape `(n,)`
   - `stage_cost(x,u,t)` returns a Python float (or `float(...)`-castable)
   - `terminal_cost(x)` returns a float

3. **Bounds presence**
   - Provide `control_bounds` always (large bounds for “unconstrained”).

4. **State constraints behavior (if any)**
   - If you see candidate depletion or constraint violations: inspect `tangent_cone_filter` and the choice of `tol`.

---

## 7) String representation

#### `__repr__`
Returns a compact summary such as:
`OCPProblem(state_dim=..., control_dim=..., T=..., bounds=yes/no)`.

Useful for debugging logs and quick inspection.

---

## 8) Where `OCPProblem` is used in the solver

- `core.adaptivity.solve_optimal_control`: reads `n`, `m`, bounds, and calls `f`, `l`, `g`.
- `core.hamiltonian.compute_H`: uses `control_bounds_tuple`, `project_control`, `admissible_control`, and optionally `tangent_ok`.
- `core.smoothing.eval_H_smooth`: repeatedly calls `f` and `l` while building the soft-min approximation.
- `core.newton.solve_tpbvp` and `core.integrators.*`: call `f`, `l`, and (via finite differences) `g`.

This clean separation is the reason the solver can be reused across very different examples (LQR, constrained double integrator, Dubins).
