# `core/problem.py` — Problem definition interface (`OCPProblem`)

This module defines the class

```python
OCPProblem
```

which is the central container for problem-specific information in the solver.

The numerical algorithms in the rest of the repository are designed to be
problem-agnostic. They do not know the formulas for the dynamics, cost,
constraints, Hamiltonian, or feasibility logic. Instead, they call methods on an
`OCPProblem` instance.

Thus `OCPProblem` is the interface between:

1. the mathematical optimal control problem;
2. the Hamiltonian and PA-bundle routines;
3. the smoothed PMP residual/Jacobian assembly;
4. the Newton solver;
5. the adaptive mesh/bundle/smoothing loop.

---

## 1) Mathematical model

The base problem is a finite-horizon optimal control problem in Bolza form:

$$
\begin{equation}
\min_{u(\cdot)}
\left[
g(x(T))
+
\int_0^T
\ell(x(t),u(t),t)\,dt
\right],
\end{equation}
$$

subject to

$$
\begin{equation}
\dot x(t)
=
f(x(t),u(t),t),
\qquad
x(0)=x_0.
\end{equation}
$$

Here:

- $x(t)\in\mathbb{R}^n$ is the state;
- $u(t)\in\mathbb{R}^m$ is the control;
- $f$ is the dynamics;
- $\ell$ is the running cost;
- $g$ is the terminal cost;
- $T$ is the final time.

The class also supports optional box bounds on controls and states:

$$
\begin{equation}
u_{\min}
\le
u(t)
\le
u_{\max},
\end{equation}
$$

and

$$
\begin{equation}
x_{\min}
\le
x(t)
\le
x_{\max}.
\end{equation}
$$

These bounds are componentwise.

---

## 2) Current constructor

The current constructor is:

```python
def __init__(
    self,
    dynamics,
    stage_cost,
    terminal_cost,
    x0,
    T,
    control_bounds=None,
    state_bounds=None,
    hamiltonian_true=None,
    u_star_fn=None,
    hamiltonian_grad_fn=None,
    hamiltonian_smooth_fn=None,
    barrier_stage_cost_fn=None,
    barrier_grad_x_fn=None,
    barrier_margin_fn=None,
    u_star_local_fn=None,
    tangent_ok_fn=None,
    step_feasible_control_fn=None,
    feasibility_refinement_fn=None,
    state_feasible_fn=None,
    project_state_fn=None,
    fraction_to_boundary_fn=None,
):
    ...
```

The required arguments are:

| Argument | Meaning |
|---|---|
| `dynamics` | Function implementing $f(x,u,t)$. |
| `stage_cost` | Function implementing $\ell(x,u,t)$. |
| `terminal_cost` | Function implementing $g(x)$. |
| `x0` | Initial state. |
| `T` | Final time horizon. |

All other arguments are optional hooks used by advanced solver features.

---

## 3) Constructor argument groups

The constructor arguments can be understood in five groups.

---

### 3.1 Core OCP data

```python
dynamics
stage_cost
terminal_cost
x0
T
```

These define the baseline Bolza problem.

The constructor stores them as:

```python
self.f_fn = dynamics
self.l_fn = stage_cost
self.g_fn = terminal_cost
self.x0 = np.asarray(x0, dtype=float)
self.T = float(T)
```

The methods `f`, `l`, and `g` later call these stored functions.

---

### 3.2 Box constraints

```python
control_bounds=None
state_bounds=None
```

If control bounds are supplied, they should be a pair

```python
(u_min, u_max)
```

where each entry is a one-dimensional NumPy-compatible array of shape `(m,)`.

The constructor copies them:

```python
self.u_min = np.asarray(u_min, dtype=float).copy()
self.u_max = np.asarray(u_max, dtype=float).copy()
```

If state bounds are supplied, they should be a pair

```python
(x_min, x_max)
```

with arrays of shape `(n,)`, and they are copied similarly.

The copies are important: changing the original arrays outside the class does
not mutate the problem definition stored inside `OCPProblem`.

If bounds are not provided, the corresponding internal fields are set to `None`.

---

### 3.3 Hamiltonian and oracle hooks

```python
hamiltonian_true=None
u_star_fn=None
u_star_local_fn=None
hamiltonian_grad_fn=None
hamiltonian_smooth_fn=None
```

These optional callbacks allow the problem to provide more information than just
$f$, $\ell$, and $g$.

They are used by different parts of the solver:

| Hook | Main purpose |
|---|---|
| `hamiltonian_true` | Provide a problem-specific true or reference Hamiltonian value. |
| `u_star_fn` | Provide a pointwise optimal control candidate $u^\star(x,p,t)$. |
| `u_star_local_fn` | Provide a local, possibly step-size-dependent optimal control candidate. |
| `hamiltonian_grad_fn` | Provide explicit gradients $(H_p,H_x)$. |
| `hamiltonian_smooth_fn` | Provide a custom smoothed Hamiltonian and gradients. |

These hooks are optional. If they are absent, the solver falls back to generic
PA-bundle, smoothing, and candidate-minimization routines.

---

### 3.4 Barrier hooks

```python
barrier_stage_cost_fn=None
barrier_grad_x_fn=None
barrier_margin_fn=None
```

These hooks support barrier-style formulations, used for some constrained
examples.

The problem stores a barrier parameter:

```python
self.mu_barrier = 0.0
```

When `self.mu_barrier > 0` and a barrier stage-cost callback is provided, the
stage cost method `l(...)` adds the barrier contribution.

The barrier hooks are:

| Hook | Meaning |
|---|---|
| `barrier_stage_cost_fn` | Extra barrier running cost. |
| `barrier_grad_x_fn` | Optional barrier gradient with respect to state. |
| `barrier_margin_fn` | Function returning a margin to the active barrier/constraint. |

---

### 3.5 Feasibility and projection hooks

```python
tangent_ok_fn=None
step_feasible_control_fn=None
feasibility_refinement_fn=None
state_feasible_fn=None
project_state_fn=None
fraction_to_boundary_fn=None
```

These hooks customize constrained-problem behavior.

They are used by:

- `compute_H` and `PABundle.evaluate`, through local control feasibility;
- `adaptivity.py`, for feasibility-driven mesh refinement;
- `newton.py`, for feasibility-aware line search and projection;
- state and trajectory feasibility checks.

The hooks are:

| Hook | Main purpose |
|---|---|
| `tangent_ok_fn` | Custom tangent-cone or viability check for a control. |
| `step_feasible_control_fn` | Check whether a control is feasible over a local time step. |
| `feasibility_refinement_fn` | Ask the adaptive loop to refine intervals for problem-specific feasibility reasons. |
| `state_feasible_fn` | Custom state feasibility predicate. |
| `project_state_fn` | Project a state back into the feasible set. |
| `fraction_to_boundary_fn` | Compute a safe Newton step fraction before crossing a boundary. |

If these hooks are absent, the class uses default box-based behavior where
possible.

---

## 4) Stored attributes

After construction, the class stores the required callables and optional hooks
as attributes.

Core attributes:

```python
self.f_fn
self.l_fn
self.g_fn
self.x0
self.T
```

Control-bound attributes:

```python
self.u_min
self.u_max
```

State-bound attributes:

```python
self.x_min
self.x_max
```

Hamiltonian/oracle attributes:

```python
self.hamiltonian_true_fn
self.u_star_fn
self.u_star_local_fn
self.hamiltonian_grad_fn
self.hamiltonian_smooth_fn
```

Barrier attributes:

```python
self.barrier_stage_cost_fn
self.barrier_grad_x_fn
self.barrier_margin_fn
self.mu_barrier
```

Feasibility/projection attributes:

```python
self.tangent_ok_fn
self.step_feasible_control_fn
self.feasibility_refinement_fn
self.state_feasible_fn
self.project_state_fn
self.fraction_to_boundary_fn
```

This design makes `OCPProblem` a single object carrying both the mathematical
problem and the optional solver-specific callbacks.

---

## 5) Minimal example: LQR-style problem

A simple LQR-style problem can be built with only the core data and control
bounds.

```python
import numpy as np
from core.problem import OCPProblem

A = np.array([[0.0, 1.0], [0.0, 0.0]])
B = np.array([[0.0], [1.0]])

Q = np.eye(2)
R = 1e-2 * np.eye(1)
Qf = Q

x0 = np.array([1.0, 0.0])
T = 1.0

def dynamics(x, u, t):
    return A.dot(x) + B.dot(u)

def stage_cost(x, u, t):
    return float(x.dot(Q.dot(x)) + u.T.dot(R).dot(u))

def terminal_cost(x):
    return float(x.dot(Qf.dot(x)))

u_min = np.array([-5.0])
u_max = np.array([5.0])

problem = OCPProblem(
    dynamics=dynamics,
    stage_cost=stage_cost,
    terminal_cost=terminal_cost,
    x0=x0,
    T=T,
    control_bounds=(u_min, u_max),
    state_bounds=None,
)
```

The bounds make the control set compact, which is useful for PA-bundle seeding
and Hamiltonian candidate generation. For scalar bounded controls, the current
Hamiltonian routine can also perform bounded scalar minimization inside those
bounds.

---

## 6) Core wrappers: `f`, `l`, and `g`

The solver never calls the raw user functions directly. Instead, it uses the
standard methods

```python
problem.f(x, u, t)
problem.l(x, u, t)
problem.g(x)
```

These methods provide a stable API for all solver modules.

---

### 6.1 Dynamics wrapper `f`

The method

```python
def f(self, x: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
    return np.asarray(self.f_fn(x, u, t), dtype=float)
```

returns the state velocity

$$
\begin{equation}
f(x,u,t)
=
\dot x.
\end{equation}
$$

The result is converted to a floating-point NumPy array. The expected shape is

```python
(n,)
```

where $n$ is the state dimension.

---

### 6.2 Stage-cost wrapper `l`

The method

```python
def l(self, x: np.ndarray, u: np.ndarray, t: float) -> float:
    base = float(self.l_fn(x, u, t))
    if self.mu_barrier > 0.0 and self.barrier_stage_cost_fn is not None:
        base += float(self.barrier_stage_cost_fn(x, u, t, self.mu_barrier))
    return base
```

returns the running cost

$$
\begin{equation}
\ell(x,u,t).
\end{equation}
$$

If no barrier is active, this is simply the user-provided stage cost.

If

```python
self.mu_barrier > 0.0
```

and

```python
self.barrier_stage_cost_fn is not None
```

then the method adds a barrier contribution:

$$
\begin{equation}
\ell_{\mathrm{total}}(x,u,t)
=
\ell_{\mathrm{base}}(x,u,t)
+
\ell_{\mathrm{barrier}}(x,u,t;\mu).
\end{equation}
$$

The barrier parameter is stored as

```python
self.mu_barrier
```

and is initialized to zero in the constructor.

---

### 6.3 Terminal-cost wrapper `g`

The method

```python
def g(self, x: np.ndarray) -> float:
    return float(self.g_fn(x))
```

returns the terminal cost

$$
\begin{equation}
g(x).
\end{equation}
$$

This is used in the terminal boundary condition and in mesh-objective
diagnostics.

---

## 7) Oracle control: `u_star`

The method

```python
def u_star(
    self,
    x: np.ndarray,
    p: np.ndarray,
    t: float,
    restricted: bool = False,
    tol: float = 1e-8,
    dt: Optional[float] = None,
):
    ...
```

returns a problem-specific pointwise Hamiltonian minimizer when such a callback
is available.

It has two possible callback paths.

---

### 7.1 Local oracle path

If

```python
self.u_star_local_fn is not None
```

then the method calls

```python
return self.u_star_local_fn(x, p, t, restricted, tol, dt)
```

This local oracle can depend on:

- state $x$;
- costate $p$;
- time $t$;
- whether restricted feasibility is requested;
- tolerance `tol`;
- local step size `dt`.

It is useful when the locally optimal control depends on step-size-dependent
feasibility or other local restrictions.

The expected return is

```python
u, ok
```

where `u` is either a control vector or `None`, and `ok` indicates whether the
control is acceptable under the requested restriction.

---

### 7.2 Standard oracle path

If no local oracle is provided but

```python
self.u_star_fn is not None
```

then the method calls

```python
u = self.u_star_fn(x, p, t)
```

The result is converted to a floating-point array and projected to the control
bounds if bounds exist:

```python
u = np.asarray(u, dtype=float)
if self.u_min is not None:
    u = self.project_control(u)
```

If `restricted=True`, the method also checks

```python
self.tangent_ok(x, u, t, tol=tol)
```

and returns

```python
u, ok
```

where `ok` is the tangent/viability result.

If `restricted=False`, the method returns

```python
u, True
```

---

### 7.3 No oracle available

If neither oracle callback exists, the method returns

```python
None, False
```

This tells the caller to fall back to generic candidate minimization.

---

## 8) True/reference Hamiltonian: `hamiltonian_true`

The method

```python
def hamiltonian_true(
    self,
    x: np.ndarray,
    p: np.ndarray,
    t: float,
    restricted: bool = False,
    tol: float = 1e-8,
    dt: Optional[float] = None,
) -> Optional[float]:
    ...
```

returns a problem-specific Hamiltonian value when available.

If

```python
self.hamiltonian_true_fn is not None
```

the callback is used directly:

```python
return self.hamiltonian_true_fn(x, p, t)
```

Otherwise, the method tries to use `u_star(...)`:

```python
u, ok = self.u_star(x, p, t, restricted=restricted, tol=tol, dt=dt)
```

If `u` is available and either the problem is unrestricted or `ok=True`, it
evaluates

$$
\begin{equation}
H(p,x,t)
=
p^\top f(x,u,t)
+
\ell(x,u,t).
\end{equation}
$$

In code:

```python
return float(np.dot(p, self.f(x, u, t)) + self.l(x, u, t))
```

If neither a true Hamiltonian callback nor a valid oracle control is available,
the method returns

```python
None
```

This method is a convenience/reference hook. The generic Hamiltonian routine
`compute_H` can still build candidate controls without it.

---

## 9) Explicit Hamiltonian gradients: `hamiltonian_gradients`

The method

```python
def hamiltonian_gradients(
    self,
    x: np.ndarray,
    p: np.ndarray,
    t: float,
):
    ...
```

returns explicit Hamiltonian gradients when the problem provides them.

If no callback was supplied, the method raises

```python
ValueError("No Hamiltonian gradient function provided for this problem.")
```

Otherwise, it calls

```python
return self.hamiltonian_grad_fn(x, p, t)
```

The expected return is

```python
Hp, Hx
```

where

$$
\begin{equation}
H_p
=
\nabla_p H(p,x,t),
\qquad
H_x
=
\nabla_x H(p,x,t).
\end{equation}
$$

These gradients are used by `integrators.py` and `adaptivity.py` when
`use_explicit_hamiltonian_gradients=True`.

---

## 10) Custom smoothed Hamiltonian: `hamiltonian_smooth`

The method

```python
def hamiltonian_smooth(
    self,
    x: np.ndarray,
    p: np.ndarray,
    t: float,
    delta: float,
):
    ...
```

returns a problem-specific smoothed Hamiltonian and gradients.

If no callback was supplied, it raises

```python
ValueError("No smooth Hamiltonian function provided for this problem.")
```

Otherwise, it calls

```python
return self.hamiltonian_smooth_fn(x, p, t, delta)
```

The expected return is

```python
H_delta, Hp, Hx
```

where

$$
\begin{equation}
H_\delta
\end{equation}
$$

is a smoothed Hamiltonian value, and

$$
\begin{equation}
H_p=\nabla_p H_\delta,
\qquad
H_x=\nabla_x H_\delta.
\end{equation}
$$

The default `eval_H_smooth` routine in `smoothing.py` checks for this callback
and delegates to it if present.

---

## 11) Barrier margin

The method

```python
def barrier_margin(self, x: np.ndarray, t: float):
    if self.barrier_margin_fn is None:
        return None
    return self.barrier_margin_fn(x, t)
```

returns a problem-specific barrier or constraint margin when available.

If no barrier margin callback was supplied, the method returns `None`.

This is mainly used for constrained examples and diagnostics, where one wants to
track distance to an active constraint or barrier boundary.

---

## 12) Control bounds

The method

```python
def get_control_bounds(self):
    ...
```

returns the stored control bounds.

If bounds are available, it returns copies:

```python
return self.u_min.copy(), self.u_max.copy()
```

If no control bounds were supplied, it returns

```python
None
```

The method

```python
def control_bounds_tuple(self):
    ...
```

returns the internal bounds as a tuple

```python
(self.u_min, self.u_max)
```

or `None` if bounds are absent.

The difference is:

- `get_control_bounds` returns copies, safer for external use;
- `control_bounds_tuple` returns the stored arrays directly, useful internally
  when the caller does not mutate them.

---

## 13) Control admissibility

The method

```python
def admissible_control(
    self,
    u: np.ndarray,
    x: Optional[np.ndarray] = None,
    t: float = 0.0,
) -> bool:
    ...
```

checks whether a control satisfies the control box bounds.

If no control bounds are provided, every control is considered admissible:

```python
if self.u_min is None:
    return True
```

If bounds exist, the method checks componentwise inequalities with a small
tolerance:

$$
\begin{equation}
u_{\min}-10^{-12}
\le
u
\le
u_{\max}+10^{-12}.
\end{equation}
$$

In code:

```python
return bool(
    np.all(u_arr >= self.u_min - 1e-12)
    and np.all(u_arr <= self.u_max + 1e-12)
)
```

The optional arguments `x` and `t` are accepted for interface compatibility, but
the default implementation does not use them.

---

## 14) Control projection

The method

```python
def project_control(self, u: np.ndarray) -> np.ndarray:
    ...
```

projects a control onto the control box.

If no control bounds are available, it simply returns the input as a float array:

```python
return np.asarray(u, dtype=float)
```

If bounds exist, it applies componentwise clipping:

```python
return np.minimum(np.maximum(u, self.u_min), self.u_max)
```

Mathematically,

$$
\begin{equation}
(\Pi_A(u))_j
=
\min
\left\{
\max\left\{u_j,(u_{\min})_j\right\},
(u_{\max})_j
\right\}.
\end{equation}
$$

This projection is used by Hamiltonian candidate routines to keep oracle or
candidate controls inside the admissible box.

---

## 15) Step-feasible controls

The method

```python
def step_feasible_control(
    self,
    x: np.ndarray,
    u: np.ndarray,
    t: float,
    dt: Optional[float],
    tol: float = 1e-10,
) -> bool:
    ...
```

checks whether a control is feasible over a local step of length `dt`.

If no step-feasibility callback was supplied, the method returns

```python
True
```

Otherwise it calls

```python
self.step_feasible_control_fn(x, u, t, dt, tol)
```

and returns its boolean value.

This hook is useful for problems where feasibility is not purely pointwise in
$(x,u,t)$, but depends on whether the state can safely move over a finite time
step $\Delta t$.

For example, a control may be tangent-feasible at the current state but still
violate a path constraint over a large step.

---

## 16) Local control feasibility

The method

```python
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
    ...
```

is the main local control feasibility interface used by the Hamiltonian and PA
bundle routines.

It combines three checks:

1. box admissibility;
2. tangent/viability feasibility when `restricted=True`;
3. optional step-feasibility when `dt` is provided.

The current logic is:

```python
if not self.admissible_control(u, x, t):
    return False

if restricted and not self.tangent_ok(x, u, t, tol=tol):
    return False

if dt is not None and not self.step_feasible_control(x, u, t, dt, tol=tol):
    return False

return True
```

Thus a control is locally feasible only if it satisfies all active tests.

Mathematically, when `restricted=True`, this represents a local admissible set
of the form

$$
\begin{equation}
A_{\mathrm{loc}}(x,t,\Delta t)
=
\left\{
u:
u\in A,
\quad
f(x,u,t)\in T_K(x),
\quad
u \text{ passes step-feasibility}
\right\}.
\end{equation}
$$

When `restricted=False`, the tangent-cone check is skipped, but box
admissibility and step-feasibility may still apply.

This method is called by:

- `compute_H` in `hamiltonian.py`;
- `PABundle.evaluate` in `pa_bundle.py`;
- feasibility-refinement logic in `adaptivity.py`.

---

## 17) Tangent-cone filter

The method

```python
def tangent_cone_filter(
    self,
    x: np.ndarray,
    f_candidates: Sequence[np.ndarray],
    tol: float = 1e-8,
) -> np.ndarray:
    ...
```

checks whether candidate state velocities lie in the tangent cone of the box
state constraint set.

If no state bounds are defined, all candidates are considered viable:

```python
if self.x_min is None or self.x_max is None:
    return np.ones(len(f_candidates), dtype=bool)
```

Assume state bounds define

$$
\begin{equation}
K
=
[x_{\min},x_{\max}].
\end{equation}
$$

For each candidate velocity $v=f(x,u,t)$, the default box tangent rule is:

- if $x_j$ is on the lower boundary, the velocity must not point downward;
- if $x_j$ is on the upper boundary, the velocity must not point upward.

Using tolerance `tol`, the lower-bound rejection condition is

$$
\begin{equation}
|x_j-(x_{\min})_j|<\mathrm{tol}
\quad
\text{and}
\quad
v_j<0.
\end{equation}
$$

The upper-bound rejection condition is

$$
\begin{equation}
|x_j-(x_{\max})_j|<\mathrm{tol}
\quad
\text{and}
\quad
v_j>0.
\end{equation}
$$

If either condition holds for any component, the velocity is marked infeasible.

The method returns a boolean mask with one entry per candidate velocity.

---

## 18) Tangent feasibility for one control: `tangent_ok`

The method

```python
def tangent_ok(
    self,
    x: np.ndarray,
    u: np.ndarray,
    t: float,
    tol: float = 1e-8,
) -> bool:
    ...
```

checks tangent feasibility for a single control.

If a custom callback was supplied,

```python
self.tangent_ok_fn is not None
```

then the method calls

```python
self.tangent_ok_fn(x, u, t, tol)
```

and returns its boolean value.

Otherwise, if no state bounds are defined, it returns `True`.

If state bounds are present, it computes the velocity

$$
\begin{equation}
v=f(x,u,t),
\end{equation}
$$

and calls

```python
self.tangent_cone_filter(x, [v], tol)
```

The result is `True` if the velocity lies in the tangent cone of the box
constraint set at the current state.

---

## 19) How the control feasibility methods interact

The control-related methods form a hierarchy:

```python
admissible_control(u)
    checks box bounds

project_control(u)
    clips to box bounds

tangent_ok(x, u, t)
    checks state-bound viability

step_feasible_control(x, u, t, dt)
    checks optional finite-step feasibility

local_control_feasible(x, u, t, restricted, dt)
    combines admissibility, tangent feasibility, and step feasibility
```

The most important method for the rest of the solver is
`local_control_feasible`. It is the one used when selecting Hamiltonian
minimizers under restrictions.

The projection method is separate: a projected control is not automatically
guaranteed to be tangent-feasible or step-feasible. Projection only enforces
control bounds.

---

## 20) State bounds

The method

```python
def get_state_bounds(self):
    ...
```

returns copies of the state bounds when they exist:

```python
return self.x_min.copy(), self.x_max.copy()
```

If no state bounds were provided, it returns

```python
None
```

The method

```python
def state_bounds_tuple(self):
    ...
```

returns the internal state-bound tuple

```python
(self.x_min, self.x_max)
```

or `None` if state bounds are absent.

As with control bounds:

- `get_state_bounds` is safer for external use because it returns copies;
- `state_bounds_tuple` is an internal convenience method.

---

## 21) State admissibility

The method

```python
def admissible_state(self, x: np.ndarray) -> bool:
    ...
```

checks whether a state satisfies the state box bounds.

If no state bounds are available, every state is considered admissible:

```python
if self.x_min is None:
    return True
```

If bounds exist, the method checks

$$
\begin{equation}
x_{\min}-10^{-10}
\le
x
\le
x_{\max}+10^{-10}.
\end{equation}
$$

In code:

```python
return bool(
    np.all(x_arr >= self.x_min - 1e-10)
    and np.all(x_arr <= self.x_max + 1e-10)
)
```

This method is a simple box-feasibility check.

---

## 22) State feasibility callback

The method

```python
def state_feasible(
    self,
    x: np.ndarray,
    t: float = 0.0,
    tol: float = 1e-10,
) -> bool:
    ...
```

checks whether a state is feasible at a time $t$.

If a custom callback was provided,

```python
self.state_feasible_fn is not None
```

then the method calls

```python
self.state_feasible_fn(x, t, tol)
```

Otherwise, it falls back to

```python
self.admissible_state(x)
```

This allows examples to define feasibility conditions that are more general than
componentwise box bounds.

---

## 23) Trajectory feasibility

The method

```python
def trajectory_feasible(
    self,
    X: np.ndarray,
    t_nodes: np.ndarray,
    tol: float = 1e-10,
) -> bool:
    ...
```

checks whether every state node in a trajectory is feasible.

It loops through the mesh nodes:

```python
for x_i, t_i in zip(X, t_nodes):
    if not self.state_feasible(x_i, float(t_i), tol=tol):
        return False
return True
```

Mathematically, it checks

$$
\begin{equation}
X_i\in K(t_i)
\qquad
\text{for all } i=0,\dots,N,
\end{equation}
$$

where $K(t_i)$ is the feasible state set implied by `state_feasible`.

This method is used by the Newton line search to reject trial steps that produce
infeasible trajectories.

---

## 24) State projection

The method

```python
def project_state(
    self,
    x: np.ndarray,
    t: float = 0.0,
    tol: float = 1e-12,
) -> np.ndarray:
    ...
```

projects a state back into the feasible set.

If a custom projection callback exists,

```python
self.project_state_fn is not None
```

then the method calls

```python
self.project_state_fn(x, t, tol)
```

Otherwise, if no state bounds exist, it returns the state as a float array.

If state bounds exist, the default projection clips componentwise:

$$
\begin{equation}
(\Pi_K(x))_j
=
\min
\left\{
\max\left\{x_j,(x_{\min})_j\right\},
(x_{\max})_j
\right\}.
\end{equation}
$$

This is the state analogue of `project_control`.

---

## 25) Trajectory projection

The method

```python
def project_trajectory(
    self,
    X: np.ndarray,
    t_nodes: np.ndarray,
    tol: float = 1e-12,
) -> np.ndarray:
    ...
```

projects every state node in a trajectory:

```python
return np.vstack([
    self.project_state(x_i, float(t_i), tol=tol)
    for x_i, t_i in zip(X, t_nodes)
])
```

This method is used by `newton.py` as a fallback when a trial Newton step fails
or when the least-squares fallback returns an infeasible trajectory.

The projection only changes the state trajectory. Costates are handled by the
caller when repacking the Newton vector.

---

## 26) Fraction-to-boundary step

The method

```python
def fraction_to_boundary_step(
    self,
    X: np.ndarray,
    dX: np.ndarray,
    t_nodes: np.ndarray,
    safety: float = 0.99,
    tol: float = 1e-12,
) -> float:
    ...
```

computes a safe fraction of a Newton step before crossing state bounds.

This method is used in `newton.py` to initialize the damping parameter for a
trial step.

If a custom callback exists,

```python
self.fraction_to_boundary_fn is not None
```

then the method delegates to it:

```python
return float(self.fraction_to_boundary_fn(X, dX, t_nodes, safety, tol))
```

If no state bounds exist, it returns

```python
1.0
```

meaning the full step is allowed from the standpoint of box bounds.

---

### 26.1 Default box-bound computation

For box bounds, the method examines each component of each state node.

For a trial step

$$
\begin{equation}
X^{trial}
=
X+\alpha dX,
\end{equation}
$$

it seeks a fraction $\alpha$ that keeps the state inside the box.

For each component, if the direction is positive,

$$
\begin{equation}
dX_{ij}>0,
\end{equation}
$$

then the upper bound gives

$$
\begin{equation}
\alpha
\le
\frac{(x_{\max})_j-X_{ij}}{dX_{ij}}.
\end{equation}
$$

If the direction is negative,

$$
\begin{equation}
dX_{ij}<0,
\end{equation}
$$

then the lower bound gives

$$
\begin{equation}
\alpha
\le
\frac{(x_{\min})_j-X_{ij}}{dX_{ij}}.
\end{equation}
$$

The code computes the minimum such admissible ratio and multiplies it by a
safety factor:

$$
\begin{equation}
\alpha_{\mathrm{safe}}
=
\mathrm{safety}
\cdot
\min(\text{admissible ratios}).
\end{equation}
$$

The returned value is clipped to the interval $[0,1]$:

$$
\begin{equation}
0
\le
\alpha_{\mathrm{safe}}
\le
1.
\end{equation}
$$

This prevents Newton's line search from starting with a step that immediately
crosses a state boundary.

---

## 27) How state feasibility methods interact with Newton

The state-related methods form the feasibility layer used by `newton.py`.

```python
trajectory_feasible(X_trial, t_nodes)
```

is used to reject infeasible line-search trial steps.

```python
fraction_to_boundary_step(X_curr, dX, t_nodes)
```

is used to choose an initial safe damping parameter.

```python
project_trajectory(X_trial, t_nodes)
```

is used as a fallback when a trial trajectory is infeasible or when the
least-squares fallback returns an infeasible trajectory.

Thus, `OCPProblem` supplies the Newton solver with all state-feasibility logic.
The Newton solver does not need to know whether feasibility comes from simple box
bounds or custom problem-specific callbacks.

---

## 28) Dimension helpers

The class provides several properties and helper methods for state and control
dimensions.

---

### 28.1 State dimension

The property

```python
@property
def state_dim(self) -> int:
    return self.x0.size
```

returns the state dimension

$$
\begin{equation}
n=\dim(x).
\end{equation}
$$

The property

```python
@property
def n(self) -> int:
    return self.state_dim
```

is an alias used for compatibility.

Thus,

```python
problem.n == problem.state_dim == problem.x0.size
```

---

### 28.2 Control dimension

The property

```python
@property
def m(self) -> Optional[int]:
    ...
```

returns the control dimension when it can be inferred from control bounds.

If `u_min` exists, then

```python
return self.u_min.size
```

If `u_max` exists, then

```python
return self.u_max.size
```

If no control bounds are available, it returns

```python
None
```

The property

```python
@property
def control_dim(self) -> int:
    ...
```

also tries to infer the control dimension from bounds, but returns `0` when
bounds are absent.

Thus:

```python
problem.m
```

is useful when the caller wants `None` to mean "unknown", while

```python
problem.control_dim
```

always returns an integer.

This distinction matters in `adaptivity.py`, where the initial PA bundle can be
seeded only if the control dimension is known.

---

## 29) String representation

The method

```python
def __repr__(self) -> str:
    return (
        f"OCPProblem(state_dim={self.state_dim}, "
        f"control_dim={self.control_dim}, "
        f"T={self.T}, "
        f"bounds={'yes' if self.u_min is not None else 'no'})"
    )
```

returns a compact summary such as

```python
OCPProblem(state_dim=2, control_dim=1, T=1.0, bounds=yes)
```

This is mainly for debugging and logging.

---

## 30) How `OCPProblem` connects to the solver modules

`OCPProblem` is used throughout the core solver. The table below summarizes the
main interactions.

| Module | How it uses `OCPProblem` |
|---|---|
| `hamiltonian.py` | Calls `f`, `l`, `project_control`, `u_star`, and `local_control_feasible` when computing Hamiltonian candidate minima. |
| `pa_bundle.py` | Calls `f`, `l`, and `local_control_feasible` when evaluating the PA-bundle surrogate. |
| `smoothing.py` | Calls `f`, `l`, and optionally `hamiltonian_smooth`. |
| `integrators.py` | Calls smoothed or explicit Hamiltonian gradients; uses `g` for the terminal condition. |
| `newton.py` | Calls `trajectory_feasible`, `project_trajectory`, and `fraction_to_boundary_step` for feasibility-aware damping. |
| `adaptivity.py` | Uses bounds, oracle controls, explicit gradients, local feasibility, feasibility-refinement callbacks, state projection, and objective evaluation. |

This is why `problem.py` is the central problem-specific interface: all
mathematical details and optional customizations enter the solver through this
class.

---

## 31) Callback contracts

Because many constructor arguments are optional callbacks, it is important that
their signatures and return values match what the solver expects.

---

### 31.1 Required callbacks

The required callbacks are:

```python
dynamics(x, u, t) -> np.ndarray
stage_cost(x, u, t) -> float
terminal_cost(x) -> float
```

Expected shapes:

```python
dynamics(x, u, t).shape == (n,)
x.shape == (n,)
u.shape == (m,)
```

The costs should be scalar or convertible to `float`.

---

### 31.2 Oracle control callbacks

The standard oracle control callback is

```python
u_star_fn(x, p, t) -> np.ndarray
```

The local oracle callback is

```python
u_star_local_fn(x, p, t, restricted, tol, dt) -> (u_or_none, ok)
```

where `ok` is a boolean indicating whether the returned control is acceptable
under the requested local restriction.

---

### 31.3 Hamiltonian callbacks

The true/reference Hamiltonian callback is

```python
hamiltonian_true(x, p, t) -> float
```

The explicit-gradient callback is

```python
hamiltonian_grad_fn(x, p, t) -> (Hp, Hx)
```

where

```python
Hp.shape == (n,)
Hx.shape == (n,)
```

The custom smoothed Hamiltonian callback is

```python
hamiltonian_smooth_fn(x, p, t, delta) -> (H_delta, Hp, Hx)
```

---

### 31.4 Barrier callbacks

Barrier callbacks have the following expected signatures:

```python
barrier_stage_cost_fn(x, u, t, mu) -> float
barrier_grad_x_fn(x, t, mu) -> np.ndarray
barrier_margin_fn(x, t) -> float or diagnostic object
```

The current `l(...)` method uses `barrier_stage_cost_fn` when `mu_barrier > 0`.

---

### 31.5 Feasibility and projection callbacks

The feasibility/projection callbacks are expected to behave like:

```python
tangent_ok_fn(x, u, t, tol) -> bool

step_feasible_control_fn(x, u, t, dt, tol) -> bool

feasibility_refinement_fn(x, p, t, dt, tol) -> None or dict

state_feasible_fn(x, t, tol) -> bool

project_state_fn(x, t, tol) -> np.ndarray

fraction_to_boundary_fn(X, dX, t_nodes, safety, tol) -> float
```

The returned fraction-to-boundary value should usually lie in $[0,1]$.

---

## 32) Debugging checklist

When a problem behaves unexpectedly, check the following points.

---

### 32.1 Shapes

Use one-dimensional arrays:

```python
x.shape == (n,)
u.shape == (m,)
p.shape == (n,)
```

Avoid column vectors such as `(n, 1)` or `(m, 1)`, because NumPy broadcasting can
silently create wrong dynamics, costs, or dot products.

---

### 32.2 Dynamics output

Check that

```python
problem.f(x, u, t).shape == (n,)
```

If dynamics returns a scalar, column vector, or list with inconsistent length,
the residual assembly can fail or produce incorrect Jacobians.

---

### 32.3 Scalar costs

Check that both costs are convertible to float:

```python
float(problem.l(x, u, t))
float(problem.g(x))
```

If a cost returns an array of shape `(1, 1)`, explicit conversion may fail or
produce confusing behavior.

---

### 32.4 Bounds

Check that bounds have correct shapes:

```python
u_min.shape == u_max.shape == (m,)
x_min.shape == x_max.shape == (n,)
```

Also check for accidentally reversed bounds, such as `u_min > u_max`.

---

### 32.5 Missing control dimension

If no control bounds are supplied, then

```python
problem.m is None
problem.control_dim == 0
```

Some parts of the adaptive solver need the control dimension to seed the PA
bundle. If bounds are omitted, make sure the experiment supplies an initial
bundle or another way to infer controls.

---

### 32.6 Local feasibility too strict

If `compute_H` or `PABundle.evaluate` returns no feasible control, inspect:

```python
problem.local_control_feasible(x, u, t, restricted=True, dt=dt)
```

and its subchecks:

```python
admissible_control
tangent_ok
step_feasible_control
```

The control may satisfy box bounds but fail tangent or step feasibility.

---

### 32.7 Newton feasibility problems

If Newton reports many feasibility rejections, inspect:

```python
trajectory_feasible
fraction_to_boundary_step
project_trajectory
```

The issue may be poor initial guesses, too-large Newton steps, active state
constraints, or projection logic that is inconsistent with the feasibility test.

---

### 32.8 Custom hooks

If a custom hook is supplied, verify its signature exactly matches what
`OCPProblem` expects.

Common mistakes include:

- returning only `u` instead of `(u, ok)` from `u_star_local_fn`;
- returning gradients in the wrong order from `hamiltonian_grad_fn`;
- forgetting to project oracle controls;
- returning non-boolean values from feasibility callbacks;
- returning a projected state with the wrong shape.

---

## 33) Summary

`OCPProblem` is the problem-specific interface used by the whole solver.

At minimum, it stores:

$$
\begin{equation}
f(x,u,t),
\qquad
\ell(x,u,t),
\qquad
g(x),
\qquad
x_0,
\qquad
T.
\end{equation}
$$

Optionally, it also stores:

- control and state bounds;
- oracle controls and Hamiltonian values;
- explicit Hamiltonian gradients;
- custom smoothed Hamiltonians;
- barrier terms;
- tangent and step-feasibility logic;
- state feasibility and projection logic;
- fraction-to-boundary logic for Newton damping.

The solver modules remain generic because all problem-specific information is
accessed through this one interface.