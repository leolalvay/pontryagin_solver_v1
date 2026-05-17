# `core/constraints.py` — Box-constraint utilities (projection by clipping)

This module contains two small utilities that implement **box constraints** by **Euclidean projection**, i.e., *componentwise clipping*.

They are intentionally minimal: rather than implementing a full state-constrained PMP (with multipliers / complementarity), these functions provide a practical way to keep iterates inside simple bound sets.

---

## 1) Mathematical objects: control set $A$ and state set $K$

In the optimal control model we typically enforce:

- **Control constraints**: $u(t)\in A$
- **State constraints**: $x(t)\in K$

In this repo, the most common choice is that both sets are **axis-aligned boxes**:

$$
A = [u_{\min},u_{\max}]
=
\{u\in\mathbb{R}^m:\; u_{\min}\le u\le u_{\max}\},
$$

$$
K = [x_{\min},x_{\max}]
=
\{x\in\mathbb{R}^n:\; x_{\min}\le x\le x_{\max}\},
$$

where all inequalities are componentwise.

---

## 2) Projection onto a box: why “clipping” is the right formula

Given any closed convex set $C$, the (Euclidean) projection of a point $z$ onto $C$ is

$$
\Pi_C(z)
=
\arg\min_{y\in C}\; \|y-z\|_2^2.
$$

When $C$ is a **box** $[l,u]$, this projection has a closed form:

- In 1D:
$$
\Pi_{[l,u]}(z) = \min(\max(z,l),u).
$$

- In $d$ dimensions (componentwise):
$$
\Pi_{[l,u]}(z)_k = \min\big(\max(z_k,l_k),u_k\big),\qquad k=1,\dots,d.
$$

So “clipping” is not a heuristic here: for box constraints it is exactly the Euclidean projector.

---

## 3) What each function means

### `enforce_state_bounds(x, x_min, x_max)`

This returns the projected state
$$
x_{\mathrm{proj}} = \Pi_{[x_{\min},x_{\max}]}(x),
$$
implemented as the vectorized operation
$$
x_{\mathrm{proj}} = \min(\max(x,x_{\min}),x_{\max}),
$$
where `min`/`max` are understood componentwise.

**Interpretation:** if a numerical update produces a state slightly outside $K$ (e.g., due to time discretization), this function maps it back into the feasible box.

### `project_control(u, u_min, u_max)`

This returns the projected control
$$
u_{\mathrm{proj}} = \Pi_{[u_{\min},u_{\max}]}(u)
= \min(\max(u,u_{\min}),u_{\max}).
$$

**Interpretation:** when an internal computation proposes a control outside the admissible bounds, this enforces $u\in A$ by projection.

---

## 4) Implementation details (what the code literally does)

Both functions use the NumPy identity

- `np.maximum(z, l)` applies $\max(\cdot,\cdot)$ componentwise,
- `np.minimum(·, u)` applies $\min(\cdot,\cdot)$ componentwise,

so

$$
\Pi_{[l,u]}(z) = \min(\max(z,l),u)
$$

is computed without loops and is fast for arrays.

**Shape expectations:** `x`, `x_min`, `x_max` (and similarly for `u`) should be the same shape or broadcastable by NumPy.

---

## 5) Numerical/optimality note (important)

Projecting $x$ back into $K$ is a **feasibility repair** step. It is not, by itself, the full mathematical treatment of state constraints in PMP.

A principled PMP treatment typically enforces viability via tangent cone conditions (e.g., restricting admissible velocities so that $f(x,u,t)\in T_K(x)$ on the boundary), or introduces constraint multipliers. This repo contains the machinery for restricted Hamiltonians elsewhere; the utilities here are simply the “box projection” building block.

In short:
- For **control bounds**, projection is standard and usually harmless.
- For **state bounds**, projection is useful for guarding numerics, but it can alter the continuous-time dynamics if it is used as the only constraint mechanism.

---
## Where `core/constraints.py` is used in the repo

This module provides **box-projection (clipping)** helpers, but in the current codebase the same clipping logic is primarily accessed through methods on `OCPProblem`.

### Control bounds ($u \in A$)

- **`core/problem.py` → `OCPProblem.project_control(u)`**  
  Implements the Euclidean projection onto the control box $A=[u_{\min},u_{\max}]$ via
  $$
  \Pi_A(u)=\min(\max(u,u_{\min}),u_{\max}).
  $$
  This is mathematically identical to `core/constraints.project_control(u, u_min, u_max)` (the module function is currently redundant).

- **`core/hamiltonian.py` → `compute_H(..., candidate_controls, restricted=False)`**  
  When iterating over `candidate_controls` (e.g., controls coming from a `PABundle`), the code calls
  `problem.project_control(u)` to ensure each candidate lies in $A$ before evaluating the Hamiltonian plane value
  $$
  g(p,x,t;u)=p^\top f(x,u,t)+\ell(x,u,t).
  $$

- **`core/problem.py` → `OCPProblem.admissible_control(u)`**  
  Provides a boolean check $u\in A$ and is used inside the Hamiltonian candidate loop as a final filter.

### State bounds ($x \in K$)

- **`core/problem.py` → `OCPProblem.get_state_bounds()` / `OCPProblem.admissible_state(x)`**  
  Expose the box $K=[x_{\min},x_{\max}]$ and allow checking feasibility $x\in K$.

- **`core/problem.py` → `OCPProblem.tangent_cone_filter(...)` and `OCPProblem.tangent_ok(x,u,t)`**  
  These implement a **viability / tangent-cone restriction** for box state constraints: on an active boundary, candidate velocities that point outward are rejected.  
  This is used by **`core/hamiltonian.py` → `compute_H(..., restricted=True)`**, which computes the restricted Hamiltonian $H_K$ by minimizing only over controls whose velocity satisfies
  $$
  f(x,u,t)\in T_K(x).
  $$

### Note on `enforce_state_bounds`

- **`core/constraints.enforce_state_bounds(x, x_min, x_max)` is currently not called elsewhere**.  
  If desired, it can be used as a *numerical safety guard* after a time-step update (e.g., in an integrator) to clip states back into $K$, but the current implementation enforces state constraints mainly via the tangent-cone viability filter rather than post-step projection.
