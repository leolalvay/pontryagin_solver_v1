# `core/newton.py` — Damped Newton solver for the shooting system $F(z)=0$

This module solves the nonlinear **shooting system**
$$
F(z)=0
$$
that encodes the discretized (smoothed) PMP TPBVP. The residual map $F$ is assembled by the integrator layer (`core/integrators.py`) and wrapped by the shooting layer (`core/shooting.py`).

Newton here is **not** a time integrator. It is a root-finding method on a large nonlinear system whose unknowns are the full discrete trajectories $(X_i,P_i)$ packed into a single vector $z$.

---

## 1) Unknown vector and dimensions

On a time grid $t_0,\dots,t_N$ with $x(t)\in\mathbb{R}^n$:

- $X_i\in\mathbb{R}^n$, $i=0,\dots,N$
- $P_i\in\mathbb{R}^n$, $i=0,\dots,N$
- $X_0=x_0$ is fixed data

The unknown vector is
$$
z=(X_1,\dots,X_N,\;P_0,\dots,P_N)\in\mathbb{R}^{(2N+1)n}.
$$

The residual vector $F(z)$ has the same dimension:
$$
F(z)\in\mathbb{R}^{(2N+1)n},
$$
because it stacks $2N$ step-residual blocks (each in $\mathbb{R}^n$) plus an $\mathbb{R}^n$ terminal boundary block.

---

## 2) Newton’s method in this repo (what is being computed)

Given an iterate $z^{(k)}$, Newton forms the linearized system
$$
J(z^{(k)})\,\Delta z^{(k)} = -F(z^{(k)}),
\qquad
J(z^{(k)}) \approx \frac{\partial F}{\partial z}(z^{(k)}),
$$
and updates
$$
z^{(k+1)} = z^{(k)} + \alpha^{(k)}\,\Delta z^{(k)}.
$$

The scalar $\alpha^{(k)}\in(0,1]$ is a **damping / line-search** parameter used to improve robustness when far from the solution.

---

## 3) Implementation flow (how the code is structured)

The core Newton loop follows this pattern:

1) **Evaluate residual**
   - call the shooting residual function to get $F(z^{(k)})$

2) **Stopping test**
   - check $\|F(z^{(k)})\|$ against a tolerance

3) **Assemble Jacobian**
   - call the Jacobian builder (finite differences in v1) to get $J(z^{(k)})$

4) **Solve linear system**
   - solve for $\Delta z^{(k)}$ using a dense solve
     $$J\Delta z=-F.$$

5) **Damping / acceptance**
   - try $\alpha=1$ first; if the residual does not decrease sufficiently, reduce $\alpha$ (e.g., multiply by a factor) until accepted

6) **Update iterate**
   - set $z^{(k+1)} = z^{(k)} + \alpha \Delta z^{(k)}$

---

## 4) Key code fragments (to connect math to implementation)

### Residual evaluation

The solver treats the residual as a black box:

```python
F = residual_fun(z)   # F(z)
normF = np.linalg.norm(F)
```

Here `residual_fun` ultimately calls `assemble_residual(...)` in the integrator layer, after unpacking $z$ into $(X,P)$.

### Jacobian evaluation

In the first version, the Jacobian is built numerically:

```python
J = jacobian_fun(z)   # approximate ∂F/∂z, usually finite differences
```

This is expensive but simple and consistent with the “prototype-first” philosophy of the repo.

### Newton step

```python
dz = np.linalg.solve(J, -F)   # solve J dz = -F
```

### Damped update (line search)

The update is applied with a damping factor `alpha`:

```python
z_new = z + alpha * dz
F_new = residual_fun(z_new)
```

A standard acceptance criterion is “residual decreases”:
$$
\|F(z+\alpha\Delta z)\| < \|F(z)\|.
$$

If not satisfied, the code reduces `alpha` (e.g., `alpha *= beta` with $0<\beta<1$) and retries.

---

## 5) What convergence means here

When Newton converges, we have
$$
\|F(z^\star)\|\approx 0,
$$
which implies:

- all discrete symplectic Euler step equations hold across the mesh, and
- the terminal boundary condition holds.

Unpacking $z^\star$ yields the discrete trajectories $(X^\star,P^\star)$ approximating $(x(t_i),p(t_i))$.

---

## 6) Practical numerical notes

- **Cost:** Finite-difference Jacobians scale poorly with the dimension of $z$ (roughly two residual evaluations per unknown). This is acceptable for small $(N,n)$ and for validation, but should be upgraded for large problems (analytic/AD Jacobians, sparse structure, etc.).
- **Damping matters:** The shooting system can be ill-conditioned, especially for small $\delta$ and/or coarse meshes. Damping helps avoid divergence.
- **Stopping criteria:** A robust implementation often uses both a residual norm tolerance and a step-size tolerance, e.g., $\|F\|\le \varepsilon_F$ and $\|\Delta z\|\le \varepsilon_z$.

---
