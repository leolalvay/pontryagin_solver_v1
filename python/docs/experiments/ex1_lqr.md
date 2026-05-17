# `experiments/ex1_lqr.py` — Example 1: Linear Quadratic Regulator (LQR)

This experiment is a **minimal validation case** for the solver on a smooth, well-understood optimal control problem: a 2D linear system with a quadratic running cost and quadratic terminal cost (LQR).

The script does three things:

1) defines the LQR problem in the `OCPProblem` interface,
2) calls the adaptive Pontryagin solver `solve_optimal_control(...)`,
3) post-processes the discrete solution $(X,P)$ to (approximately) reconstruct the optimal control $u^*(t_i)$ and estimate the objective value.

---

## 1) Problem definition (math)

### State, control, horizon

- State: $$x(t)\in\mathbb{R}^2$$
- Control: $$u(t)\in\mathbb{R}^1$$
- Horizon: $$t\in[0,T],\quad T=1$$
- Initial condition: $$x(0)=x_0=\begin{bmatrix}1\\0\end{bmatrix}$$

### Dynamics

The dynamics are linear:
$$
\dot x(t) = A x(t) + B u(t),
$$
with
$$
A=\begin{bmatrix}0&1\\0&0\end{bmatrix},
\qquad
B=\begin{bmatrix}0\\1\end{bmatrix}.
$$

In code:
```python
def dynamics(x, u, t):
    return A.dot(x) + B.dot(u)
```

### Costs

Running cost (quadratic):
$$
\ell(x,u,t)=x^\top Q x + u^\top R u,
\qquad
Q=I_2,\ \ R=10^{-2}I_1.
$$

Terminal cost:
$$
g(x)=x^\top Q_f x,
\qquad
Q_f = Q.
$$

In code:
```python
def stage_cost(x, u, t):
    return float(x.dot(Q.dot(x)) + u.T.dot(R).dot(u))

def terminal_cost(x):
    return float(x.dot(Qf.dot(x)))
```

### Control bounds (practical detail)

The LQR is conceptually *unconstrained*, but the implementation requires bounds for the control search. The script uses wide bounds:
$$
u_{\min}=-5,\qquad u_{\max}=5,
$$
which acts as an “approximately unconstrained” LQR.

In code:
```python
u_min = np.array([-5.0])
u_max = np.array([5.0])
```

---

## 2) Discretization used by the experiment

The initial time mesh is uniform with 20 segments:
- $N=20$ intervals,
- $N+1=21$ nodes.

In code:
```python
t_nodes = np.linspace(0.0, T, 21)
```

The adaptive outer loop may refine this mesh further depending on the indicators, but the experiment starts from this baseline mesh.

---

## 3) Calling the solver (implementation flow)

The `OCPProblem` is constructed with the dynamics/cost functions and bounds:

```python
prob = OCPProblem(dynamics, stage_cost, terminal_cost, x0, T,
                  control_bounds=(u_min, u_max), state_bounds=None)
```

Then the adaptive solver is called:

```python
result = solve_optimal_control(
    prob, t_nodes,
    tol_time=1e-3, tol_PA=1e-3, tol_delta=1e-3,
    max_iters=5, delta0=0.1
)
```

Interpretation of parameters:
- `delta0=0.1` initializes the smoothing level for $H_\delta$.
- `tol_time`, `tol_PA`, `tol_delta` control the outer-loop stopping tests.
- `max_iters=5` bounds how many outer adaptations (mesh/bundle/delta updates) are attempted.

The returned `result` dictionary is expected to contain at least:
- `result['t_nodes']`: final mesh,
- `result['X']`: state trajectory array of shape $$(N+1,2)$$,
- `result['P']`: costate trajectory array of shape $$(N+1,2)$$,
- `result['bundle']`: final PA bundle,
- `result['delta']`: final smoothing parameter,
- `result['log']`: per-outer-iteration diagnostics.

---

## 4) Post-processing: reconstructing a discrete control $u^*(t_i)$

The solver internally works with the Hamiltonian minimization through the bundle/smoothing machinery. For plotting/validation, the experiment reconstructs a control at each node by calling:

```python
_, u_star = compute_H(prob, P[i], X[i], mesh[i], bundle.controls, restricted=True)
```

Conceptually, at each node $t_i$ it computes:
$$
u^*(t_i) \in \arg\min_{u\in\mathcal{U}_{\text{candidates}}}
\Big( P_i \cdot f(X_i,u,t_i) + \ell(X_i,u,t_i)\Big),
$$
where `restricted=True` applies the repo's “restricted” selection logic and $\mathcal{U}_{\text{candidates}}$ is a finite candidate set (bundle planes + corners of the control box).

The output `controls` has shape $$(N+1,1)$$ (one control per node).

---

## 5) Post-processing: objective value estimate (sum approximation)

The true objective is:
$$
J(u)=\int_0^T \ell(x(t),u(t),t)\,dt + g(x(T)).
$$

The experiment approximates it on the discrete mesh using a left Riemann sum:
$$
J \approx g(X_N) + \sum_{i=0}^{N-1}\ell(X_i,u_i,t_i)\,\Delta t_i,
\qquad \Delta t_i = t_{i+1}-t_i.
$$

In code:
```python
obj = prob.g(X[-1])
for i in range(len(mesh) - 1):
    dt = mesh[i + 1] - mesh[i]
    u_i = controls[i]
    obj += prob.l(X[i], u_i, mesh[i]) * dt
```

This is a **diagnostic estimate** used for sanity checks (not a certified error bound).

---

## 6) Diagnostics printed by the experiment

The script prints:
- mesh size (`len(mesh)`),
- number of bundle planes (`bundle.num_planes()`),
- the estimated objective value,
- the full outer-loop indicator history (`result['log']`), which should include the time / PA / delta indicators and Newton diagnostics per outer iteration.

---

## Notes for validation

- This LQR problem has a known closed-form solution via the Riccati equation. While the experiment does not compare against the Riccati solution, it is a good candidate for future regression tests (e.g., comparing the cost and trajectories to a reference).
- Because the Hamiltonian minimization is implemented via a **finite candidate set** (bundle + box corners), the reconstructed control is an approximation to the true continuous minimizer unless the candidate set is sufficiently rich.

---
