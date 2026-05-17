# `core/newton.py` — Feasibility-aware damped Newton solver for the shooting system

This module solves the fixed-mesh discrete PMP two-point boundary value problem
(TPBVP). The problem is represented as a nonlinear shooting system

$$
\begin{equation}
F(z)=0,
\end{equation}
$$

where $z$ is the packed vector of state and costate unknowns.

The residual $F$ and Jacobian $J=\partial F/\partial z$ are provided by the
shooting layer, which wraps the residual and Jacobian assembly in
`core/integrators.py`.

Newton's method here is not a time integrator. It is a nonlinear root-finding
method applied to the whole discrete trajectory at once.

---

## 1) Unknown vector

On a mesh

$$
\begin{equation}
0=t_0<t_1<\cdots<t_N=T,
\end{equation}
$$

the discrete variables are

$$
\begin{equation}
X_i\approx x(t_i),
\qquad
P_i\approx p(t_i),
\qquad
i=0,\dots,N.
\end{equation}
$$

The initial state is fixed:

$$
\begin{equation}
X_0=x_0.
\end{equation}
$$

Therefore, the Newton unknown vector is

$$
\begin{equation}
z
=
(X_1,\dots,X_N,\;P_0,\dots,P_N).
\end{equation}
$$

The functions `pack_unknowns` and `unpack_unknowns` from `integrators.py`
convert between the array representation `(X,P)` and the packed vector `z`.

---

## 2) Main function signature

The main routine is

```python
def solve_tpbvp(
    problem,
    t_nodes: np.ndarray,
    bundle,
    delta: float,
    X_init: np.ndarray = None,
    P_init: np.ndarray = None,
    tol: float = 1e-10,
    max_iter: int = 50,
    use_explicit_hamiltonian_gradients: bool = False,
    fallback_solver: str | None = "least_squares",
) -> tuple:
    ...
```

It returns

```python
X_sol, P_sol, info
```

where `X_sol` and `P_sol` are the solved state and costate arrays, and `info`
is a dictionary of convergence diagnostics.

---

## 3) Inputs

| Argument | Meaning |
|---|---|
| `problem` | The `OCPProblem` instance. It provides $x_0$, dynamics, costs, constraints, feasibility checks, and optional projection logic. |
| `t_nodes` | Fixed time mesh used for this TPBVP solve. |
| `bundle` | PA bundle used by the smoothed Hamiltonian unless explicit-gradient mode is active. |
| `delta` | Smoothing parameter for the smoothed PA Hamiltonian. |
| `X_init` | Optional initial state trajectory guess, shape `(N+1,n)`. |
| `P_init` | Optional initial costate trajectory guess, shape `(N+1,n)`. |
| `tol` | Residual infinity-norm tolerance for convergence. |
| `max_iter` | Maximum number of damped Newton iterations before fallback or return. |
| `use_explicit_hamiltonian_gradients` | If `True`, the shooting residual/Jacobian may use problem-supplied Hamiltonian gradients. |
| `fallback_solver` | If set to `"least_squares"`, the code tries a nonlinear least-squares fallback when Newton stalls. |

---

## 4) Default initial guesses

The routine first determines

```python
N_plus_1 = t_nodes.size
n = problem.x0.size
```

If `X_init` is not provided, the code constructs a simple linear interpolation
from the initial state to zero:

```python
X_init = np.zeros((N_plus_1, n))
X_init[0] = problem.x0

for i in range(1, N_plus_1):
    alpha = i / (N_plus_1 - 1)
    X_init[i] = (1 - alpha) * problem.x0
```

Mathematically,

$$
\begin{equation}
X_i^{init}
=
(1-\alpha_i)x_0,
\qquad
\alpha_i
=
\frac{i}{N},
\qquad
i=0,\dots,N.
\end{equation}
$$

This means the default guess linearly moves from $x_0$ at $t_0$ to zero at
$t_N$.

If `P_init` is not provided, the costate guess is initialized to zero:

$$
\begin{equation}
P_i^{init}=0,
\qquad
i=0,\dots,N.
\end{equation}
$$

In code:

```python
P_init = np.zeros((N_plus_1, n))
```

The initial arrays are then packed into the Newton vector:

```python
z = pack_unknowns(X_init, P_init)
```

---

## 5) Residual and Jacobian closures

Inside `solve_tpbvp`, the code defines a residual closure:

```python
def residual(vec):
    return shooting_residual(
        problem,
        t_nodes,
        vec,
        bundle,
        delta,
        use_explicit_gradients=use_explicit_hamiltonian_gradients,
    )
```

Thus

$$
\begin{equation}
\texttt{residual}(z)=F(z).
\end{equation}
$$

It also defines a Jacobian closure:

```python
def jacobian(vec):
    J = shooting_jacobian(
        problem,
        t_nodes,
        vec,
        bundle,
        delta,
        use_explicit_gradients=use_explicit_hamiltonian_gradients,
    )
    return J.toarray() if issparse(J) else J
```

This closure is mainly used by the nonlinear least-squares fallback, which
expects a dense or array-like Jacobian.

During the main Newton loop, the code calls `shooting_jacobian` directly and
then tries sparse linear algebra first.

---

## 6) Newton iteration

The Newton loop is

```python
for it in range(max_iter):
    F = residual(z)
    normF = np.linalg.norm(F, ord=np.inf)

    if normF < tol:
        break

    J = shooting_jacobian(...)
    solve J dz = -F
    perform damped feasibility-aware update
```

The residual norm used for convergence is the infinity norm:

$$
\begin{equation}
\|F(z)\|_\infty
=
\max_j |F_j(z)|.
\end{equation}
$$

The convergence test is

$$
\begin{equation}
\|F(z)\|_\infty < \mathrm{tol}.
\end{equation}
$$

---

## 7) Newton linear system

At each iteration, Newton solves

$$
\begin{equation}
J(z^{(k)})\Delta z^{(k)}
=
-F(z^{(k)}).
\end{equation}
$$

The code calls the shooting Jacobian:

```python
J = shooting_jacobian(
    problem,
    t_nodes,
    z,
    bundle,
    delta,
    use_explicit_gradients=use_explicit_hamiltonian_gradients,
)
```

The current Jacobian assembly in `integrators.py` returns a sparse matrix, so
the Newton solver first tries sparse LU.

---

### 7.1 Sparse LU solve

The first linear solve attempt is:

```python
lu = splu(csc_matrix(J), permc_spec="COLAMD")
dz = lu.solve(-F)
```

This converts the matrix to CSC format and applies sparse LU factorization with
the `COLAMD` permutation strategy.

This is the preferred path for the current sparse block-stencil Jacobian.

---

### 7.2 Dense solve fallback

If sparse LU fails, the code converts the matrix to dense form if necessary:

```python
J_dense = J.toarray() if issparse(J) else J
```

Then it tries the dense linear solve:

```python
dz = np.linalg.solve(J_dense, -F)
```

This is useful if sparse factorization fails but the dense matrix is nonsingular.

---

### 7.3 Linear least-squares fallback

If the dense solve also fails with a singular matrix error, the code solves the
linearized system in the least-squares sense:

```python
dz, *_ = np.linalg.lstsq(J_dense, -F, rcond=None)
```

This produces a step even when the Jacobian is singular or nearly singular.

Thus the current linear solve hierarchy is:

```python
sparse LU
    -> dense solve
        -> dense least-squares solve
```

---

## 8) Solver phase flags

At the start of the solve, the code sets

```python
solver_phase = "newton"
fallback_used = False
```

These values are later stored in the returned `info` dictionary.

If the nonlinear least-squares fallback is used after the Newton loop, then

```python
solver_phase = "least_squares_fallback"
fallback_used = True
```

The linear least-squares fallback inside a Newton step does not change
`solver_phase`; it is only a way to compute the Newton direction for that
iteration.

---

## 9) Trial trajectory helper

The helper

```python
def _trial_trajectory(problem, t_nodes, z_trial, feasibility_tol):
    X_trial, P_trial = unpack_unknowns(z_trial, problem.x0)
    feasible = problem.trajectory_feasible(X_trial, t_nodes, tol=feasibility_tol)
    return X_trial, P_trial, feasible
```

unpacks a trial Newton vector and checks whether the resulting state trajectory
is feasible.

Mathematically, for a trial vector

$$
\begin{equation}
z_{\mathrm{trial}}
=
z+\lambda\Delta z,
\end{equation}
$$

the helper reconstructs

$$
\begin{equation}
(X_{\mathrm{trial}},P_{\mathrm{trial}})
=
\mathrm{unpack}(z_{\mathrm{trial}})
\end{equation}
$$

and checks whether

$$
\begin{equation}
X_{\mathrm{trial}}
\in
K
\end{equation}
$$

at the mesh nodes, according to the problem's feasibility logic.

This helper is used inside the line search.

---

## 10) Fraction-to-boundary step limiter

After computing the Newton direction `dz`, the code extracts the state part of
the current iterate and the state part of the Newton direction:

```python
X_curr, _ = unpack_unknowns(z, problem.x0)
dX, _ = unpack_unknowns(dz, np.zeros_like(problem.x0))
```

The call

```python
problem.fraction_to_boundary_step(
    X_curr,
    dX,
    t_nodes,
    safety=0.99,
    tol=feasibility_tol,
)
```

computes a maximum safe fraction of the Newton step before crossing a state
constraint boundary.

The initial damping parameter is

```python
lam = min(1.0, fraction_to_boundary)
lam = max(lam, 1e-8)
```

Thus the trial update starts with

$$
\begin{equation}
0<\lambda\le 1.
\end{equation}
$$

This is a feasibility-oriented step limiter. If no active state constraints are
present, the returned fraction is typically $1$, and the method starts with a
full Newton step.

---

## 11) Feasibility-aware Armijo line search

The Newton update is not accepted immediately. The code performs a damped line
search.

For a candidate damping parameter $\lambda$, the trial vector is

$$
\begin{equation}
z_{\mathrm{trial}}
=
z+\lambda\Delta z.
\end{equation}
$$

In code:

```python
z_trial = z + lam * dz
```

The code then checks feasibility:

```python
X_trial, _, feasible = _trial_trajectory(
    problem,
    t_nodes,
    z_trial,
    feasibility_tol,
)
```

If the trajectory is infeasible, the trial is rejected and the damping parameter
is halved:

```python
n_feasibility_rejections += 1
lam *= 0.5
continue
```

If the trajectory is feasible, the residual is evaluated:

```python
F_new = residual(z_trial)
normF_new = np.linalg.norm(F_new, ord=np.inf)
```

The acceptance criterion is Armijo-like:

```python
if normF_new <= (1 - 1e-4 * lam) * normF:
    z = z_trial
    accepted = True
    break
```

Mathematically, the trial is accepted if

$$
\begin{equation}
\|F(z+\lambda\Delta z)\|_\infty
\le
(1-10^{-4}\lambda)\,
\|F(z)\|_\infty.
\end{equation}
$$

If the condition fails, the code halves $\lambda$:

```python
lam *= 0.5
```

The line search stops when either a trial is accepted or

```python
lam <= 1e-8
```

---

## 12) Projection fallback after failed line search

If the feasibility-aware line search fails, the code has one additional
projection fallback, but only when the problem provides state projection logic:

```python
if not accepted and last_trial is not None and problem.project_state_fn is not None:
    ...
```

The code takes the last trial trajectory, projects its state trajectory, and
checks feasibility:

```python
X_trial, P_trial, _ = _trial_trajectory(...)
X_proj = problem.project_trajectory(X_trial, t_nodes, tol=feasibility_tol)
```

If the projected trajectory is feasible, it repacks the projected state with the
trial costate:

```python
z_proj = pack_unknowns(X_proj, P_trial)
```

Then it accepts the projection only if it improves the residual:

```python
F_proj = residual(z_proj)
normF_proj = np.linalg.norm(F_proj, ord=np.inf)

if normF_proj < normF:
    z = z_proj
    accepted = True
    n_projection_fallbacks += 1
```

So the projection fallback accepts

$$
\begin{equation}
z_{\mathrm{proj}}
=
\mathrm{pack}(X_{\mathrm{proj}},P_{\mathrm{trial}})
\end{equation}
$$

only if

$$
\begin{equation}
\|F(z_{\mathrm{proj}})\|_\infty
<
\|F(z)\|_\infty.
\end{equation}
$$

If this fallback also fails, the Newton loop terminates early:

```python
if not accepted:
    break
```

---

## 13) End of the Newton loop

After the Newton loop exits, the code computes the final residual:

```python
final_residual = residual(z)
final_norm = np.linalg.norm(final_residual, ord=np.inf)
```

If

$$
\begin{equation}
\|F(z)\|_\infty < \mathrm{tol},
\end{equation}
$$

then the Newton phase succeeded.

If the residual is still too large, the code may try a nonlinear least-squares
fallback.

---

## 14) Nonlinear least-squares fallback

If

```python
final_norm >= tol
```

and

```python
fallback_solver == "least_squares"
```

the code calls SciPy's trust-region reflective least-squares solver:

```python
lsq = least_squares(
    residual,
    z,
    jac=jacobian,
    method="trf",
    xtol=tol,
    ftol=tol,
    gtol=tol,
    max_nfev=max(200, 10 * (max_iter + 1)),
)
```

This fallback minimizes the nonlinear residual in the least-squares sense:

$$
\begin{equation}
\min_z
\frac{1}{2}
\|F(z)\|_2^2.
\end{equation}
$$

The fallback uses the same residual function and the Jacobian closure defined
earlier.

The maximum number of function evaluations is

$$
\begin{equation}
\max(200,\;10(\mathrm{max\_iter}+1)).
\end{equation}
$$

After the least-squares solve, the code sets

```python
z = lsq.x
```

and unpacks the result.

---

### 14.1 Projection after least-squares fallback

After least squares, the code checks whether the resulting trajectory is
feasible:

```python
X_lsq, P_lsq = unpack_unknowns(z, problem.x0)
```

If it is infeasible and the problem provides projection logic, the code tries

```python
X_proj = problem.project_trajectory(X_lsq, t_nodes, tol=feasibility_tol)
```

If the projected trajectory is feasible, it repacks:

```python
z = pack_unknowns(X_proj, P_lsq)
```

This increments the projection fallback counter.

Finally, the residual is recomputed:

```python
final_residual = residual(z)
final_norm = np.linalg.norm(final_residual, ord=np.inf)
```

and the solver phase is updated:

```python
solver_phase = "least_squares_fallback"
fallback_used = True
```

---

## 15) Returned solution and diagnostics

The final packed vector is unpacked:

```python
X_sol, P_sol = unpack_unknowns(z, problem.x0)
```

The routine returns

```python
X_sol, P_sol, info
```

where `info` is

```python
{
    "iterations": it + 1,
    "residual_norm": final_norm,
    "solver_phase": solver_phase,
    "fallback_used": fallback_used,
    "feasibility_rejections": int(n_feasibility_rejections),
    "projection_fallbacks": int(n_projection_fallbacks),
}
```

The fields mean:

| Field | Meaning |
|---|---|
| `iterations` | Number of Newton-loop iterations executed. |
| `residual_norm` | Final infinity norm $\|F(z)\|_\infty$. |
| `solver_phase` | `"newton"` or `"least_squares_fallback"`. |
| `fallback_used` | Whether the nonlinear least-squares fallback was used. |
| `feasibility_rejections` | Number of line-search trial steps rejected because the trial trajectory was infeasible. |
| `projection_fallbacks` | Number of times a projected trajectory was accepted. |

---

## 16) Overall algorithm summary

The complete workflow is:

```python
build default X_init and P_init if needed
z = pack_unknowns(X_init, P_init)

define residual(z)
define jacobian(z)

solver_phase = "newton"
fallback_used = False

for it in range(max_iter):

    F = residual(z)
    normF = ||F||_inf

    if normF < tol:
        break

    J = shooting_jacobian(...)

    try sparse LU solve for dz
    else try dense solve
    else try dense linear least-squares solve

    compute fraction-to-boundary step limit
    lam = initial damping parameter

    while lam > 1e-8:
        z_trial = z + lam * dz

        if trajectory infeasible:
            lam *= 0.5
            continue

        if Armijo residual decrease holds:
            accept z_trial
            break

        lam *= 0.5

    if line search failed:
        try projected-state fallback

    if still not accepted:
        break

final_norm = ||residual(z)||_inf

if final_norm >= tol and fallback_solver == "least_squares":
    run nonlinear least_squares fallback
    optionally project final trajectory

return unpacked X, P, info
```

---

## 17) Practical debugging notes

### 17.1 Residual does not decrease

If the residual does not decrease during Newton, inspect:

- the Jacobian accuracy from `integrators.py`;
- the smoothing parameter $\delta$;
- the initial guess;
- whether trial steps are being rejected for infeasibility;
- whether the line search is reducing `lam` to very small values.

---

### 17.2 Many feasibility rejections

A large value of `info["feasibility_rejections"]` means many trial trajectories
violated state feasibility.

Possible causes include:

- coarse mesh;
- poor initial guess;
- overly aggressive Newton direction;
- active state constraints;
- insufficient fraction-to-boundary limiting;
- feasibility checks that are too strict.

---

### 17.3 Projection fallback is not a substitute for convergence

The projection fallback can recover a feasible state trajectory after a failed
trial step, but it is accepted only if it reduces the residual. Frequent
projection fallback use may indicate that the Newton direction is fighting the
state constraints.

---

### 17.4 Least-squares fallback interpretation

If `info["solver_phase"] == "least_squares_fallback"`, the returned solution was
not obtained solely by the damped Newton loop. It was improved by minimizing

$$
\begin{equation}
\frac{1}{2}\|F(z)\|_2^2.
\end{equation}
$$

This can be very useful for robustness, but the final residual should still be
checked against the desired tolerance.

---

### 17.5 Explicit-gradient mode

If `use_explicit_hamiltonian_gradients=True`, the shooting residual and
Jacobian use explicit Hamiltonian gradients when the problem supplies them.

This can improve accuracy and robustness for examples with known analytic
Hamiltonian derivatives. However, the Newton workflow itself is unchanged.

---

## 18) Summary

`newton.py` solves the fixed-mesh TPBVP by applying a feasibility-aware damped
Newton method to

$$
\begin{equation}
F(z)=0.
\end{equation}
$$

At each Newton iteration it:

1. evaluates the residual;
2. assembles the Jacobian;
3. solves a linear Newton system;
4. limits the step by state feasibility;
5. performs an Armijo-like residual decrease line search;
6. optionally projects failed trial states;
7. optionally falls back to nonlinear least squares.

The output is the solved state and costate trajectories together with diagnostic
information describing how the solve proceeded.