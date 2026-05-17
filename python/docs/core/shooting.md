# `core/shooting.py` — Newton-ready shooting wrappers

This module provides a thin interface between:

1. the discrete PMP residual and Jacobian assembly in `core/integrators.py`;
2. the damped Newton solver in `core/newton.py`.

The file does not implement the time discretization itself. Instead, it receives
a packed unknown vector `z`, reconstructs the state and costate arrays `(X,P)`,
and delegates the actual residual or Jacobian assembly to `integrators.py`.

The two public routines are:

```python
shooting_residual(...)
shooting_jacobian(...)
```

They define the nonlinear system

$$
\begin{equation}
F(z)=0
\end{equation}
$$

and its Jacobian

$$
\begin{equation}
J(z)
=
\frac{\partial F}{\partial z}.
\end{equation}
$$

Newton's method operates on this packed system.

---

## 1) Discrete unknowns and packing convention

Let the time mesh be

$$
\begin{equation}
0=t_0<t_1<\cdots<t_N=T.
\end{equation}
$$

Let the state and costate dimensions be $n$. The discrete trajectories are

$$
\begin{equation}
X_i\approx x(t_i),
\qquad
P_i\approx p(t_i),
\qquad
i=0,\dots,N.
\end{equation}
$$

The initial state is fixed by the problem:

$$
\begin{equation}
X_0=x_0.
\end{equation}
$$

Therefore, $X_0$ is not included in the Newton unknown vector. The unknowns are:

- the states $X_1,\dots,X_N$;
- all costates $P_0,\dots,P_N$.

The packed vector is

$$
\begin{equation}
z
=
(X_1,\dots,X_N,\;P_0,\dots,P_N)
\in
\mathbb{R}^{(2N+1)n}.
\end{equation}
$$

This convention is implemented by `pack_unknowns` and `unpack_unknowns` in
`core/integrators.py`.

Although `shooting.py` imports both helpers, the two wrapper functions mainly
use `unpack_unknowns`, because they receive an already-packed vector `z`.

---

## 2) Residual system represented by `F(z)`

The shooting residual is the vector of all discrete PMP equations plus the
terminal boundary condition.

For each interval $[t_i,t_{i+1}]$, define

$$
\begin{equation}
\Delta t_i
=
t_{i+1}-t_i.
\end{equation}
$$

The current integrator evaluates Hamiltonian gradients at the
symplectic-Euler point

$$
\begin{equation}
(P_{i+1},X_i,t_i).
\end{equation}
$$

The state residual block is

$$
\begin{equation}
r_x^i
=
X_{i+1}
-
X_i
-
\Delta t_i
H_p(P_{i+1},X_i,t_i),
\qquad
i=0,\dots,N-1.
\end{equation}
$$

The costate residual block is

$$
\begin{equation}
r_p^i
=
P_i
-
P_{i+1}
-
\Delta t_i
H_x(P_{i+1},X_i,t_i),
\qquad
i=0,\dots,N-1.
\end{equation}
$$

Here

$$
\begin{equation}
H_p
=
\nabla_p H,
\qquad
H_x
=
\nabla_x H.
\end{equation}
$$

Depending on the mode, these gradients come either from the smoothed PA
Hamiltonian $H_\delta$ or from explicit Hamiltonian-gradient callbacks supplied
by the problem.

The terminal cost is $g(X_N)$. In the current code, the terminal boundary
condition is

$$
\begin{equation}
P_N-\nabla g(X_N)=0.
\end{equation}
$$

Thus the terminal residual block is

$$
\begin{equation}
r_{\mathrm{bc}}
=
P_N-\nabla g(X_N).
\end{equation}
$$

The full residual vector is

$$
\begin{equation}
F(z)
=
\begin{bmatrix}
r_x^0\\
r_p^0\\
\vdots\\
r_x^{N-1}\\
r_p^{N-1}\\
r_{\mathrm{bc}}
\end{bmatrix}.
\end{equation}
$$

Each $r_x^i$ and $r_p^i$ lies in $\mathbb{R}^n$, and
$r_{\mathrm{bc}}\in\mathbb{R}^n$. Therefore,

$$
\begin{equation}
F(z)
\in
\mathbb{R}^{(2N+1)n}.
\end{equation}
$$

This matches the dimension of $z$, so the shooting system is square.

---

## 3) Current function signatures

The current residual wrapper is

```python
def shooting_residual(
    problem,
    t_nodes: np.ndarray,
    z: np.ndarray,
    bundle,
    delta: float,
    use_explicit_gradients: bool = False,
) -> np.ndarray:
    ...
```

The current Jacobian wrapper is

```python
def shooting_jacobian(
    problem,
    t_nodes: np.ndarray,
    z: np.ndarray,
    bundle,
    delta: float,
    use_explicit_gradients: bool = False,
) -> np.ndarray:
    ...
```

The shared arguments are:

| Argument | Meaning |
|---|---|
| `problem` | The `OCPProblem` instance. It supplies $x_0$, dynamics, costs, terminal cost, and optional Hamiltonian-gradient callbacks. |
| `t_nodes` | The current time mesh. |
| `z` | Packed Newton unknown vector containing $X_1,\dots,X_N,P_0,\dots,P_N$. |
| `bundle` | PA bundle used by the smoothed Hamiltonian, unless explicit-gradient mode is active. |
| `delta` | Smoothing parameter for the smoothed PA Hamiltonian. |
| `use_explicit_gradients` | If `True`, the integrator layer may use problem-provided Hamiltonian gradients instead of the smoothed PA gradients. |

The wrappers do not modify these inputs. They unpack `z`, call the integrator
layer, and return the requested residual or Jacobian.

---

## 4) `shooting_residual`

The residual wrapper is implemented as

```python
def shooting_residual(...):
    x0 = problem.x0
    X, P = unpack_unknowns(z, x0)
    return assemble_residual(
        problem,
        t_nodes,
        X,
        P,
        bundle,
        delta,
        use_explicit_gradients=use_explicit_gradients,
    )
```

The workflow is:

1. read the fixed initial condition from the problem;
2. unpack the Newton vector `z` into arrays `X` and `P`;
3. call `assemble_residual`;
4. return the residual vector.

Mathematically, this evaluates

$$
\begin{equation}
F(z)
=
F\bigl(
X(z),P(z);
t_{\mathrm{nodes}},U_{\mathrm{bundle}},\delta
\bigr).
\end{equation}
$$

The function `shooting_residual` does not know the details of each residual
block. Those details live in `integrators.py`.

---

## 5) `shooting_jacobian`

The Jacobian wrapper is implemented as

```python
def shooting_jacobian(...):
    x0 = problem.x0
    X, P = unpack_unknowns(z, x0)
    return assemble_jacobian(
        problem,
        t_nodes,
        X,
        P,
        bundle,
        delta,
        use_explicit_gradients=use_explicit_gradients,
    )
```

The workflow is:

1. read the fixed initial condition from the problem;
2. unpack the Newton vector `z` into arrays `X` and `P`;
3. call `assemble_jacobian`;
4. return the Jacobian matrix.

Mathematically, this evaluates

$$
\begin{equation}
J(z)
=
\frac{\partial F}{\partial z}(z).
\end{equation}
$$

The current Jacobian assembly is handled by `integrators.py`. It uses the local
block structure of the residual and assembles a sparse matrix. Some derivative
blocks are approximated by finite differences locally, but `shooting.py` itself
does not build those finite differences.

---

## 6) Explicit-gradient mode

Both wrappers accept

```python
use_explicit_gradients: bool = False
```

and pass it directly to the integrator layer.

When `use_explicit_gradients=True`, and when the problem provides the required
Hamiltonian-gradient callback, the residual assembly can use explicit
Hamiltonian gradients instead of gradients from the smoothed PA Hamiltonian.

Conceptually, this changes the source of

$$
\begin{equation}
H_p,
\qquad
H_x,
\end{equation}
$$

inside the residual blocks, but it does not change the packing convention or the
structure of the shooting system.

---

## 7) How `newton.py` uses these wrappers

The Newton solver constructs residual and Jacobian functions around the current
mesh, bundle, and smoothing parameter.

Conceptually:

```python
def residual(z):
    return shooting_residual(
        problem,
        t_nodes,
        z,
        bundle,
        delta,
        use_explicit_gradients=use_explicit_hamiltonian_gradients,
    )

def jacobian(z):
    return shooting_jacobian(
        problem,
        t_nodes,
        z,
        bundle,
        delta,
        use_explicit_gradients=use_explicit_hamiltonian_gradients,
    )
```

Newton then solves linearized systems of the form

$$
\begin{equation}
J(z^{(k)})
\Delta z^{(k)}
=
-F(z^{(k)}),
\end{equation}
$$

and updates

$$
\begin{equation}
z^{(k+1)}
=
z^{(k)}
+
\alpha
\Delta z^{(k)},
\end{equation}
$$

where $\alpha$ is chosen by the damping or line-search logic in `newton.py`.

Thus `shooting.py` supplies the Newton-ready map $F(z)$ and Jacobian $J(z)$,
while `newton.py` controls the nonlinear iteration.

---

## 8) What `shooting.py` does not do

This file is intentionally small. It does **not**:

1. construct the initial guess;
2. pack the initial arrays into `z` for the Newton solve;
3. implement the symplectic Euler residual blocks;
4. compute Hamiltonian gradients directly;
5. build finite-difference Jacobian blocks;
6. perform line search or damping;
7. update the adaptive mesh or PA bundle.

Those tasks are handled by other modules:

| Task | Module |
|---|---|
| Initial packing and Newton iteration | `newton.py` |
| Residual and Jacobian assembly | `integrators.py` |
| Smoothed Hamiltonian gradients | `smoothing.py` |
| Explicit Hamiltonian gradients | `problem.py` callbacks |
| Adaptive mesh/bundle/delta updates | `adaptivity.py` |

The purpose of `shooting.py` is only to translate between the packed Newton
vector and the array-based residual/Jacobian assembly routines.

---

## 9) Summary

The module `shooting.py` defines the Newton-facing nonlinear shooting system.

The packed unknown vector is

$$
\begin{equation}
z
=
(X_1,\dots,X_N,P_0,\dots,P_N).
\end{equation}
$$

The residual is

$$
\begin{equation}
F(z)
=
\begin{bmatrix}
r_x^0\\
r_p^0\\
\vdots\\
r_x^{N-1}\\
r_p^{N-1}\\
r_{\mathrm{bc}}
\end{bmatrix},
\end{equation}
$$

with terminal condition

$$
\begin{equation}
r_{\mathrm{bc}}
=
P_N-\nabla g(X_N).
\end{equation}
$$

The two wrappers simply perform

```python
X, P = unpack_unknowns(z, problem.x0)
```

and delegate to

```python
assemble_residual(...)
assemble_jacobian(...)
```

This keeps the Newton solver independent of the internal array layout of the
discrete state and costate trajectories.