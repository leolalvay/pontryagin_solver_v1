# `core/shooting.py` — Nonlinear shooting system $F(z)=0$ for the discrete smoothed PMP

This module is the **bridge** between:

- the **discretized PMP dynamics** (implemented as residual blocks in `core/integrators.py`), and
- the **Newton solver** (implemented in `core/newton.py`).

Concretely, `shooting.py` defines the nonlinear map
$$
F:\mathbb{R}^m \to \mathbb{R}^m,
\qquad
F(z)=0,
$$
where $z$ is the vector of unknown discrete states/costates. Newton never “integrates forward” here: it simply tries to drive $F(z)$ to zero.

---

## 1) What are the unknowns $z$?

On a mesh $0=t_0<\dots<t_N=T$ with $x(t)\in\mathbb{R}^n$, we store node values:

- $X_i \approx x(t_i)\in\mathbb{R}^n$, for $i=0,\dots,N$,
- $P_i \approx p(t_i)\in\mathbb{R}^n$, for $i=0,\dots,N$.

The initial state is fixed: $X_0=x_0$ is **data**, not an unknown.

Therefore the unknown vector is packed as
$$
z = (X_1,\dots,X_N,\;P_0,\dots,P_N)\in\mathbb{R}^{(2N+1)n}.
$$

In code this packing/unpacking is delegated to `core/integrators.py`:

- `pack_unknowns(X,P) -> z`  (flattens and concatenates),
- `unpack_unknowns(z,x0) -> (X,P)` (reshapes and reinserts $X_0=x_0$).

---

## 2) What is the shooting residual map $F(z)$?

Given a candidate trajectory $(X,P)$ (hence a candidate $z$), the residual map $F(z)$ is constructed by enforcing:

1) the symplectic Euler step equations at each interval $[t_i,t_{i+1}]$,
2) the terminal boundary condition $P_N + \nabla g(X_N)=0$.

All of these are assembled into one vector:

$$
F(z)
=
\begin{bmatrix}
r_x^{(0)}\\
r_p^{(0)}\\
\vdots\\
r_x^{(N-1)}\\
r_p^{(N-1)}\\
r_{\mathrm{bc}}
\end{bmatrix}
\in\mathbb{R}^{(2N+1)n}.
$$

This assembly is implemented in `core/integrators.assemble_residual(...)`.

### Dimension check

- For each time step $i$ we have:
  - $r_x^{(i)}\in\mathbb{R}^n$,
  - $r_p^{(i)}\in\mathbb{R}^n$,
  hence $2Nn$ equations total.
- Terminal boundary condition adds $r_{\mathrm{bc}}\in\mathbb{R}^n$.

So
$$
\dim(F)=2Nn+n=(2N+1)n=\dim(z),
$$
i.e. the shooting system is **square**.

---

## 3) What does `shooting.py` actually provide?

At a practical level, this file exposes utilities to evaluate:

- the residual vector $F(z)$,
- (optionally) its Jacobian matrix $J(z)\approx \partial F/\partial z$,

by calling the integrator layer.

The conceptual structure is:

```python
# pseudo-structure (not exact code)
def F_of_z(z):
    X, P = unpack_unknowns(z, x0)
    return assemble_residual(problem, t_nodes, X, P, bundle, delta)

def J_of_z(z):
    X, P = unpack_unknowns(z, x0)
    return assemble_jacobian(problem, t_nodes, X, P, bundle, delta)
```

So `shooting.py` does not contain the numerical discretization itself; it *wraps* it into a Newton-ready interface.

---

## 4) How Newton uses this module

Newton requires:

- a function returning the current residual $F(z)$,
- a function returning (or approximating) the Jacobian $J(z)$,
- an initial guess $z^{(0)}$.

This module supplies the first two, while the initialization and iteration logic lives in `core/newton.py`.

A typical Newton step is:
$$
J(z^{(k)})\,\Delta z^{(k)} = -F(z^{(k)}),
\qquad
z^{(k+1)} = z^{(k)} + \alpha\,\Delta z^{(k)}.
$$

Because $F(z)$ is constructed as “all step residuals + terminal BC”, driving $F(z)\to 0$ means:

- every discrete symplectic Euler update is satisfied, and
- the terminal transversality condition is satisfied.

That is exactly the discrete TPBVP solution.

---

## 5) Notes on implementation choices

- The repo uses **finite differences** to build Jacobians (via the integrator layer). This is expensive for large $(N,n)$ but is acceptable for a first version and small test cases.
- The smoothing parameter $\delta$ enters only through Hamiltonian evaluations inside `assemble_residual`, so from Newton’s perspective it is just another parameter passed through the residual function.
- The PA bundle influences $H_\delta$ and its gradients; `shooting.py` treats the bundle as a fixed object passed into residual/Jacobian evaluation.

---
