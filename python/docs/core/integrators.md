# `core/integrators.py` — Symplectic Euler residual + finite-difference Jacobian

This module turns the **δ-smoothed Pontryagin system** into a **nonlinear algebraic system**
$$
F(z)=0
$$
that can be solved by Newton (via the shooting layer). The key is that we do **not** “integrate forward” as a standalone simulator; instead, we **assemble residual blocks** that enforce the symplectic Euler updates and the terminal boundary condition.

---

## 1) Continuous system being discretized (what we want to enforce)

Using the smoothed Hamiltonian $H_\delta(p,x,t)$, the canonical PMP dynamics are
$$
\dot x(t)=\nabla_p H_\delta(p(t),x(t),t),
\qquad
-\dot p(t)=\nabla_x H_\delta(p(t),x(t),t),
$$
with boundary conditions
$$
x(0)=x_0,
\qquad
p(T)+\nabla g(x(T))=0.
$$

In the implementation, $\nabla_p H_\delta$ and $\nabla_x H_\delta$ are obtained from
`eval_H_smooth(...)`, and $\nabla g$ is approximated by finite differences.

---

## 2) Unknown vector $z$ (how the solver stores $(X,P)$)

On a mesh $0=t_0<\dots<t_N=T$, we store samples
$$
X_i\approx x(t_i),\qquad P_i\approx p(t_i).
$$

The initial state $X_0=x_0$ is **fixed**, so the unknown vector concatenates
$$
z = (X_1,\dots,X_N,\;P_0,\dots,P_N).
$$

This is exactly what `pack_unknowns` does:

```python
def pack_unknowns(X, P):
    N_plus_1, n = X.shape
    N = N_plus_1 - 1
    z = np.zeros((N * n + (N + 1) * n,)) #z=(X_1,..,P_N)
    z[0:N * n] = X[1:, :].reshape(N * n)      # x_1,...,x_N
    z[N * n:]  = P.reshape((N + 1) * n)       # p_0,...,p_N
    return z
```

And `unpack_unknowns` reconstructs $(X,P)$ from $z$ while reinserting $X_0=x_0$:

```python
def unpack_unknowns(z, x0):
    n = x0.size
    total = z.size // n
    N = (total - 1) // 2
    X = np.zeros((N + 1, n))
    P = np.zeros((N + 1, n))
    X[0, :]   = x0
    X[1:, :]  = z[0:N * n].reshape((N, n))
    P[:, :]   = z[N * n:].reshape((N + 1, n))
    return X, P
```

---

## 3) Symplectic Euler discretization (residual blocks)

For each step $i=0,\dots,N-1$ with $\Delta t_i=t_{i+1}-t_i$, symplectic Euler is enforced as:

**State update (gradient at the start):**
$$
X_{i+1} = X_i + \Delta t_i\,\nabla_p H_\delta(P_i,X_i,t_i).
$$

**Costate update (gradient at the end):**
$$
P_i = P_{i+1} + \Delta t_i\,\nabla_x H_\delta(P_{i+1},X_{i+1},t_{i+1}).
$$

The code enforces these by assembling residuals
$$
r_x^{(i)} = X_i + \Delta t_i\,\nabla_p H_\delta(P_i,X_i,t_i) - X_{i+1},
$$
$$
r_p^{(i)} = P_{i+1} + \Delta t_i\,\nabla_x H_\delta(P_{i+1},X_{i+1},t_{i+1}) - P_i.
$$

Here is the exact implementation pattern inside `assemble_residual`:
```python
def assemble_residual(problem, t_nodes: np.ndarray, X: np.ndarray, P: np.ndarray, bundle, delta: float) -> np.ndarray:
N_plus_1 = t_nodes.size        # number of time nodes = N+1
N = N_plus_1 - 1               # number of steps
n = X.shape[1]                 # dimension of the state (and the costate)
residual = np.zeros((2 * N * n + n,))
offset =0
```

There are $N$ steps.

At each step, there are **2 vector equations** of dimension $n$:

- One for the **state**  $r_{x_i} \in \mathbb{R}^n$
- One for the **costate** $r_{p_i} \in \mathbb{R}^n$

That gives  $2 N n$ components.

At the end, there is a **terminal boundary condition** $r_{bc} \in \mathbb{R}^n $.

**Total:**  
$$
2 N n + n
$$

`offset` is a cursor that indicates the position in the residual vector where we are writting in.
```python
for i in range(N):
        dt = t_nodes[i + 1] - t_nodes[i]
        x_i = X[i]
        x_ip1 = X[i + 1]
        p_i = P[i]
        p_ip1 = P[i + 1]
        # gradient at start
        _, grad_p_i, _ = eval_H_smooth(problem, bundle, p_i, x_i, t_nodes[i], delta)
        # gradient at end (for costate update)
        _, _, grad_x_ip1 = eval_H_smooth(problem, bundle, p_ip1, x_ip1, t_nodes[i + 1], delta)
        # state residual r_x = x_i + dt * grad_p - x_{i+1}
        r_x = x_i + dt * grad_p_i - x_ip1
        residual[offset:offset + n] = r_x
        offset += n
        # costate residual r_p = p_{i+1} + dt * grad_x_ip1 - p_i
        r_p = p_ip1 + dt * grad_x_ip1 - p_i
        residual[offset:offset + n] = r_p
        offset += n
```

For each time step iteration `i`, the residual $r_x^{i}$ and $r_p^{i}$ are stored in this order. This means that the assembly of the residual is 
$$
F=(r_x^{0},r_p^{0},r_x^{1},r_p^1,\cdots,r_x^{N-1},r_p^{N-1},.)
$$
```python
# terminal boundary condition: p_N + ∇g(x_N) = 0
    x_N = X[-1]
    p_N = P[-1]
    # gradient of g by finite difference
    g_grad = np.zeros_like(p_N)
    eps = 1e-6
    for j in range(n):
        x_plus = x_N.copy()
        x_minus = x_N.copy()
        x_plus[j] += eps
        x_minus[j] -= eps
        g_plus = problem.g(x_plus)
        g_minus = problem.g(x_minus)
        g_grad[j] = (g_plus - g_minus) / (2 * eps)
    r_bc = p_N + g_grad
    residual[offset:] = r_bc
```
The terminal boundary condition is added to obtain
$$
F=(r_x^{0},r_p^{0},r_x^{1},r_p^1,\cdots,r_x^{N-1},r_p^{N-1},r_{bc})
$$
---

## 4) Terminal boundary condition $p(T)+\nabla g(x(T))=0$

At the final node $(X_N,P_N)$, the code appends the boundary residual
$$
r_{\mathrm{bc}} = P_N + \nabla g(X_N).
$$

Since the problem object exposes $g(x)$, the gradient is computed by central finite differences:

```python
x_N = X[-1]
p_N = P[-1]
g_grad = np.zeros_like(p_N)
eps = 1e-6
for j in range(n):
    x_plus  = x_N.copy(); x_plus[j]  += eps
    x_minus = x_N.copy(); x_minus[j] -= eps
    g_plus  = problem.g(x_plus)
    g_minus = problem.g(x_minus)
    g_grad[j] = (g_plus - g_minus) / (2 * eps)

r_bc = p_N + g_grad
```

---
# `assemble_jacobian` (core / `integrators.py`)

This note documents **both the mathematics and the implementation** of:

```python
assemble_jacobian(problem, t_nodes, X, P, bundle, delta)
```

It is written so that you can debug future issues by checking (i) the discretization, (ii) the index maps, and (iii) the block-wise Jacobian contributions.

---

## 1) Purpose

`assemble_jacobian(...)` builds the Jacobian matrix

$$
J = \frac{\partial F}{\partial z},
$$

where $F(z)=0$ is the TPBVP residual produced by the symplectic-Euler shooting discretization, and $z$ is the vector of unknowns (state + costate) in the **repo ordering**.

The key implementation idea is **locality**: each residual block depends only on a few neighboring unknown blocks (a block stencil). Therefore we compute only those Jacobian blocks (linear blocks analytically + nonlinear blocks by local central differences).

---

## 2) Definitions, shapes, and repo conventions

### 2.1 Time grid and trajectories

Let:

- $N = \texttt{t\_nodes.size} - 1$ (number of time steps),
- $n = \texttt{X.shape[1]}$ (dimension of state/costate).

Inputs:

- `t_nodes`: shape $(N+1,)$ with nodes $t_0<\dots<t_N$,
- `X`: shape $(N+1,n)$ with states $x_0,\dots,x_N$,
- `P`: shape $(N+1,n)$ with costates $p_0,\dots,p_N$.

### 2.2 Unknown vector $z$ (repo order)

The repo packs unknowns as:

$$
z = (x_1,\dots,x_N,\; p_0,\dots,p_N)\in\mathbb{R}^{(2N+1)n}.
$$

Important: $x_0$ is fixed (given by the OCP) and **is not part of $z$**.

Hence:

$$
m = (2N+1)n,\qquad J\in\mathbb{R}^{m\times m}.
$$

### 2.3 Residual vector $F$ (row order)

The residual is ordered as:

$$
F = (r_x^0, r_p^0, r_x^1, r_p^1, \dots, r_x^{N-1}, r_p^{N-1}, r_{bc}),
$$

with each block $r_x^i,r_p^i,r_{bc}\in\mathbb{R}^n$. Total length is $2Nn+n = (2N+1)n = m$.

---

## 3) Discretization being linearized

For each time step $i=0,\dots,N-1$ define:

$$
\Delta t_i = t_{i+1}-t_i.
$$

The smoothed Hamiltonian is evaluated through:

$$
(\_, \nabla_p H_\delta, \nabla_x H_\delta)
=
\texttt{eval\_H\_smooth}(\texttt{problem},\texttt{bundle},p_{i+1},x_i,t_i,\delta).
$$

### 3.1 Residual blocks

State residual:

$$
r_x^i = x_{i+1} - x_i - \Delta t_i\,\nabla_p H_\delta(p_{i+1},x_i,t_i).
$$

Costate residual:

$$
r_p^i = p_i - p_{i+1} - \Delta t_i\,\nabla_x H_\delta(p_{i+1},x_i,t_i).
$$

Terminal boundary condition:

$$
r_{bc} = p_N + \nabla g(x_N).
$$

In the code, $\nabla g(x_N)$ is computed by central finite differences (same convention as in `assemble_residual`).

---

## 4) Locality / sparsity pattern (block stencil)

From the formulas:

- $r_x^i$ depends on $x_{i+1}, x_i, p_{i+1}$,
- $r_p^i$ depends on $p_i, p_{i+1}, x_i$,
- $r_{bc}$ depends on $x_N, p_N$.

So each residual block row touches only a few unknown blocks. The Jacobian is therefore **block-banded** (block size $n\times n$).

Special case $i=0$: the residual depends on $x_0$, but $x_0$ is not in $z$, so there is **no Jacobian column for $x_0$**. This is why the code only adds $\partial/\partial x_i$ contributions when $i\ge 1$.

---

## 5) Decomposition into linear vs nonlinear parts

The code introduces (conceptually):

$$
\phi(i) = -\Delta t_i\,\nabla_p H_\delta(p_{i+1},x_i,t_i),
\qquad
\psi(i) = -\Delta t_i\,\nabla_x H_\delta(p_{i+1},x_i,t_i),
$$

so:

$$
r_x^i = (x_{i+1}-x_i) + \phi(i),
\qquad
r_p^i = (p_i-p_{i+1}) + \psi(i).
$$

### 5.1 Exact linear Jacobian blocks (inserted analytically)

From the explicit linear terms:

- $\frac{\partial r_x^i}{\partial x_{i+1}} = I_n$,
- $\frac{\partial r_x^i}{\partial x_i} = -I_n$ (only if $i\ge 1$),
- $\frac{\partial r_p^i}{\partial p_i} = I_n$,
- $\frac{\partial r_p^i}{\partial p_{i+1}} = -I_n$,
- $\frac{\partial r_{bc}}{\partial p_N} = I_n$.

These blocks do not require finite differences and are inserted directly.

### 5.2 Nonlinear Jacobian blocks (approximated by local central differences)

The remaining blocks come from derivatives of $\phi(i)$ and $\psi(i)$:

- $\frac{\partial \phi(i)}{\partial x_i},\ \frac{\partial \psi(i)}{\partial x_i}$ ($i\ge 1$),
- $\frac{\partial \phi(i)}{\partial p_{i+1}},\ \frac{\partial \psi(i)}{\partial p_{i+1}}$.

These correspond to second-derivative objects of the smoothed Hamiltonian (mixed Hessians). The implementation approximates these columns using **central differences** with:

- `eps = 1e-7` (Jacobian FD step).

For example, for a coordinate $\ell$:

$$
\frac{\partial \phi(i)}{\partial x_i^{(\ell)}} \approx
\frac{\phi(i; x_i^{(\ell)}+\texttt{eps})-\phi(i; x_i^{(\ell)}-\texttt{eps})}{2\,\texttt{eps}},
$$

and similarly for $\psi(i)$ and perturbations of $p_{i+1}$.

---

## 6) Row/column index maps used in the code

The implementation uses helper maps consistent with the repo ordering.

### 6.1 Column starts in $z$

- For $x_k$ (only valid for $k=1,\dots,N$):

$$
\texttt{col\_x}(k) = (k-1)n.
$$

- For $p_j$ (valid for $j=0,\dots,N$):

$$
\texttt{col\_p}(j) = Nn + jn.
$$

### 6.2 Row starts in $F$

- For $r_x^i$:

$$
\texttt{row\_rx}(i) = (2i)n.
$$

- For $r_p^i$:

$$
\texttt{row\_rp}(i) = (2i+1)n.
$$

- For the terminal block:

$$
\texttt{row\_bc} = (2N)n.
$$

These indices are used to place each $n\times n$ block into the correct rows/columns.

---

## 7) Implementation walkthrough (block-by-block)

This section is meant to map directly onto the code structure.

### 7.1 Sparse assembly strategy (COO triplets → CSR)

Even though $J$ is sparse, repeatedly writing into CSR/CSC with slicing is fragile/slow.
Therefore the function collects entries as triplets:

- `rows.append(i)`, `cols.append(j)`, `data.append(value)`

and then builds:

```python
J = coo_matrix((data, (rows, cols)), shape=(m, m)).tocsr()
J.sum_duplicates()
```

This is robust: multiple contributions to the same entry (e.g., linear + nonlinear) are safely accumulated.

### 7.2 Adding identity blocks

The helper:

```python
add_I(rr0, cc0, sign)
```

adds `sign * I_n` at block position $(rr0:rr0+n,\ cc0:cc0+n)$ by inserting $n$ diagonal entries.

This is used for the exact linear blocks listed in Section 5.1.

### 7.3 Time-step loop $i=0,\dots,N-1$

For each $i$:

#### (A) Linear blocks for $r_x^i$
- Always add $+I_n$ at columns of $x_{i+1}$.
- Add $-I_n$ at columns of $x_i$ only if $i\ge 1$ (since $x_0$ is not in $z$).

#### (B) Linear blocks for $r_p^i$
- Always add $+I_n$ at columns of $p_i$.
- Always add $-I_n$ at columns of $p_{i+1}$.

#### (C) Nonlinear columns via finite differences

**(C1) Perturbations of $x_i$ (only if $i\ge 1$)**

For each coordinate $\ell=0,\dots,n-1$:

1. Temporarily perturb `X[i, ell]` to $+\texttt{eps}$ and compute `phi(i), psi(i)`.
2. Perturb to $-\texttt{eps}$ and compute again.
3. Restore the original value.

This produces column vectors:

- $d\phi \in \mathbb{R}^n$, added into rows of $r_x^i$ at the scalar column corresponding to $x_i^{(\ell)}$,
- $d\psi \in \mathbb{R}^n$, added into rows of $r_p^i$ at the same column.

**(C2) Perturbations of $p_{i+1}$ (always)**

Similarly, for each coordinate $\ell$, perturb `P[i+1, ell]` and compute:

- $\partial\phi/\partial p_{i+1}^{(\ell)}$ contributes to $r_x^i$ rows,
- $\partial\psi/\partial p_{i+1}^{(\ell)}$ contributes to $r_p^i$ rows.

Note: for $r_p^i$ the column of $p_{i+1}$ already has a linear contribution $-I_n$; the FD contribution is the *extra nonlinear part*.

### 7.4 Boundary condition blocks

#### (A) $\partial r_{bc}/\partial p_N$
Add $+I_n$ at the block for $p_N$.

#### (B) $\partial r_{bc}/\partial x_N$ by finite differences
For each coordinate $\ell$:

1. Perturb `X[N, ell]` to $+\texttt{eps}$ and evaluate:

$$
\texttt{bc\_block()} = p_N + \nabla g(x_N).
$$

2. Perturb to $-\texttt{eps}$, evaluate again, then restore.

The resulting column is an approximation to:

$$
\frac{\partial r_{bc}}{\partial x_N^{(\ell)}} \approx \frac{\texttt{bc\_block}(x_N^{(\ell)}+\texttt{eps})-\texttt{bc\_block}(x_N^{(\ell)}-\texttt{eps})}{2\,\texttt{eps}}.
$$

**Numerical note (nested FD):** `bc_block()` itself computes $\nabla g$ by finite differences (with `epsg = 1e-6`), so this is effectively a finite-difference approximation of a Hessian-like quantity. This is consistent with how the residual is defined in the repo.

---

## 8) Numerical / debugging notes

1) **In-place perturbations.**
The code perturbs entries of `X` and `P` in place and restores them. This is correct in single-thread usage, but it is a red flag if parallelizing Jacobian assembly in the future.

2) **Finite difference steps.**
- Jacobian FD step: `eps = 1e-7`.
- Terminal gradient FD step (inside `bc_block`): `epsg = 1e-6`.

If Newton becomes noisy or stagnates, step-size tuning is one of the first knobs to try.

3) **Consistency requirements.**
This Jacobian is only correct if all conventions match the residual:
- evaluation at $(p_{i+1}, x_i, t_i)$,
- residual ordering $F=(r_x^0,r_p^0,\dots,r_{bc})$,
- unknown ordering $z=(x_1,\dots,x_N,p_0,\dots,p_N)$.

If any of these conventions change, you must update both the formulas and the index maps.

---

## 9) Minimal correctness checklist (fast sanity tests)

When debugging Jacobian issues, verify:

1) Shape:
$$
J.\texttt{shape} = ((2N+1)n,\ (2N+1)n).
$$

2) Exact identity blocks appear:
- $\partial r_x^i/\partial x_{i+1}=I_n$,
- $\partial r_p^i/\partial p_i=I_n$,
- $\partial r_{bc}/\partial p_N=I_n$.

3) Special case $i=0$:
No column exists for $x_0$. Therefore no writes should target $x_0$.

4) Signs:
- $r_x^i = x_{i+1}-x_i-\Delta t_i\nabla_pH_\delta$,
- $r_p^i = p_i-p_{i+1}-\Delta t_i\nabla_xH_\delta$.

Any sign change in the residual must be mirrored in the Jacobian.

---

## 10) Complexity (order-of-magnitude)

Let $C_H$ be the cost of one call to `eval_H_smooth`.

Per step $i$:
- For $p_{i+1}$ derivatives: $2n$ evaluations (plus/minus per coordinate) for both $\phi$ and $\psi$.
- For $x_i$ derivatives (if $i\ge 1$): another $2n$ evaluations.

So the Jacobian assembly cost scales roughly like:

$$
O(Nn\,C_H),
$$

which is much cheaper than global FD on the entire residual (which would scale like $O(m\,C_F)$ with $m=(2N+1)n$ and $C_F\sim O(NC_H)$).

---
