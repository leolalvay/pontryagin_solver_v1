# `core/integrators.py` — Discrete PMP residual and sparse Jacobian assembly

This module converts the discrete smoothed Pontryagin system into the nonlinear
algebraic system

$$
\begin{equation}
F(z)=0
\end{equation}
$$

used by the shooting and Newton layers.

The module provides:

```python
pack_unknowns(X, P)
unpack_unknowns(z, x0)
assemble_residual(problem, t_nodes, X, P, bundle, delta, use_explicit_gradients=False)
assemble_jacobian(problem, t_nodes, X, P, bundle, delta, use_explicit_gradients=False)
```

The key point is that this file does not run a forward simulation. Instead, it
assembles residual blocks that enforce the discrete PMP equations and the
terminal boundary condition.

---

## 1) Continuous PMP system and sign convention

The solver uses a minimum Hamiltonian convention. For a Hamiltonian $H$, the PMP
system is written in the form

$$
\begin{equation}
\dot x(t)
=
\nabla_p H(p(t),x(t),t),
\end{equation}
$$

and

$$
\begin{equation}
-\dot p(t)
=
\nabla_x H(p(t),x(t),t).
\end{equation}
$$

Equivalently,

$$
\begin{equation}
\dot p(t)
=
-\nabla_x H(p(t),x(t),t).
\end{equation}
$$

The initial state is fixed:

$$
\begin{equation}
x(0)=x_0.
\end{equation}
$$

The current implementation uses the terminal condition

$$
\begin{equation}
p(T)-\nabla g(x(T))=0,
\end{equation}
$$

or equivalently

$$
\begin{equation}
p(T)=\nabla g(x(T)).
\end{equation}
$$

This sign convention is important because the code assembles the terminal
residual as

```python
r_bc = p_N - g_grad
```

where `g_grad` is a finite-difference approximation of $\nabla g(X_N)$.

---

## 2) Discrete mesh and trajectory arrays

Let the mesh be

$$
\begin{equation}
0=t_0<t_1<\cdots<t_N=T.
\end{equation}
$$

The discrete state and costate arrays are

$$
\begin{equation}
X_i\approx x(t_i),
\qquad
P_i\approx p(t_i),
\qquad
i=0,\dots,N.
\end{equation}
$$

In code,

```python
X.shape == (N + 1, n)
P.shape == (N + 1, n)
```

where $n$ is the state dimension.

The initial state is fixed:

$$
\begin{equation}
X_0=x_0.
\end{equation}
$$

Therefore, $X_0$ is not included in the Newton unknown vector.

---

## 3) Unknown vector packing

The unknown vector is ordered as

$$
\begin{equation}
z
=
(X_1,\dots,X_N,\;P_0,\dots,P_N)
\in
\mathbb{R}^{(2N+1)n}.
\end{equation}
$$

The states $X_1,\dots,X_N$ contribute $Nn$ unknowns, and the costates
$P_0,\dots,P_N$ contribute $(N+1)n$ unknowns. Thus,

$$
\begin{equation}
\dim z
=
Nn+(N+1)n
=
(2N+1)n.
\end{equation}
$$

---

### 3.1 `pack_unknowns`

The function

```python
def pack_unknowns(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    ...
```

creates `z` from the full trajectory arrays.

The implementation is:

```python
N_plus_1, n = X.shape
N = N_plus_1 - 1

z = np.zeros((N * n + (N + 1) * n,))

z[0:N * n] = X[1:, :].reshape(N * n)
z[N * n:] = P.reshape((N + 1) * n)
```

Thus the first block of `z` stores

$$
\begin{equation}
X_1,\dots,X_N,
\end{equation}
$$

and the second block stores

$$
\begin{equation}
P_0,\dots,P_N.
\end{equation}
$$

---

### 3.2 `unpack_unknowns`

The function

```python
def unpack_unknowns(z: np.ndarray, x0: np.ndarray):
    ...
```

reconstructs `(X, P)` from `z` and reinserts the fixed initial state $x_0$.

The code infers $N$ from the length of `z`:

```python
n = x0.size
total = z.size // n
N = (total - 1) // 2
```

Because

$$
\begin{equation}
\frac{\mathrm{len}(z)}{n}
=
2N+1.
\end{equation}
$$

Then it sets

```python
X[0, :] = x0
X[1:, :] = z[0:N * n].reshape((N, n))
P[:, :] = z[N * n:].reshape((N + 1, n))
```

So `unpack_unknowns` is the inverse of `pack_unknowns` under the repository's
unknown ordering.

---

## 4) Hamiltonian-gradient source

The helper

```python
def _hamiltonian_gradients(
    problem,
    bundle,
    p,
    x,
    t,
    delta,
    dt=None,
    use_explicit_gradients=False,
):
    ...
```

selects where the Hamiltonian gradients come from.

If

```python
use_explicit_gradients=True
```

and the problem provides `problem.hamiltonian_grad_fn`, the helper returns

```python
None, *problem.hamiltonian_gradients(x, p, t)
```

That is, it uses problem-provided formulas for

$$
\begin{equation}
H_p,
\qquad
H_x.
\end{equation}
$$

Otherwise, it calls

```python
eval_H_smooth(problem, bundle, p, x, t, delta, dt=dt)
```

and obtains gradients from the smoothed PA Hamiltonian $H_\delta$.

Thus, the residual assembly has two modes:

1. **smoothed PA mode**, using `eval_H_smooth`;
2. **explicit-gradient mode**, using `problem.hamiltonian_gradients`.

The residual structure is the same in both modes. Only the source of
$H_p$ and $H_x$ changes.

---

## 5) `assemble_residual`

The current signature is

```python
def assemble_residual(
    problem,
    t_nodes: np.ndarray,
    X: np.ndarray,
    P: np.ndarray,
    bundle,
    delta: float,
    use_explicit_gradients: bool = False,
) -> np.ndarray:
    ...
```

It returns a vector of length

$$
\begin{equation}
2Nn+n
=
(2N+1)n.
\end{equation}
$$

The residual ordering is

$$
\begin{equation}
F
=
(r_x^0,r_p^0,r_x^1,r_p^1,\dots,r_x^{N-1},r_p^{N-1},r_{\mathrm{bc}}).
\end{equation}
$$

Each block lies in $\mathbb{R}^n$.

---

## 6) Current discrete residual blocks

For each time step $i=0,\dots,N-1$, define

$$
\begin{equation}
\Delta t_i
=
t_{i+1}-t_i.
\end{equation}
$$

The current code evaluates both Hamiltonian gradients at the mixed
symplectic-Euler point

$$
\begin{equation}
(P_{i+1},X_i,t_i).
\end{equation}
$$

In code:

```python
_, grad_p, grad_x = _hamiltonian_gradients(
    problem,
    bundle,
    P[i + 1],
    X[i],
    t_nodes[i],
    delta,
    dt=dt,
    use_explicit_gradients=use_explicit_gradients,
)
```

The state residual is

$$
\begin{equation}
r_x^i
=
X_{i+1}
-
X_i
-
\Delta t_i
H_p(P_{i+1},X_i,t_i).
\end{equation}
$$

In code:

```python
r_x = x_ip1 - x_i - dt * grad_p
```

The costate residual is

$$
\begin{equation}
r_p^i
=
P_i
-
P_{i+1}
-
\Delta t_i
H_x(P_{i+1},X_i,t_i).
\end{equation}
$$

In code:

```python
r_p = p_i - p_ip1 - dt * grad_x
```

The residual blocks are appended in order:

```python
residual[offset:offset + n] = r_x
offset += n

residual[offset:offset + n] = r_p
offset += n
```

Therefore, driving these residuals to zero enforces

$$
\begin{equation}
X_{i+1}
=
X_i
+
\Delta t_i
H_p(P_{i+1},X_i,t_i),
\end{equation}
$$

and

$$
\begin{equation}
P_i
=
P_{i+1}
+
\Delta t_i
H_x(P_{i+1},X_i,t_i).
\end{equation}
$$

The second equation is consistent with $-\dot p=H_x$ under a backward-style
costate update.

---

## 7) Terminal boundary residual

After all step residuals are assembled, the code computes the terminal gradient
of the terminal cost $g$ by central finite differences.

It sets

```python
x_N = X[-1]
p_N = P[-1]
g_grad = np.zeros_like(p_N)
eps = 1e-6
```

For each state coordinate $j$, it forms

$$
\begin{equation}
x_N^+
=
x_N+\varepsilon e_j,
\qquad
x_N^-
=
x_N-\varepsilon e_j,
\end{equation}
$$

with

$$
\begin{equation}
\varepsilon=10^{-6}.
\end{equation}
$$

Then

$$
\begin{equation}
(\nabla g(x_N))_j
\approx
\frac{
g(x_N+\varepsilon e_j)
-
g(x_N-\varepsilon e_j)
}{
2\varepsilon
}.
\end{equation}
$$

The terminal residual is

$$
\begin{equation}
r_{\mathrm{bc}}
=
P_N-\nabla g(X_N).
\end{equation}
$$

In code:

```python
r_bc = p_N - g_grad
residual[offset:] = r_bc
```

This completes the residual vector

$$
\begin{equation}
F(z)
\in
\mathbb{R}^{(2N+1)n}.
\end{equation}
$$

---

## 8) Residual code workflow summary

The residual assembly can be summarized as:

```python
N = len(t_nodes) - 1
n = X.shape[1]
residual = np.zeros((2 * N * n + n,))
offset = 0

for i in range(N):
    dt = t_nodes[i + 1] - t_nodes[i]

    x_i = X[i]
    x_ip1 = X[i + 1]
    p_i = P[i]
    p_ip1 = P[i + 1]

    _, grad_p, grad_x = _hamiltonian_gradients(
        problem,
        bundle,
        p_ip1,
        x_i,
        t_nodes[i],
        delta,
        dt=dt,
        use_explicit_gradients=use_explicit_gradients,
    )

    r_x = x_ip1 - x_i - dt * grad_p
    r_p = p_i - p_ip1 - dt * grad_x

    write r_x into residual
    write r_p into residual

compute g_grad at X[-1] by central differences
r_bc = P[-1] - g_grad
write r_bc into residual

return residual
```

The key conventions to remember are:

1. unknown order:
   $z=(X_1,\dots,X_N,P_0,\dots,P_N)$;
2. residual order:
   $F=(r_x^0,r_p^0,\dots,r_x^{N-1},r_p^{N-1},r_{\mathrm{bc}})$;
3. Hamiltonian-gradient evaluation point:
   $(P_{i+1},X_i,t_i)$;
4. terminal sign:
   $r_{\mathrm{bc}}=P_N-\nabla g(X_N)$.

   ---

# `assemble_jacobian`

The function

```python
def assemble_jacobian(
    problem,
    t_nodes: np.ndarray,
    X: np.ndarray,
    P: np.ndarray,
    bundle,
    delta: float,
    use_explicit_gradients: bool = False,
):
    ...
```

assembles the Jacobian matrix

$$
\begin{equation}
J
=
\frac{\partial F}{\partial z}
\end{equation}
$$

for the residual defined by `assemble_residual`.

The current implementation exploits the local dependence structure of the
residual. It does not build the Jacobian by perturbing every component of the
global vector $z$ and recomputing the entire residual. Instead, it assembles the
nonzero local blocks directly.

The returned matrix is a sparse CSR matrix.

---

## 9) Dimensions and ordering

Let

$$
\begin{equation}
N
=
\texttt{t\_nodes.size}-1,
\qquad
n
=
\texttt{X.shape[1]}.
\end{equation}
$$

The unknown vector has length

$$
\begin{equation}
m
=
(2N+1)n.
\end{equation}
$$

In code:

```python
m = (2 * N + 1) * n
```

Thus

$$
\begin{equation}
J\in\mathbb{R}^{m\times m}.
\end{equation}
$$

The unknown ordering is

$$
\begin{equation}
z
=
(X_1,\dots,X_N,\;P_0,\dots,P_N).
\end{equation}
$$

The residual ordering is

$$
\begin{equation}
F
=
(r_x^0,r_p^0,r_x^1,r_p^1,\dots,r_x^{N-1},r_p^{N-1},r_{\mathrm{bc}}).
\end{equation}
$$

---

## 10) Finite-difference step sizes

The current Jacobian routine uses two local finite-difference step sizes:

```python
eps_x = 1e-7
eps_p = 1e-7
```

They are used for perturbing state and costate variables, respectively.

Inside the terminal boundary helper `bc_block`, the terminal gradient
$\nabla g(X_N)$ is itself computed by central differences with

```python
epsg = 1e-6
```

Thus the terminal $x_N$ Jacobian block is a finite difference of a
finite-difference gradient. It approximates a Hessian-like terminal-cost block.

---

## 11) Row and column maps

The code defines local index maps that convert mathematical block indices into
matrix row and column offsets.

---

### 11.1 State columns

The state unknowns are $X_1,\dots,X_N$. Therefore, for $k=1,\dots,N$,

$$
\begin{equation}
\texttt{col\_x}(k)
=
(k-1)n.
\end{equation}
$$

There is no column for $X_0$, because the initial state is fixed data.

---

### 11.2 Costate columns

The costate unknowns are $P_0,\dots,P_N$. They begin after the state block.
Thus, for $j=0,\dots,N$,

$$
\begin{equation}
\texttt{col\_p}(j)
=
Nn+jn.
\end{equation}
$$

---

### 11.3 Residual rows

The state residual block $r_x^i$ starts at

$$
\begin{equation}
\texttt{row\_rx}(i)
=
(2i)n.
\end{equation}
$$

The costate residual block $r_p^i$ starts at

$$
\begin{equation}
\texttt{row\_rp}(i)
=
(2i+1)n.
\end{equation}
$$

The terminal boundary block starts at

$$
\begin{equation}
\texttt{row\_bc}
=
(2N)n.
\end{equation}
$$

These maps ensure that each local $n$-vector block is inserted into the correct
part of the global Jacobian.

---

## 12) Local nonlinear parts: `phi` and `psi`

The code separates each residual block into a linear part and a nonlinear
Hamiltonian-gradient part.

For the state residual,

$$
\begin{equation}
r_x^i
=
(X_{i+1}-X_i)
+
\phi(i),
\end{equation}
$$

where

$$
\begin{equation}
\phi(i)
=
-\Delta t_i
H_p(P_{i+1},X_i,t_i).
\end{equation}
$$

For the costate residual,

$$
\begin{equation}
r_p^i
=
(P_i-P_{i+1})
+
\psi(i),
\end{equation}
$$

where

$$
\begin{equation}
\psi(i)
=
-\Delta t_i
H_x(P_{i+1},X_i,t_i).
\end{equation}
$$

In code:

```python
def phi(i):
    dt = t_nodes[i + 1] - t_nodes[i]
    _, grad_p, _ = _hamiltonian_gradients(
        problem,
        bundle,
        P[i + 1],
        X[i],
        t_nodes[i],
        delta,
        dt=dt,
        use_explicit_gradients=use_explicit_gradients,
    )
    return -dt * grad_p
```

and

```python
def psi(i):
    dt = t_nodes[i + 1] - t_nodes[i]
    _, _, grad_x = _hamiltonian_gradients(
        problem,
        bundle,
        P[i + 1],
        X[i],
        t_nodes[i],
        delta,
        dt=dt,
        use_explicit_gradients=use_explicit_gradients,
    )
    return -dt * grad_x
```

Thus the Jacobian is assembled from:

1. exact derivatives of the explicit linear terms;
2. finite-difference derivatives of `phi(i)` and `psi(i)`.

---

## 13) Terminal boundary helper: `bc_block`

The helper

```python
def bc_block() -> np.ndarray:
    ...
```

returns the terminal residual block

$$
\begin{equation}
r_{\mathrm{bc}}
=
P_N-\nabla g(X_N).
\end{equation}
$$

Inside `bc_block`, the terminal gradient is approximated by central differences:

$$
\begin{equation}
(\nabla g(X_N))_j
\approx
\frac{
g(X_N+\varepsilon_g e_j)
-
g(X_N-\varepsilon_g e_j)
}{
2\varepsilon_g
},
\qquad
\varepsilon_g=10^{-6}.
\end{equation}
$$

Then

$$
\begin{equation}
\texttt{bc\_block()}
=
P_N-\nabla g(X_N).
\end{equation}
$$

This sign must match the residual assembly in `assemble_residual`.

---

## 14) Sparse assembly strategy

The Jacobian is assembled using COO triplets:

```python
rows = []
cols = []
data = []
```

Each scalar matrix entry is appended as:

```python
rows.append(row_index)
cols.append(col_index)
data.append(value)
```

At the end, the matrix is built and converted to CSR:

```python
J = coo_matrix((data, (rows, cols)), shape=(m, m)).tocsr()
J.sum_duplicates()
return J
```

This strategy is robust because different terms may contribute to the same
matrix entry. The call `sum_duplicates()` combines those contributions.

---

## 15) Exact linear blocks

The helper

```python
def add_I(rr0: int, cc0: int, sign: float):
    ...
```

inserts the block

$$
\begin{equation}
\mathrm{sign}\,I_n
\end{equation}
$$

at the row offset `rr0` and column offset `cc0`.

---

### 15.1 Linear blocks in `r_x^i`

Recall

$$
\begin{equation}
r_x^i
=
X_{i+1}-X_i+\phi(i).
\end{equation}
$$

Therefore,

$$
\begin{equation}
\frac{\partial r_x^i}{\partial X_{i+1}}
=
I_n.
\end{equation}
$$

The code always inserts this block:

```python
cx_ip1 = col_x(i + 1)
add_I(rr_x, cx_ip1, +1.0)
```

For $i\ge 1$, $X_i$ is also an unknown, and

$$
\begin{equation}
\frac{\partial r_x^i}{\partial X_i}
=
-I_n
+
\frac{\partial \phi(i)}{\partial X_i}.
\end{equation}
$$

The exact linear part is inserted as

```python
if i >= 1:
    cx_i = col_x(i)
    add_I(rr_x, cx_i, -1.0)
```

For $i=0$, $X_0$ is fixed and has no column in $z$, so no $X_0$ block is added.

---

### 15.2 Linear blocks in `r_p^i`

Recall

$$
\begin{equation}
r_p^i
=
P_i-P_{i+1}+\psi(i).
\end{equation}
$$

Therefore,

$$
\begin{equation}
\frac{\partial r_p^i}{\partial P_i}
=
I_n,
\end{equation}
$$

and

$$
\begin{equation}
\frac{\partial r_p^i}{\partial P_{i+1}}
=
-I_n
+
\frac{\partial \psi(i)}{\partial P_{i+1}}.
\end{equation}
$$

The exact linear parts are inserted as

```python
cp_i = col_p(i)
add_I(rr_p, cp_i, +1.0)

cp_ip1 = col_p(i + 1)
add_I(rr_p, cp_ip1, -1.0)
```

---

### 15.3 Linear block in terminal condition

For

$$
\begin{equation}
r_{\mathrm{bc}}
=
P_N-\nabla g(X_N),
\end{equation}
$$

the derivative with respect to $P_N$ is

$$
\begin{equation}
\frac{\partial r_{\mathrm{bc}}}{\partial P_N}
=
I_n.
\end{equation}
$$

The code inserts

```python
cp_N = col_p(N)
add_I(row_bc, cp_N, +1.0)
```

---

## 16) Finite-difference nonlinear blocks

The remaining Jacobian blocks are derivatives of `phi`, `psi`, and the terminal
gradient.

These derivatives are approximated by local central differences, perturbing
only the variable block on which the local residual depends.

---

### 16.1 Perturbing `X_i`

For $i\ge 1$, the code perturbs each component of `X[i]`.

For coordinate $\ell$, it stores the old value, then evaluates `phi(i)` and
`psi(i)` at

$$
\begin{equation}
X_i^{+}
=
X_i+\varepsilon_x e_\ell,
\qquad
X_i^{-}
=
X_i-\varepsilon_x e_\ell.
\end{equation}
$$

It then computes

$$
\begin{equation}
\frac{\partial \phi(i)}{\partial X_i^{(\ell)}}
\approx
\frac{
\phi(i;X_i^{+})-\phi(i;X_i^{-})
}{
2\varepsilon_x
},
\end{equation}
$$

and

$$
\begin{equation}
\frac{\partial \psi(i)}{\partial X_i^{(\ell)}}
\approx
\frac{
\psi(i;X_i^{+})-\psi(i;X_i^{-})
}{
2\varepsilon_x
}.
\end{equation}
$$

These column vectors are inserted into the rows for $r_x^i$ and $r_p^i$ at the
column corresponding to $X_i^{(\ell)}$.

For $i=0$, this perturbation is skipped because $X_0$ is fixed and not part of
the unknown vector.

---

### 16.2 Perturbing `P_{i+1}`

For every interval $i$, the residual depends on $P_{i+1}$ through the
Hamiltonian-gradient evaluation point. Therefore, for every coordinate $\ell$,
the code perturbs `P[i + 1, ell]`.

It computes

$$
\begin{equation}
\frac{\partial \phi(i)}{\partial P_{i+1}^{(\ell)}}
\approx
\frac{
\phi(i;P_{i+1}^{+})-\phi(i;P_{i+1}^{-})
}{
2\varepsilon_p
},
\end{equation}
$$

and

$$
\begin{equation}
\frac{\partial \psi(i)}{\partial P_{i+1}^{(\ell)}}
\approx
\frac{
\psi(i;P_{i+1}^{+})-\psi(i;P_{i+1}^{-})
}{
2\varepsilon_p
}.
\end{equation}
$$

The resulting columns are inserted into:

- rows of $r_x^i$ for derivatives of `phi`;
- rows of $r_p^i$ for derivatives of `psi`.

For $r_p^i$, the column of $P_{i+1}$ already contains the exact linear
contribution $-I_n$. The finite-difference contribution is added on top of that
linear block.

---

### 16.3 Perturbing `X_N` in the boundary condition

The boundary residual depends on $X_N$ through $\nabla g(X_N)$.

For each coordinate $\ell$, the code perturbs `X[N, ell]` and evaluates
`bc_block()` at plus and minus perturbations:

$$
\begin{equation}
\frac{\partial r_{\mathrm{bc}}}{\partial X_N^{(\ell)}}
\approx
\frac{
r_{\mathrm{bc}}(X_N+\varepsilon_x e_\ell)
-
r_{\mathrm{bc}}(X_N-\varepsilon_x e_\ell)
}{
2\varepsilon_x
}.
\end{equation}
$$

Because

$$
\begin{equation}
r_{\mathrm{bc}}
=
P_N-\nabla g(X_N),
\end{equation}
$$

this block approximates

$$
\begin{equation}
-\nabla^2 g(X_N)
\end{equation}
$$

under the current terminal sign convention.

The implementation computes this block numerically rather than requiring an
analytic Hessian of the terminal cost.

---

## 17) Local sparsity pattern

The Jacobian sparsity follows directly from the residual dependencies:

- $r_x^i$ depends on $X_i$, $X_{i+1}$, and $P_{i+1}$;
- $r_p^i$ depends on $X_i$, $P_i$, and $P_{i+1}$;
- $r_{\mathrm{bc}}$ depends on $X_N$ and $P_N$.

Since $X_0$ is fixed, there is no column for $X_0$.

This local dependence is why the Jacobian can be assembled as a sparse
block-stencil matrix rather than as a dense global finite-difference matrix.

---

## 18) How this module is used

The functions in `integrators.py` are used indirectly by the Newton solver
through `shooting.py`.

The shooting layer receives a packed vector `z`, unpacks it into `(X,P)`, and
then calls

```python
assemble_residual(...)
```

or

```python
assemble_jacobian(...)
```

The Newton layer then solves linearized systems of the form

$$
\begin{equation}
J(z^{(k)})\Delta z^{(k)}
=
-F(z^{(k)}).
\end{equation}
$$

The updated vector is unpacked again into state and costate trajectories.

Thus, `integrators.py` is the layer where the mathematical discrete PMP system
is converted into arrays and sparse matrices suitable for Newton's method.

---

## 19) Consistency requirements

The residual and Jacobian must use exactly the same conventions. In the current
implementation, these conventions are:

1. unknown ordering:

$$
\begin{equation}
z=(X_1,\dots,X_N,P_0,\dots,P_N);
\end{equation}
$$

2. residual ordering:

$$
\begin{equation}
F=(r_x^0,r_p^0,\dots,r_x^{N-1},r_p^{N-1},r_{\mathrm{bc}});
\end{equation}
$$

3. Hamiltonian-gradient evaluation point:

$$
\begin{equation}
(P_{i+1},X_i,t_i);
\end{equation}
$$

4. residual signs:

$$
\begin{equation}
r_x^i
=
X_{i+1}
-
X_i
-
\Delta t_i H_p(P_{i+1},X_i,t_i),
\end{equation}
$$

$$
\begin{equation}
r_p^i
=
P_i
-
P_{i+1}
-
\Delta t_i H_x(P_{i+1},X_i,t_i);
\end{equation}
$$

5. terminal residual:

$$
\begin{equation}
r_{\mathrm{bc}}
=
P_N-\nabla g(X_N).
\end{equation}
$$

If any of these conventions change, both `assemble_residual` and
`assemble_jacobian` must be updated together.

---

## 20) Explicit-gradient mode

Both residual and Jacobian assembly accept

```python
use_explicit_gradients: bool = False
```

When this flag is false, Hamiltonian gradients are obtained from the smoothed PA
Hamiltonian:

```python
eval_H_smooth(problem, bundle, p, x, t, delta, dt=dt)
```

When this flag is true and the problem provides `hamiltonian_grad_fn`, the code
uses

```python
problem.hamiltonian_gradients(x, p, t)
```

This changes the source of

$$
\begin{equation}
H_p,
\qquad
H_x,
\end{equation}
$$

but not the residual structure, unknown ordering, or Jacobian sparsity pattern.

---

## 21) Role of `dt`

The local time step

$$
\begin{equation}
\Delta t_i=t_{i+1}-t_i
\end{equation}
$$

appears explicitly in the residual blocks and is also passed to
`_hamiltonian_gradients`.

In the default smoothing implementation, `dt` is not used directly. However,
the argument is important because some problem-specific Hamiltonian,
feasibility, or smoothing logic may depend on the local step size.

The current integrator consistently passes

```python
dt=dt
```

when evaluating Hamiltonian gradients.

---

## 22) Computational cost

### 22.1 Residual cost

Each residual evaluation calls the Hamiltonian-gradient routine once per time
interval. Therefore, the residual cost scales like

$$
\begin{equation}
O(N\,C_H),
\end{equation}
$$

where $C_H$ is the cost of one Hamiltonian-gradient evaluation.

In the default smoothed PA mode, $C_H$ depends on:

- the number of bundle controls;
- the state dimension;
- the cost of evaluating `problem.f` and `problem.l`;
- the finite differences used inside `eval_H_smooth` to compute `grad_x`.

---

### 22.2 Jacobian cost

The Jacobian uses local finite differences of `phi`, `psi`, and `bc_block`.

For each interval, it perturbs:

- $X_i$ when $i\ge 1$;
- $P_{i+1}$ always.

For each perturbation coordinate, it evaluates both `phi(i)` and `psi(i)` at
plus and minus perturbations.

Thus the Jacobian cost scales roughly like

$$
\begin{equation}
O(Nn\,C_H),
\end{equation}
$$

where $n$ is the state dimension.

This is much cheaper than a global finite-difference Jacobian that would
perturb each component of $z$ and recompute the full residual every time. Such a
global approach would scale more like

$$
\begin{equation}
O((2N+1)n \cdot N C_H).
\end{equation}
$$

The current implementation is therefore more efficient because it exploits the
local block structure.

---

## 23) Numerical notes

### 23.1 In-place perturbations

The Jacobian routine perturbs entries of `X` and `P` in place and restores them
immediately afterward.

For example:

```python
old = X[i, ell]
X[i, ell] = old + eps_x
...
X[i, ell] = old - eps_x
...
X[i, ell] = old
```

This is efficient, but it assumes single-threaded execution. If the Jacobian
assembly is parallelized in the future, this in-place perturbation pattern would
need to be redesigned.

---

### 23.2 Finite-difference steps

The current finite-difference steps are:

```python
eps_x = 1e-7
eps_p = 1e-7
epsg = 1e-6
```

where:

- `eps_x` is used for state perturbations in the Jacobian;
- `eps_p` is used for costate perturbations in the Jacobian;
- `epsg` is used to approximate $\nabla g(X_N)$ inside `bc_block`.

If Newton becomes noisy, stagnates, or produces an inaccurate linear solve,
these finite-difference scales are important diagnostics.

---

### 23.3 Terminal Hessian approximation

The boundary block with respect to $X_N$ is computed by finite-differencing
`bc_block`, while `bc_block` itself computes $\nabla g(X_N)$ by finite
differences.

Therefore, this part approximates a terminal Hessian-like block by nested finite
differences.

Under the current sign convention,

$$
\begin{equation}
r_{\mathrm{bc}}
=
P_N-\nabla g(X_N),
\end{equation}
$$

so the analytic terminal $X_N$ block would be

$$
\begin{equation}
-\nabla^2 g(X_N).
\end{equation}
$$

---

### 23.4 Sparse matrix format

The Jacobian is built as COO triplets and converted to CSR:

```python
J = coo_matrix((data, (rows, cols)), shape=(m, m)).tocsr()
J.sum_duplicates()
```

CSR is appropriate for the Newton solver, which later converts or factors the
matrix as needed.

---

## 24) Debugging checklist

When residual or Jacobian behavior looks wrong, check the following points.

---

### 24.1 Residual and Jacobian sign consistency

Verify that the residual uses

$$
\begin{equation}
r_x^i
=
X_{i+1}-X_i-\Delta t_i H_p,
\end{equation}
$$

and

$$
\begin{equation}
r_p^i
=
P_i-P_{i+1}-\Delta t_i H_x.
\end{equation}
$$

The Jacobian formulas must match these signs.

---

### 24.2 Terminal condition sign

Check that both residual and Jacobian use

$$
\begin{equation}
r_{\mathrm{bc}}
=
P_N-\nabla g(X_N).
\end{equation}
$$

A sign mismatch here changes the terminal Hessian contribution and can break
Newton convergence.

---

### 24.3 Unknown ordering

Make sure every packing, unpacking, and Jacobian index map uses

$$
\begin{equation}
z=(X_1,\dots,X_N,P_0,\dots,P_N).
\end{equation}
$$

A mismatch in unknown ordering usually produces a Jacobian with the right shape
but wrong columns.

---

### 24.4 Fixed initial state

Remember that $X_0$ is not an unknown. Therefore:

- `col_x(0)` is not valid;
- there is no Jacobian column for $X_0$;
- derivatives with respect to $X_i$ are inserted only for $i\ge 1$.

---

### 24.5 Evaluation point

Both residual and Jacobian must evaluate Hamiltonian gradients at

$$
\begin{equation}
(P_{i+1},X_i,t_i).
\end{equation}
$$

If this point changes, the residual formulas and Jacobian dependencies change.

---

### 24.6 Shape consistency

Check that

```python
X.shape == (N + 1, n)
P.shape == (N + 1, n)
t_nodes.shape == (N + 1,)
```

and that `problem.x0.shape == (n,)`.

Shape mismatches can silently corrupt reshaping in `pack_unknowns` or
`unpack_unknowns`.

---

### 24.7 Sparse Jacobian sanity

A quick structural check is:

```python
J.shape == ((2 * N + 1) * n, (2 * N + 1) * n)
```

and `J` should be sparse with nonzeros near the local block stencil. If the
Jacobian is unexpectedly dense or has wrong dimensions, inspect the row/column
maps.

---

## 25) Summary

`integrators.py` defines the discrete algebraic PMP system used by the solver.

It uses the packed unknown vector

$$
\begin{equation}
z=(X_1,\dots,X_N,P_0,\dots,P_N),
\end{equation}
$$

and assembles the residual

$$
\begin{equation}
F=(r_x^0,r_p^0,\dots,r_x^{N-1},r_p^{N-1},r_{\mathrm{bc}}).
\end{equation}
$$

The current step residuals are

$$
\begin{equation}
r_x^i
=
X_{i+1}
-
X_i
-
\Delta t_i H_p(P_{i+1},X_i,t_i),
\end{equation}
$$

and

$$
\begin{equation}
r_p^i
=
P_i
-
P_{i+1}
-
\Delta t_i H_x(P_{i+1},X_i,t_i).
\end{equation}
$$

The terminal residual is

$$
\begin{equation}
r_{\mathrm{bc}}
=
P_N-\nabla g(X_N).
\end{equation}
$$

The Jacobian is assembled as a sparse local block-stencil matrix, with exact
linear blocks and finite-difference nonlinear blocks.

This module is therefore the numerical core that turns the smoothed Pontryagin
conditions into a Newton-solvable algebraic system.