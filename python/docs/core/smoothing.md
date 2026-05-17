# `core/smoothing.py` — Smoothed PA Hamiltonian (`eval_H_smooth`)

This module implements the smoothed Hamiltonian evaluation used by the
fixed-mesh TPBVP solver and by the adaptive error indicators.

The main routine is

```python
eval_H_smooth(problem, bundle, p, x, t, delta, dt=None)
```

It returns

```python
H_delta, grad_p, grad_x
```

where:

- `H_delta` is the smoothed Hamiltonian value;
- `grad_p` is the gradient with respect to the costate $p$;
- `grad_x` is the gradient with respect to the state $x$.

The default implementation smooths the PA-bundle Hamiltonian using a
log-sum-exp soft minimum. However, the routine also supports a problem-specific
override: if the problem supplies `problem.hamiltonian_smooth_fn`, then
`eval_H_smooth` delegates directly to that callback.

---

## 1) Function contract

The current function signature is

```python
def eval_H_smooth(
    problem,
    bundle,
    p: np.ndarray,
    x: np.ndarray,
    t: float,
    delta: float,
    dt: Optional[float] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    ...
```

---

### 1.1 Inputs

| Argument | Meaning |
|---|---|
| `problem` | The `OCPProblem` instance providing dynamics, running cost, and optional custom smoothing. |
| `bundle` | The PA bundle containing the controls over which the default smoothed Hamiltonian is built. |
| `p` | Costate vector, shape `(n,)`. |
| `x` | State vector, shape `(n,)`. |
| `t` | Current time. |
| `delta` | Smoothing parameter. Smaller $\delta$ gives a sharper approximation to the nonsmooth PA minimum. |
| `dt` | Optional local time step. The current default implementation does not use `dt`, but the argument is part of the interface because related Hamiltonian/oracle/feasibility routines may depend on local step size. |

---

### 1.2 Outputs

The routine returns

```python
H_delta, grad_p, grad_x
```

with

```python
type(H_delta) == float
grad_p.shape == p.shape
grad_x.shape == x.shape
```

Mathematically,

$$
\begin{equation}
\texttt{grad\_p}
=
\nabla_p H_\delta(p,x,t),
\qquad
\texttt{grad\_x}
=
\nabla_x H_\delta(p,x,t).
\end{equation}
$$

These gradients are the quantities used by the discrete PMP residual and by the
time-discretization indicator.

---

## 2) Custom smoothing hook

The first branch in the code is

```python
if getattr(problem, "hamiltonian_smooth_fn", None) is not None:
    return problem.hamiltonian_smooth(x, p, t, delta)
```

Thus, if the problem object provides a custom smooth-Hamiltonian callback, the
default PA-bundle log-sum-exp computation is skipped.

The callback is expected to return

```python
H_delta, grad_p, grad_x
```

with the same meaning as the default implementation.

This hook is useful when a problem has an analytic smoothed Hamiltonian or a
problem-specific regularization that should replace the generic PA-bundle
soft-min.

Note that the custom callback is called as

```python
problem.hamiltonian_smooth(x, p, t, delta)
```

so the current custom hook does not receive `dt`.

---

## 3) Bundle planes and notation

Assume no custom smoothing hook is supplied. Then the routine uses the controls
currently stored in the PA bundle.

Let

$$
\begin{equation}
U_{\mathrm{bundle}}
=
\{u^{(1)},\dots,u^{(M)}\}
\end{equation}
$$

be the set of bundle controls. We use $M$ for the number of bundle controls
or planes. This avoids conflict with the control dimension, often denoted by
$m$.

For each bundle control $u^{(i)}$, define the plane value

$$
\begin{equation}
g_i(p,x,t)
=
p^\top f(x,u^{(i)},t)
+
\ell(x,u^{(i)},t),
\qquad
i=1,\dots,M.
\end{equation}
$$

The PA-bundle Hamiltonian surrogate is the hard minimum

$$
\begin{equation}
\bar H(p,x,t)
=
\min_{1\le i\le M}
g_i(p,x,t).
\end{equation}
$$

This function is generally nonsmooth at points where two or more planes tie for
the minimum. The purpose of `eval_H_smooth` is to replace this hard minimum by a
smooth approximation.

---

## 4) Important assumption: the bundle is not filtered here

The default implementation converts the bundle controls to floating-point arrays:

```python
feasible_controls = [np.asarray(u, dtype=float) for u in bundle.controls]
```

Despite the variable name `feasible_controls`, this routine does **not** perform
a new feasibility filter and does **not** project controls. It simply smooths
over the controls already stored in the bundle.

Therefore, the surrounding algorithm is responsible for constructing and
maintaining a meaningful PA bundle.

If the bundle is empty, the code raises

```python
RuntimeError("PABundle is empty; cannot compute smooth Hamiltonian.")
```

because neither $\bar H$ nor $H_\delta$ is defined without at least one plane.

---

## 5) Log-sum-exp soft minimum

The smoothed Hamiltonian is

$$
\begin{equation}
H_\delta(p,x,t)
=
-\delta
\log
\left(
\sum_{i=1}^M
\exp
\left(
-\frac{g_i(p,x,t)}{\delta}
\right)
\right),
\qquad
\delta>0.
\end{equation}
$$

This is a smooth approximation of the hard minimum $\bar H$. Let

$$
\begin{equation}
g_\star
=
\min_{1\le i\le M} g_i(p,x,t).
\end{equation}
$$

Then the soft-min satisfies

$$
\begin{equation}
g_\star-\delta\log M
\le
H_\delta(p,x,t)
\le
g_\star.
\end{equation}
$$

Therefore,

$$
\begin{equation}
H_\delta(p,x,t)
\le
\bar H(p,x,t),
\end{equation}
$$

and

$$
\begin{equation}
H_\delta(p,x,t)
\longrightarrow
\bar H(p,x,t)
\qquad
\text{as }
\delta\downarrow 0.
\end{equation}
$$

The parameter $\delta$ acts like a temperature:

- large $\delta$ gives a smoother, more averaged Hamiltonian;
- small $\delta$ gives a sharper approximation to the PA minimum;
- very small $\delta$ can make the weights nearly one-hot and may worsen
  conditioning.

---

## 6) Soft-min weights

Define the soft-min weights

$$
\begin{equation}
w_i(p,x,t;\delta)
=
\frac{
\exp\left(-g_i(p,x,t)/\delta\right)
}{
\sum_{j=1}^M
\exp\left(-g_j(p,x,t)/\delta\right)
},
\qquad
i=1,\dots,M.
\end{equation}
$$

They satisfy

$$
\begin{equation}
w_i\ge 0,
\qquad
\sum_{i=1}^M w_i=1.
\end{equation}
$$

The log-sum-exp derivative gives the identity

$$
\begin{equation}
\nabla H_\delta(p,x,t)
=
\sum_{i=1}^M
w_i(p,x,t;\delta)
\nabla g_i(p,x,t).
\end{equation}
$$

This identity is the reason smoothing is useful: the nonsmooth active-plane
selection in $\bar H$ is replaced by a smooth weighted average over planes.

---

## 7) Gradient with respect to the costate

For each plane,

$$
\begin{equation}
g_i(p,x,t)
=
p^\top f(x,u^{(i)},t)
+
\ell(x,u^{(i)},t).
\end{equation}
$$

Since $f$ and $\ell$ do not depend on $p$ directly,

$$
\begin{equation}
\nabla_p g_i(p,x,t)
=
f(x,u^{(i)},t).
\end{equation}
$$

Therefore,

$$
\begin{equation}
\nabla_p H_\delta(p,x,t)
=
\sum_{i=1}^M
w_i
f(x,u^{(i)},t).
\end{equation}
$$

The code returns this vector as `grad_p`.

This quantity is the smoothed state direction in the PMP system:

$$
\begin{equation}
\dot x
=
\nabla_p H_\delta(p,x,t).
\end{equation}
$$

Because the weights form a convex combination, `grad_p` is a weighted blend of
the candidate dynamics stored in the bundle.

---

## 8) Gradient with respect to the state

For each plane,

$$
\begin{equation}
\nabla_x g_i(p,x,t)
=
\left(\nabla_x f(x,u^{(i)},t)\right)^\top p
+
\nabla_x \ell(x,u^{(i)},t).
\end{equation}
$$

Thus,

$$
\begin{equation}
\nabla_x H_\delta(p,x,t)
=
\sum_{i=1}^M
w_i
\nabla_x g_i(p,x,t).
\end{equation}
$$

This is the quantity returned as `grad_x`.

It enters the PMP costate equation through

$$
\begin{equation}
-\dot p
=
\nabla_x H_\delta(p,x,t).
\end{equation}
$$

The current implementation does not require analytic derivatives of $f$ or
$\ell$ with respect to $x$. Instead, it approximates each
$\nabla_x g_i$ by central finite differences, as described in the next section.

---

## 9) Implementation workflow

The default code path in `eval_H_smooth` follows this sequence:

```python
if problem has custom hamiltonian_smooth_fn:
    return problem.hamiltonian_smooth(x, p, t, delta)

feasible_controls = [np.asarray(u, dtype=float) for u in bundle.controls]

if no controls:
    raise RuntimeError

evaluate g_vals and f_vals for all controls

compute shifted exponentials
compute H_delta
compute weights

grad_p = weighted sum of f_vals

for each state coordinate:
    finite-difference all plane values g_i
    combine plane derivatives with weights
    store component in grad_x

return H_delta, grad_p, grad_x
```

The variable names in the implementation correspond to the following
mathematical quantities.

| Code variable | Mathematical meaning |
|---|---|
| `feasible_controls[i]` | Bundle control $u^{(i)}$. |
| `g_vals[i]` | Plane value $g_i(p,x,t)$. |
| `f_vals[i, :]` | Dynamics value $f(x,u^{(i)},t)$. |
| `g_min` | Minimum plane value $\min_i g_i$. |
| `exps[i]` | Shifted exponential numerator for soft-min weights. |
| `sum_exps` | Sum of shifted exponentials. |
| `weights[i]` | Soft-min weight $w_i$. |
| `grad_p` | $\nabla_p H_\delta(p,x,t)$. |
| `grad_x` | $\nabla_x H_\delta(p,x,t)$. |

---

## 10) Evaluating the bundle planes

After the optional custom hook branch, the code reads the bundle controls:

```python
feasible_controls = [np.asarray(u, dtype=float) for u in bundle.controls]
M = len(feasible_controls)
```

The code calls this length `m`, but mathematically we denote it by $M$ to avoid
confusing it with the control dimension.

If $M=0$, the function raises an error because there is no minimum to smooth.

Then it allocates:

```python
g_vals = np.empty(M)
f_vals = np.empty((M, n))
```

where $n$ is the state dimension, obtained from `p.size`.

For each bundle control $u^{(i)}$, the code computes

```python
f_i = problem.f(x, u, t)
f_vals[i, :] = f_i
g_vals[i] = float(np.dot(p, f_i) + problem.l(x, u, t))
```

Mathematically,

$$
\begin{equation}
f_i
=
f(x,u^{(i)},t),
\end{equation}
$$

and

$$
\begin{equation}
g_i
=
p^\top f_i
+
\ell(x,u^{(i)},t).
\end{equation}
$$

The array `f_vals` is saved because the $p$-gradient is the weighted average of
these dynamics values.

---

## 11) Stable log-sum-exp computation

A direct implementation of

$$
\begin{equation}
H_\delta
=
-\delta
\log
\left(
\sum_{i=1}^M
\exp\left(-\frac{g_i}{\delta}\right)
\right)
\end{equation}
$$

can overflow or underflow when $\delta$ is small. The code therefore shifts by

```python
g_min = np.min(g_vals)
```

Let

$$
\begin{equation}
g_{\min}
=
\min_i g_i.
\end{equation}
$$

Then

$$
\begin{equation}
\sum_{i=1}^M
\exp\left(-\frac{g_i}{\delta}\right)
=
\exp\left(-\frac{g_{\min}}{\delta}\right)
\sum_{i=1}^M
\exp\left(-\frac{g_i-g_{\min}}{\delta}\right).
\end{equation}
$$

Therefore,

$$
\begin{equation}
H_\delta
=
g_{\min}
-
\delta
\log
\left(
\sum_{i=1}^M
\exp
\left(
-\frac{g_i-g_{\min}}{\delta}
\right)
\right).
\end{equation}
$$

The implementation computes the shifted exponentials as

```python
exps = np.exp(-(g_vals - g_min) / max(delta, 1e-12))
```

The denominator uses `max(delta, 1e-12)` to avoid division by zero or extremely
small denominators.

Then

```python
sum_exps = np.sum(exps)
H_delta = g_min - delta * np.log(sum_exps + 1e-300)
```

The tiny constant `1e-300` prevents `np.log(0)` if the exponential sum underflows.

**Exact code detail.** The exponentials use `max(delta, 1e-12)`, but the final
formula multiplies the logarithm by the original `delta`:

```python
H_delta = g_min - delta * np.log(...)
```

For ordinary positive $\delta$ this is the expected stable formula. The
`max(delta, 1e-12)` safeguard only affects the exponential denominator.

---

## 12) Soft-min weights in code

After computing `exps`, the weights are

```python
weights = exps / sum_exps
```

Mathematically,

$$
\begin{equation}
w_i
=
\frac{
\exp\left(-(g_i-g_{\min})/\delta\right)
}{
\sum_{j=1}^M
\exp\left(-(g_j-g_{\min})/\delta\right)
}.
\end{equation}
$$

This is equivalent to the unshifted definition because the common factor
$\exp(-g_{\min}/\delta)$ cancels.

The weights satisfy

$$
\begin{equation}
w_i\ge 0,
\qquad
\sum_{i=1}^M w_i=1.
\end{equation}
$$

When $\delta$ is small, most weight concentrates on the controls whose plane
values are near the minimum. When $\delta$ is larger, the weights are more
spread out across the bundle.

---

## 13) Computing `grad_p`

The code computes

```python
grad_p = np.sum(weights[:, None] * f_vals, axis=0)
```

This is exactly the formula

$$
\begin{equation}
\nabla_p H_\delta(p,x,t)
=
\sum_{i=1}^M
w_i
f(x,u^{(i)},t).
\end{equation}
$$

The broadcasting expression `weights[:, None] * f_vals` multiplies every row
`f_vals[i, :]` by the corresponding scalar weight `weights[i]`.

Thus `grad_p` has shape `(n,)`, the same as `p` and `x`.

---

## 14) Computing `grad_x` by finite differences

The code initializes

```python
grad_x = np.zeros_like(x)
eps = 1e-6
```

The fixed finite-difference step is

$$
\begin{equation}
\varepsilon
=
10^{-6}.
\end{equation}
$$

For each state coordinate `dim`, the code forms

```python
x_plus = x.copy()
x_minus = x.copy()

x_plus[dim] += eps
x_minus[dim] -= eps
```

Mathematically, for coordinate $k$,

$$
\begin{equation}
x^+
=
x+\varepsilon e_k,
\qquad
x^-
=
x-\varepsilon e_k.
\end{equation}
$$

Then, for every bundle control $u^{(i)}$, the code recomputes the plane value at
the perturbed states:

```python
f_plus = problem.f(x_plus, u, t)
f_minus = problem.f(x_minus, u, t)

l_plus = problem.l(x_plus, u, t)
l_minus = problem.l(x_minus, u, t)

g_plus[i] = np.dot(p, f_plus) + l_plus
g_minus[i] = np.dot(p, f_minus) + l_minus
```

So

$$
\begin{equation}
g_i^+
=
p^\top f(x+\varepsilon e_k,u^{(i)},t)
+
\ell(x+\varepsilon e_k,u^{(i)},t),
\end{equation}
$$

and

$$
\begin{equation}
g_i^-
=
p^\top f(x-\varepsilon e_k,u^{(i)},t)
+
\ell(x-\varepsilon e_k,u^{(i)},t).
\end{equation}
$$

The code approximates the partial derivative of each plane by

```python
dg_dx = (g_plus - g_minus) / (2.0 * eps)
```

That is,

$$
\begin{equation}
\frac{\partial g_i}{\partial x_k}(p,x,t)
\approx
\frac{g_i^+-g_i^-}{2\varepsilon}.
\end{equation}
$$

Finally, it combines these plane derivatives using the soft-min weights computed
at the base point:

```python
grad_x[dim] = float(np.sum(weights * dg_dx))
```

Thus

$$
\begin{equation}
\frac{\partial H_\delta}{\partial x_k}(p,x,t)
\approx
\sum_{i=1}^M
w_i
\frac{g_i^+-g_i^-}{2\varepsilon}.
\end{equation}
$$

Repeating this for all coordinates gives

$$
\begin{equation}
\nabla_x H_\delta(p,x,t)
\approx
\left(
\frac{\partial H_\delta}{\partial x_1},
\dots,
\frac{\partial H_\delta}{\partial x_n}
\right).
\end{equation}
$$

---

### 14.1 Why there is no explicit derivative of the weights

The formula

$$
\begin{equation}
\nabla_x H_\delta
=
\sum_{i=1}^M
w_i
\nabla_x g_i
\end{equation}
$$

already includes the derivative of the log-sum-exp expression. Therefore, the
implementation should not separately differentiate the weights.

The code correctly computes the weights at the base point and uses them to form
a weighted sum of plane derivatives.

---

### 14.2 Computational cost

For each state coordinate and each bundle control, the code evaluates both
$f$ and $\ell$ at $x^+$ and $x^-$. Therefore, computing `grad_x` costs about

$$
\begin{equation}
2Mn
\end{equation}
$$

extra evaluations of `problem.f` and `problem.l`, where:

- $M$ is the number of bundle controls;
- $n$ is the state dimension.

This is why large bundles or high-dimensional states can make smoothing
evaluation expensive.

---

## 15) Where `eval_H_smooth` is used

The routine appears in several parts of the solver.

---

### 15.1 Fixed-mesh TPBVP residual

The main fixed-mesh integrator calls `eval_H_smooth` through
`_hamiltonian_gradients` in `integrators.py`.

The current residual evaluates the Hamiltonian gradients at the
symplectic-Euler point

$$
\begin{equation}
(P_{i+1},X_i,t_i).
\end{equation}
$$

The residual blocks are

$$
\begin{equation}
r_x^i
=
X_{i+1}
-
X_i
-
\Delta t_i
\nabla_p H_\delta(P_{i+1},X_i,t_i),
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
\Delta t_i
\nabla_x H_\delta(P_{i+1},X_i,t_i).
\end{equation}
$$

Thus `grad_p` drives the discrete state equation, and `grad_x` drives the
discrete costate equation.

---

### 15.2 Adaptive time indicator

The adaptive loop also evaluates Hamiltonian gradients at the same type of
symplectic-Euler point:

$$
\begin{equation}
(P_{i+1},X_i,t_i).
\end{equation}
$$

It computes the time-indicator density

$$
\begin{equation}
\rho_i
=
-\frac{1}{2}
\nabla_p H_\delta(P_{i+1},X_i,t_i)^\top
\nabla_x H_\delta(P_{i+1},X_i,t_i).
\end{equation}
$$

This is why `eval_H_smooth` must provide consistent values of both gradients.

---

### 15.3 Smoothing indicator

The adaptive loop also uses the Hamiltonian value `H_delta` to estimate the
smoothing error. It compares the PA hard minimum $\bar H$ to $H_\delta$ at mesh
nodes:

$$
\begin{equation}
e_i^\delta
=
\bar H(P_i,X_i,t_i)
-
H_\delta(P_i,X_i,t_i).
\end{equation}
$$

The local smoothing contribution is computed by a trapezoidal rule:

$$
\begin{equation}
\eta_i^\delta
=
\frac{1}{2}
\left(
e_i^\delta
+
e_{i+1}^\delta
\right)
\Delta t_i.
\end{equation}
$$

---

## 16) Practical numerical notes

### 16.1 Choice of `delta`

The smoothing parameter $\delta$ controls the tradeoff between smoothness and
accuracy.

Small $\delta$ gives

$$
\begin{equation}
H_\delta
\approx
\bar H,
\end{equation}
$$

but the weights may become nearly one-hot. This can make the nonlinear TPBVP
less smooth numerically.

Large $\delta$ gives smoother weights, but increases the gap

$$
\begin{equation}
\bar H-H_\delta.
\end{equation}
$$

The adaptive loop reduces $\delta$ when the smoothing indicator is too large.

---

### 16.2 Fixed finite-difference step

The current implementation uses a fixed step

$$
\begin{equation}
\varepsilon=10^{-6}
\end{equation}
$$

for all state components. This is simple and robust for many examples, but it is
not scale-aware.

If different components of $x$ have very different magnitudes, a relative or
componentwise finite-difference step may be more accurate.

---

### 16.3 Bundle size

The bundle must contain at least one control. If `bundle.controls` is empty, the
routine raises a `RuntimeError`.

As $M$ grows, the computation becomes more expensive because every smoothing
evaluation loops over all bundle controls, and the finite-difference state
gradient costs about $2Mn$ extra model evaluations.

---

### 16.4 Feasibility of bundle controls

The default smoothing routine does not re-check feasibility of controls in the
bundle. It assumes that the bundle was built consistently by the surrounding
algorithm.

If the bundle contains controls that are inappropriate for the current state,
then the smoothed Hamiltonian will still include them. This is one reason why
`adaptivity.py` may refresh bundle support controls after mesh refinement for
problems with time-step-dependent feasibility.

---

### 16.5 Custom smooth Hamiltonians

If `problem.hamiltonian_smooth_fn` is provided, all of the default PA-bundle
logic is bypassed. In that case, correctness depends on the custom callback
returning a value and gradients consistent with the expected convention:

$$
\begin{equation}
(H_\delta,\nabla_p H_\delta,\nabla_x H_\delta).
\end{equation}
$$

---

## 17) Summary

The routine `eval_H_smooth` provides the differentiable Hamiltonian used by the
Newton-based fixed-mesh solver.

In the default PA-bundle mode, it computes

$$
\begin{equation}
H_\delta(p,x,t)
=
-\delta
\log
\left(
\sum_{i=1}^M
\exp
\left(
-\frac{
p^\top f(x,u^{(i)},t)+\ell(x,u^{(i)},t)
}{\delta}
\right)
\right).
\end{equation}
$$

It returns:

1. the smoothed value $H_\delta$;
2. the costate gradient $\nabla_p H_\delta$, computed exactly as a weighted
   average of candidate dynamics;
3. the state gradient $\nabla_x H_\delta$, computed as a weighted average of
   finite-difference plane derivatives.

This routine is the bridge between a nonsmooth PA-bundle Hamiltonian and the
smooth gradients required by the Newton TPBVP solver and adaptive indicators.

---

## 18) Debugging checklist

When `eval_H_smooth` returns unexpected values, gradients, or raises an error,
check the following points.

---

### 18.1 Empty PA bundle

The default implementation requires

```python
len(bundle.controls) > 0
```

If the bundle is empty, the routine raises

```python
RuntimeError("PABundle is empty; cannot compute smooth Hamiltonian.")
```

This usually means that the adaptive solver or experiment did not seed the PA
bundle before calling the fixed-mesh solver.

---

### 18.2 Shape consistency

Check that the state and costate vectors have compatible shapes:

```python
p.shape == x.shape == (n,)
```

and that every bundle control has the expected control shape:

```python
u.shape == (m,)
```

Avoid column-shaped arrays such as `(n, 1)` or `(m, 1)`, because NumPy
broadcasting can silently produce incorrect dot products or array assignments.

---

### 18.3 Non-finite plane values

The plane values are computed as

$$
\begin{equation}
g_i
=
p^\top f(x,u^{(i)},t)
+
\ell(x,u^{(i)},t).
\end{equation}
$$

If `problem.f` or `problem.l` returns `NaN` or `inf`, then `g_vals`,
`H_delta`, and the gradients may also become non-finite.

Check the current state, costate, controls, and model functions if this occurs.

---

### 18.4 Very small `delta`

The code protects the exponential denominator using

```python
max(delta, 1e-12)
```

but extremely small or nonpositive values of `delta` are still conceptually
problematic. The smoothing parameter should be positive:

$$
\begin{equation}
\delta>0.
\end{equation}
$$

If $\delta$ is too small, the weights can become nearly one-hot, and the
smoothed Hamiltonian behaves almost like the nonsmooth hard minimum. This may
make the Newton solve more difficult.

---

### 18.5 Finite-difference sensitivity

The state gradient uses the fixed central-difference step

$$
\begin{equation}
\varepsilon=10^{-6}.
\end{equation}
$$

If the state variables have very different scales, this step may be too large
for some components and too small for others.

Symptoms can include:

- noisy `grad_x`,
- poor Newton convergence,
- sensitivity to small changes in the mesh or initial guess.

In such cases, consider whether the problem needs analytic Hamiltonian
gradients, a custom `hamiltonian_smooth_fn`, or a scale-aware finite-difference
step.

---

### 18.6 Custom smoothing callback

If the problem provides

```python
problem.hamiltonian_smooth_fn
```

then the default PA-bundle smoothing code is bypassed.

When debugging, first check whether the routine is using the custom branch or
the default branch. A custom callback must return

```python
H_delta, grad_p, grad_x
```

with shapes and conventions consistent with the default implementation.

---

### 18.7 Bundle controls are not re-filtered

The default smoothing routine does not check

```python
problem.local_control_feasible(...)
```

for each bundle control.

It assumes that the bundle controls are meaningful for the current solve. If a
problem has state-dependent or step-size-dependent feasibility, inspect the
bundle construction and refresh logic in `adaptivity.py`.

---

## 19) Final conceptual takeaway

The purpose of `eval_H_smooth` is to replace a nonsmooth minimum over PA-bundle
planes by a differentiable soft minimum.

The hard PA surrogate is

$$
\begin{equation}
\bar H(p,x,t)
=
\min_i g_i(p,x,t).
\end{equation}
$$

The smoothed surrogate is

$$
\begin{equation}
H_\delta(p,x,t)
=
-\delta
\log
\left(
\sum_i
\exp
\left(
-\frac{g_i(p,x,t)}{\delta}
\right)
\right).
\end{equation}
$$

The returned gradients are

$$
\begin{equation}
\nabla_p H_\delta
=
\sum_i w_i f(x,u^{(i)},t),
\end{equation}
$$

and

$$
\begin{equation}
\nabla_x H_\delta
\approx
\sum_i w_i
\nabla_x g_i,
\end{equation}
$$

where $\nabla_x g_i$ is approximated by central finite differences in the
default implementation.

Thus the routine turns switching between bundle controls into smooth weights,
which is what allows the PA-bundle approximation to be used inside a
Newton-based shooting method.