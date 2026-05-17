# `core/pa_bundle.py` — Piecewise-affine control bundle (`PABundle`)

This module defines the class

```python
PABundle
```

The bundle is a small passive container for control candidates. It stores a
finite list of controls and uses them to evaluate a piecewise-affine surrogate
Hamiltonian.

The current implementation is intentionally simple:

- it stores controls in `self.controls`;
- it avoids adding near-duplicate controls;
- it evaluates the Hamiltonian over the stored controls;
- it can optionally filter stored controls through local feasibility logic.

It does **not** generate controls by itself, does **not** project controls to
bounds when inserting them, and does **not** implement a maximum capacity or
replacement policy.

---

## 1) Mathematical role of the PA bundle

For a fixed state $x$, costate $p$, time $t$, and control $u$, define the
Hamiltonian integrand

$$
\begin{equation}
\mathcal{H}(p,x,u,t)
=
p^\top f(x,u,t)
+
\ell(x,u,t).
\end{equation}
$$

The true Hamiltonian in this repository uses the minimum convention:

$$
\begin{equation}
H(p,x,t)
=
\min_{u\in A}
\mathcal{H}(p,x,u,t),
\end{equation}
$$

where $A$ is the admissible control set.

The PA bundle stores a finite set of controls

$$
\begin{equation}
U_{\mathrm{bundle}}
=
\{u^{(1)},\dots,u^{(M)}\}.
\end{equation}
$$

The bundle surrogate Hamiltonian is

$$
\begin{equation}
\bar H(p,x,t)
=
\min_{1\le i\le M}
\left\{
p^\top f(x,u^{(i)},t)
+
\ell(x,u^{(i)},t)
\right\}.
\end{equation}
$$

For fixed $(x,t)$, each stored control defines an affine function of $p$:

$$
\begin{equation}
\phi_i(p;x,t)
=
p^\top f(x,u^{(i)},t)
+
\ell(x,u^{(i)},t).
\end{equation}
$$

Therefore, $\bar H$ is a minimum of affine functions in $p$. It is
piecewise-affine and generally nonsmooth at points where multiple stored
controls tie.

---

## 2) Upper-bound interpretation

Assume the bundle controls are admissible controls, so that

$$
\begin{equation}
U_{\mathrm{bundle}}
\subseteq
A.
\end{equation}
$$

Because the true Hamiltonian minimizes over the larger set $A$, while the
bundle surrogate minimizes only over $U_{\mathrm{bundle}}$, we have

$$
\begin{equation}
H(p,x,t)
\le
\bar H(p,x,t).
\end{equation}
$$

Thus $\bar H$ is an upper approximation of the true minimum Hamiltonian.

This is the basis for the PA-bundle error gap used in the adaptive loop:

$$
\begin{equation}
\bar H(p,x,t)-H(p,x,t).
\end{equation}
$$

In practice, the adaptive code often compares $\bar H$ with the candidate
Hamiltonian returned by `compute_H`, which may include box corners, oracle
controls, scalar bounded minimizers, and bundle controls.

---

## 3) Controls are stored, not affine coefficients

The implementation does not store affine coefficients explicitly. It stores only
controls:

```python
self.controls: List[np.ndarray] = []
```

When the bundle is evaluated at a particular $(p,x,t)$, the affine quantities
are computed on the fly:

$$
\begin{equation}
f_i
=
f(x,u^{(i)},t),
\qquad
d_i
=
\ell(x,u^{(i)},t),
\end{equation}
$$

so that

$$
\begin{equation}
\phi_i(p;x,t)
=
p^\top f_i+d_i.
\end{equation}
$$

This design is important: the same stored control can define different affine
functions at different states and times, because both $f(x,u,t)$ and
$\ell(x,u,t)$ depend on $(x,t)$.

---

## 4) Class structure

The class has four methods:

```python
class PABundle:
    def __init__(self):
        ...

    def num_planes(self) -> int:
        ...

    def add_control(self, u: np.ndarray, tol: float = 1e-8) -> None:
        ...

    def evaluate(
        self,
        problem,
        p: np.ndarray,
        x: np.ndarray,
        t: float,
        dt: float | None = None,
        *,
        restricted: bool = True,
        fallback_unrestricted: bool = True,
    ) -> tuple:
        ...
```

The word "plane" refers to the affine function of $p$ induced by a stored
control at the current $(x,t)$. Since each stored control produces one affine
plane, the number of planes equals the number of stored controls.

---

## 5) Initialization

The constructor is

```python
def __init__(self):
    self.controls: List[np.ndarray] = []
```

So a new bundle starts empty.

An empty bundle cannot be evaluated. If `evaluate` is called with no stored
controls, the code raises

```python
RuntimeError("PABundle has no control candidates to evaluate.")
```

Therefore, the surrounding algorithm must seed the bundle before using it in a
Hamiltonian evaluation or in the smoothed Hamiltonian.

In the adaptive solver, the initial bundle is created in `adaptivity.py`. When
control bounds are known, the current code seeds it with

$$
\begin{equation}
u_{\mathrm{mid}}
=
\frac{1}{2}(u_{\min}+u_{\max}),
\qquad
u_{\min},
\qquad
u_{\max}.
\end{equation}
$$

If bounds are absent but the control dimension is known, it adds the zero
control.

---

## 6) `num_planes`

The method

```python
def num_planes(self) -> int:
    return len(self.controls)
```

returns the number of stored controls, equivalently the number of affine planes
available for evaluating the PA surrogate.

Mathematically,

$$
\begin{equation}
\texttt{num\_planes()}
=
|U_{\mathrm{bundle}}|
=
M.
\end{equation}
$$

This method is used by the adaptive algorithm for logging and for deciding how
many new support controls to add during PA enrichment.

---

## 7) `add_control`

The method

```python
def add_control(self, u: np.ndarray, tol: float = 1e-8) -> None:
    ...
```

adds a control to the bundle if it is not already present up to a Euclidean
tolerance.

The current implementation is:

```python
u = np.asarray(u, dtype=float)

for v in self.controls:
    if np.linalg.norm(u - v) < tol:
        return

self.controls.append(u)
```

---

### 7.1 Conversion to floating-point NumPy array

The first step is

```python
u = np.asarray(u, dtype=float)
```

This ensures that stored controls are NumPy arrays with floating-point dtype.

If the input is a Python list, tuple, or integer array, it is converted before
being stored.

---

### 7.2 Duplicate check

Before appending the new control, the method checks whether it is close to an
existing stored control.

A candidate $u$ is considered a duplicate of an existing control $v$ if

$$
\begin{equation}
\|u-v\| < \mathrm{tol}.
\end{equation}
$$

The default tolerance is

$$
\begin{equation}
\mathrm{tol}=10^{-8}.
\end{equation}
$$

If such a stored control exists, the method returns immediately and does not add
anything.

This means that `add_control` is idempotent up to tolerance: calling it many
times with the same control will not keep growing the bundle.

---

### 7.3 Append if new

If no near-duplicate is found, the control is appended:

```python
self.controls.append(u)
```

Thus the bundle grows by one plane.

The method does not return the inserted index. To check whether insertion
happened, callers compare the bundle size before and after insertion:

```python
before = bundle.num_planes()
bundle.add_control(u)
added = bundle.num_planes() > before
```

This pattern is used in the adaptive loop.

---

### 7.4 What `add_control` does not do

The current implementation of `add_control` is deliberately minimal.

It does **not**:

1. project the control to bounds;
2. check whether the control is admissible;
3. check local feasibility at a state and time;
4. enforce a maximum bundle size;
5. replace old controls;
6. rank controls by usefulness.

Therefore, callers are responsible for projecting or validating controls before
calling `add_control` when that is needed.

For example, `compute_H` projects candidate controls before evaluation when
bounds are known, and `adaptivity.py` obtains enrichment controls from
`compute_H`, which already uses the problem's feasibility logic.

---

## 8) `evaluate`

The method

```python
def evaluate(
    self,
    problem,
    p: np.ndarray,
    x: np.ndarray,
    t: float,
    dt: float | None = None,
    *,
    restricted: bool = True,
    fallback_unrestricted: bool = True,
) -> tuple:
    ...
```

evaluates the PA-bundle surrogate Hamiltonian at a single point $(p,x,t)$.

It returns

```python
best_val, best_idx
```

where `best_val` is the minimum Hamiltonian value over accepted bundle controls,
and `best_idx` is the index of the active stored control.

---

### 8.1 Mathematical target

Ignoring feasibility filters for a moment, the method computes

$$
\begin{equation}
\bar H(p,x,t)
=
\min_{u^{(i)}\in U_{\mathrm{bundle}}}
\left\{
p^\top f(x,u^{(i)},t)
+
\ell(x,u^{(i)},t)
\right\}.
\end{equation}
$$

The returned index is

$$
\begin{equation}
i^\star
\in
\arg\min_i
\left\{
p^\top f(x,u^{(i)},t)
+
\ell(x,u^{(i)},t)
\right\}.
\end{equation}
$$

Thus,

$$
\begin{equation}
\texttt{best\_val}
=
\bar H(p,x,t),
\qquad
\texttt{best\_idx}
=
i^\star.
\end{equation}
$$

When feasibility filtering is active, the minimization is restricted to the
stored controls accepted by the local feasibility test.

---

### 8.2 Empty-bundle check

The first branch is

```python
if not self.controls:
    raise RuntimeError("PABundle has no control candidates to evaluate.")
```

An empty bundle cannot define a PA minimum. The caller must ensure that at least
one control has been inserted before calling `evaluate`.

---

### 8.3 Initialization of the search

The method initializes:

```python
best_val = np.inf
best_idx = -1
tried_any = False
```

The variable `best_val` stores the smallest Hamiltonian integrand found so far.

The variable `best_idx` stores the index of the control that achieved it.

The variable `tried_any` records whether at least one stored control passed the
restricted feasibility filter and was evaluated in the main loop.

---

### 8.4 Restricted evaluation loop

The main loop is:

```python
for i, u in enumerate(self.controls):
    if restricted and not problem.local_control_feasible(
        x,
        u,
        t,
        restricted=True,
        dt=dt,
    ):
        continue

    tried_any = True
    val = float(np.dot(p, problem.f(x, u, t)) + problem.l(x, u, t))

    if val < best_val:
        best_val = val
        best_idx = i
```

If `restricted=True`, each stored control is checked with

```python
problem.local_control_feasible(x, u, t, restricted=True, dt=dt)
```

The optional `dt` is passed through because some problems use local
step-size-dependent feasibility.

If the control fails this check, it is skipped.

For every accepted control, the code computes

$$
\begin{equation}
\mathcal{H}(p,x,u^{(i)},t)
=
p^\top f(x,u^{(i)},t)
+
\ell(x,u^{(i)},t).
\end{equation}
$$

The minimum value and active index are updated whenever a smaller value is
found.

---

### 8.5 Meaning of `restricted`

When `restricted=True`, the bundle surrogate is effectively

$$
\begin{equation}
\bar H_{\mathrm{loc}}(p,x,t;\Delta t)
=
\min_{\substack{
u^{(i)}\in U_{\mathrm{bundle}}\\
u^{(i)}\ \mathrm{locally\ feasible}
}}
\left\{
p^\top f(x,u^{(i)},t)
+
\ell(x,u^{(i)},t)
\right\}.
\end{equation}
$$

When `restricted=False`, the local feasibility check is skipped in the main
loop, and all stored controls are evaluated.

---

### 8.6 Unrestricted fallback

After the main loop, it is possible that no control was evaluated because all
stored controls failed the restricted feasibility check.

This case is detected by

```python
if fallback_unrestricted and not tried_any:
    ...
```

If `fallback_unrestricted=True`, the method performs a second pass over all
stored controls without the feasibility filter:

```python
for i, u in enumerate(self.controls):
    val = float(np.dot(p, problem.f(x, u, t)) + problem.l(x, u, t))
    if val < best_val:
        best_val = val
        best_idx = i
```

This fallback returns a bundle minimum even when the restricted feasible subset
is empty.

If `fallback_unrestricted=False`, this second pass is skipped.

This option is important in the adaptive PA-error computation. In some places,
the code wants the restricted PA value to fail clearly if no locally feasible
bundle control exists, so it calls

```python
bundle.evaluate(..., restricted=True, fallback_unrestricted=False)
```

---

### 8.7 Failure after evaluation

If no active control is found, the method raises

```python
RuntimeError("PABundle has no locally feasible control candidates to evaluate.")
```

This can happen when:

1. the bundle is non-empty;
2. `restricted=True`;
3. all stored controls fail `problem.local_control_feasible`;
4. `fallback_unrestricted=False`.

It can also happen if some unexpected numerical issue prevents any valid value
from being selected.

---

### 8.8 Return value

If evaluation succeeds, the method returns

```python
return best_val, best_idx
```

where

$$
\begin{equation}
\texttt{best\_val}
=
p^\top f(x,u^{(i^\star)},t)
+
\ell(x,u^{(i^\star)},t),
\end{equation}
$$

and

$$
\begin{equation}
\texttt{best\_idx}
=
i^\star.
\end{equation}
$$

The active control itself can be recovered as

```python
u_active = bundle.controls[best_idx]
```

The returned index is used by the adaptive loop for diagnostics, such as
recording which PA plane is active at each mesh node.

---

## 9) Interaction with `smoothing.py`

The smoothing routine `eval_H_smooth` uses

```python
bundle.controls
```

directly to build the smoothed PA Hamiltonian.

If

$$
\begin{equation}
U_{\mathrm{bundle}}
=
\{u^{(1)},\dots,u^{(M)}\},
\end{equation}
$$

then `smoothing.py` forms the plane values

$$
\begin{equation}
g_i(p,x,t)
=
p^\top f(x,u^{(i)},t)
+
\ell(x,u^{(i)},t),
\end{equation}
$$

and computes the soft minimum

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
\right).
\end{equation}
$$

Unlike `PABundle.evaluate`, the default smoothing routine does **not** call
`problem.local_control_feasible` for each stored control. It assumes that the
bundle has already been built consistently by the surrounding solver.

Therefore, the same bundle can be used in two related but distinct ways:

1. `bundle.evaluate(...)` computes a hard minimum over stored controls, with
   optional local feasibility filtering;
2. `eval_H_smooth(...)` computes a soft minimum over stored controls, without
   re-filtering them.

---

## 10) Interaction with `hamiltonian.py`

The routine `compute_H` uses `bundle.controls` as one part of a larger candidate
set.

The candidate set used by `compute_H` may contain:

1. an oracle control from `problem.u_star(...)`;
2. control-box corners;
3. a scalar bounded minimizer when the control is scalar and bounded;
4. the controls already stored in the PA bundle.

Schematically,

$$
\begin{equation}
\mathcal{U}_{cand}
=
\mathcal{U}_{oracle}
\cup
\mathcal{U}_{corners}
\cup
\mathcal{U}_{scalar}
\cup
U_{\mathrm{bundle}}.
\end{equation}
$$

Thus `bundle.evaluate(...)` computes the bundle-only surrogate

$$
\begin{equation}
\bar H(p,x,t)
=
\min_{u\in U_{\mathrm{bundle}}}
\mathcal{H}(p,x,u,t),
\end{equation}
$$

whereas `compute_H(...)` computes a richer candidate Hamiltonian

$$
\begin{equation}
H_{cand}(p,x,t)
=
\min_{u\in\mathcal{U}_{cand}}
\mathcal{H}(p,x,u,t).
\end{equation}
$$

When $U_{\mathrm{bundle}}\subseteq\mathcal{U}_{cand}$ and compatible feasibility
filters are used,

$$
\begin{equation}
H_{cand}(p,x,t)
\le
\bar H(p,x,t).
\end{equation}
$$

The adaptive loop uses this gap to decide when the PA bundle needs enrichment.

---

## 11) Interaction with `adaptivity.py`

The adaptive loop is responsible for creating and enriching the bundle.

At startup, `adaptivity.py` creates

```python
bundle = PABundle()
```

and seeds it with basic controls. If bounds are available, it adds

$$
\begin{equation}
u_{\mathrm{mid}}
=
\frac{1}{2}(u_{\min}+u_{\max}),
\qquad
u_{\min},
\qquad
u_{\max}.
\end{equation}
$$

If bounds are absent but the control dimension is known, it adds the zero
control.

During the first outer iteration, the adaptive loop may call

```python
bootstrap_bundle_from_trajectory(...)
```

to add controls that are active or approximately active along the first coarse
trajectory.

Later, when the PA indicator is too large, the adaptive loop selects one or
more enrichment nodes. At those nodes, it uses `compute_H` to find candidate
controls and then calls

```python
bundle.add_control(candidate_u)
```

If the control is not a duplicate, it becomes a new PA plane.

Thus `PABundle` itself does not decide which controls are important. It only
stores the controls selected by the adaptive algorithm.

---

## 12) Restricted PA evaluation in adaptivity

The adaptive loop sometimes evaluates the PA surrogate with

```python
bundle.evaluate(
    problem,
    P[i],
    X[i],
    t_nodes[i],
    dt=dt_i,
    restricted=True,
    fallback_unrestricted=False,
)
```

The option `fallback_unrestricted=False` is important. It means:

- if no stored bundle control is locally feasible at the point;
- and restricted evaluation is requested;

then `evaluate` should raise an error rather than silently returning an
unrestricted PA value.

This is useful when computing PA gaps, because the code wants the restricted
bundle value and the restricted candidate Hamiltonian to be consistent.

Other calls use the default `fallback_unrestricted=True`, allowing a more robust
diagnostic value even if all stored controls fail the local feasibility filter.

---

## 13) Debugging checklist

When the PA bundle behaves unexpectedly, check the following points.

---

### 13.1 Bundle non-emptiness

Check

```python
bundle.num_planes() > 0
```

An empty bundle cannot be evaluated and cannot define a smoothed Hamiltonian.

---

### 13.2 Duplicate controls

If a call to

```python
bundle.add_control(u)
```

does not increase `bundle.num_planes()`, the new control was probably within
the duplicate tolerance of an existing control:

$$
\begin{equation}
\|u-v\| < 10^{-8}.
\end{equation}
$$

This is normal behavior.

---

### 13.3 Control shape

Each stored control should be a one-dimensional NumPy array:

```python
u.shape == (m,)
```

Avoid column vectors such as `(m, 1)`, because they can lead to unexpected NumPy
broadcasting in norms, dot products, or comparisons.

---

### 13.4 Bounds and projection

`add_control` does not project controls. If bounds are required, projection must
happen before insertion or during candidate generation.

If controls appear outside the admissible box, check the caller that inserted
them.

---

### 13.5 Local feasibility

If `bundle.evaluate(..., restricted=True)` fails with

```python
RuntimeError("PABundle has no locally feasible control candidates to evaluate.")
```

then all stored controls failed

```python
problem.local_control_feasible(...)
```

at the current $(x,t,\Delta t)$.

Possible causes include:

- the bundle controls are not appropriate near the current state;
- the local feasibility test is too strict;
- the time step `dt` is too large for step-feasibility constraints;
- the bundle needs to be refreshed after mesh refinement.

---

### 13.6 Smoothing includes all stored controls

Remember that `eval_H_smooth` uses all stored controls directly and does not
re-check local feasibility. If a control should not participate in smoothing,
the issue must be addressed when constructing or refreshing the bundle.

---

## 14) Summary

`PABundle` is a passive memory of control candidates.

It stores

$$
\begin{equation}
U_{\mathrm{bundle}}
=
\{u^{(1)},\dots,u^{(M)}\},
\end{equation}
$$

and evaluates

$$
\begin{equation}
\bar H(p,x,t)
=
\min_{u\in U_{\mathrm{bundle}}}
\left\{
p^\top f(x,u,t)+\ell(x,u,t)
\right\}.
\end{equation}
$$

The current implementation provides only:

1. storage of controls;
2. duplicate-safe insertion;
3. hard-min evaluation with optional local feasibility filtering;
4. active-plane index reporting.

All higher-level decisions — how to seed the bundle, when to add controls, and
which controls to add — are handled by the adaptive solver.