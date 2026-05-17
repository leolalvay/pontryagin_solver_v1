# `core/hamiltonian.py` — Reference Hamiltonian evaluation with `compute_H`

This module implements the routine

```python
compute_H(...)
```

which evaluates a pointwise Hamiltonian minimization and returns both the
minimal value and a corresponding minimizing control.

The routine is used in several places:

1. to reconstruct an approximate control trajectory from a solved state-costate
   trajectory $(X,P)$;
2. to compute the PA-bundle indicator in the adaptive outer loop through the
   gap $\bar H-H$;
3. to select new PA-bundle support controls during enrichment;
4. to perform feasibility-sensitive control checks in constrained examples.

The implementation is not only a mathematical Hamiltonian formula. It is also a
candidate-generation and filtering routine: it builds a finite or enriched set
of controls, filters them through the problem's local feasibility logic, and
selects the control with the smallest Hamiltonian integrand.

---

## 1) Mathematical convention

This repository uses the **minimum Hamiltonian convention**. For a fixed
costate $p$, state $x$, and time $t$, the Hamiltonian is

$$
\begin{equation}
H(p,x,t)
=
\min_{u\in A}
\left\{
p^\top f(x,u,t)
+
\ell(x,u,t)
\right\}.
\end{equation}
$$

Here:

- $x\in\mathbb{R}^n$ is the state;
- $p\in\mathbb{R}^n$ is the costate;
- $u\in\mathbb{R}^m$ is the control;
- $A$ is the admissible control set, usually a box;
- $f$ is the dynamics;
- $\ell$ is the running cost.

If the admissible set is represented by componentwise bounds, then

$$
\begin{equation}
A
=
[u_{\min},u_{\max}]
=
\left\{
u\in\mathbb{R}^m
:
(u_{\min})_j
\le
u_j
\le
(u_{\max})_j,
\quad
j=1,\dots,m
\right\}.
\end{equation}
$$

Many PMP references use a maximum convention. This codebase instead uses a
minimum convention because the surrounding optimal control problem is written
as a minimization problem.

For a fixed control $u$, define the Hamiltonian integrand

$$
\begin{equation}
\mathcal{H}(p,x,u,t)
=
p^\top f(x,u,t)
+
\ell(x,u,t).
\end{equation}
$$

Then the pointwise Hamiltonian minimization is

$$
\begin{equation}
H(p,x,t)
=
\min_{u\in A}
\mathcal{H}(p,x,u,t).
\end{equation}
$$

---

## 2) Restricted and locally feasible Hamiltonian

For constrained problems, not every control in the box is necessarily acceptable
at a given state and time. The code can compute a restricted Hamiltonian by
filtering controls through the problem's local feasibility logic.

Conceptually, if the state is constrained to a feasible set $K$, one may define

$$
\begin{equation}
H_K(p,x,t)
=
\min_{\substack{u\in A\\ f(x,u,t)\in T_K(x)}}
\left\{
p^\top f(x,u,t)
+
\ell(x,u,t)
\right\},
\end{equation}
$$

where $T_K(x)$ is the tangent cone of $K$ at $x$.

In the current code, this restriction is handled through

```python
problem.local_control_feasible(
    x,
    u,
    t,
    restricted=restricted,
    dt=dt,
)
```

This method may include several checks:

1. ordinary control admissibility;
2. tangent-cone or viability checks when `restricted=True`;
3. step-size-dependent feasibility checks when the problem uses local
   one-step feasibility logic.

Therefore, the implemented restricted Hamiltonian is better interpreted as

$$
\begin{equation}
H_{\mathrm{loc}}(p,x,t;\Delta t)
=
\min_{u\in A_{\mathrm{loc}}(x,t,\Delta t)}
\left\{
p^\top f(x,u,t)
+
\ell(x,u,t)
\right\},
\end{equation}
$$

where $A_{\mathrm{loc}}(x,t,\Delta t)$ denotes the set of controls accepted by
`problem.local_control_feasible`.

If `restricted=False`, the local feasibility test is called with
`restricted=False`, so tangent-cone restrictions are disabled. Basic projection
or admissibility behavior may still be problem-dependent.

---

## 3) Current function signature

The current signature is

```python
def compute_H(
    problem,
    p: np.ndarray,
    x: np.ndarray,
    t: float,
    candidate_controls: List[np.ndarray],
    restricted: bool = False,
    use_oracle: bool = False,
    dt: Optional[float] = None,
) -> Tuple[float, np.ndarray]:
    ...
```

---

### 3.1 Inputs

| Argument | Meaning |
|---|---|
| `problem` | The `OCPProblem` instance providing dynamics, cost, bounds, projections, local feasibility checks, and optional oracle controls. |
| `p` | Costate vector at the current node, shape `(n,)`. |
| `x` | State vector at the current node, shape `(n,)`. |
| `t` | Current time. |
| `candidate_controls` | Additional controls to include in the candidate set, usually `bundle.controls`. |
| `restricted` | If `True`, controls are filtered through the restricted local feasibility logic. |
| `use_oracle` | If `True`, the routine asks the problem for an oracle control through `problem.u_star(...)` and includes it as a candidate if feasible. |
| `dt` | Optional local time step. Passed to oracle and feasibility checks for problems whose local admissibility depends on $\Delta t$. |

The local step `dt` is important for examples where feasibility is not purely
pointwise in $(x,u,t)$ but depends on whether a control is acceptable over a
time step of length $\Delta t$.

---

### 3.2 Outputs

The routine returns

```python
best_val, best_control
```

where

$$
\begin{equation}
\texttt{best\_val}
=
\min_{u\in\mathcal{U}_{cand}}
\mathcal{H}(p,x,u,t),
\end{equation}
$$

and

$$
\begin{equation}
\texttt{best\_control}
\in
\arg\min_{u\in\mathcal{U}_{cand}}
\mathcal{H}(p,x,u,t).
\end{equation}
$$

Here $\mathcal{U}_{cand}$ is the candidate set constructed by the routine after
oracle insertion, box-corner generation, scalar bounded minimization, supplied
candidate insertion, projection, duplicate removal, and feasibility filtering.

If no admissible control is found, the routine returns

```python
float("inf"), None
```

This can happen, for example, when `restricted=True` and all candidates violate
the local feasibility logic.

---

## 4) Pointwise controls versus a control trajectory

It is important to distinguish a **pointwise Hamiltonian minimizer** from a
control trajectory.

For a single triple $(p,x,t)$, `compute_H` returns one control

$$
\begin{equation}
u^\star(p,x,t)
\in
\arg\min_u
\mathcal{H}(p,x,u,t).
\end{equation}
$$

A discrete control trajectory is obtained only by repeating this minimization at
each mesh node:

$$
\begin{equation}
U_i
=
u^\star(P_i,X_i,t_i),
\qquad
i=0,\dots,N.
\end{equation}
$$

Thus `compute_H` does not solve for an entire function $u(t)$ at once. It solves
a local minimization problem at a single state-costate-time point.

---

## 5) Implementation walkthrough

The routine `compute_H` can be understood as the following pipeline:

```python
candidates = []

optionally add oracle control
add control-box corners
if scalar bounded control:
    add bounded scalar minimizer
add supplied candidate controls
remove duplicates
filter by local feasibility
evaluate Hamiltonian integrand
return the best feasible candidate
```

Thus the function is not just a formula for $H$. It is a concrete numerical
procedure for constructing and searching a candidate set.

---

### 5.1 Start with an empty candidate list

The routine begins with

```python
candidates = []
bounds = problem.control_bounds_tuple()
```

The value `bounds` is either `None` or a pair

```python
u_min, u_max = bounds
```

representing the control box

$$
\begin{equation}
A
=
[u_{\min},u_{\max}].
\end{equation}
$$

If bounds are unavailable, the routine cannot generate box corners or run the
scalar bounded minimization. In that case, it relies on the oracle candidate, if
enabled, and on the supplied `candidate_controls`.

---

### 5.2 Optional oracle candidate

If

```python
use_oracle=True
```

and the problem object has a method `u_star`, the routine calls

```python
u_oracle, ok = problem.u_star(
    x,
    p,
    t,
    restricted=restricted,
    dt=dt,
)
```

The oracle is expected to return:

- `u_oracle`: a candidate minimizing control, or `None`;
- `ok`: a boolean indicating whether the oracle control is acceptable under the
  requested restriction.

If the oracle returns a control and either the problem is unrestricted or
`ok=True`, the control is added to the candidate list.

If bounds are available, the oracle control is projected back to the box before
being stored:

$$
\begin{equation}
u_{\mathrm{oracle}}
\leftarrow
\Pi_A(u_{\mathrm{oracle}}).
\end{equation}
$$

Here $\Pi_A$ is componentwise projection onto the control box:

$$
\begin{equation}
(\Pi_A(u))_j
=
\min
\left\{
\max\left\{u_j,(u_{\min})_j\right\},
(u_{\max})_j
\right\}.
\end{equation}
$$

**Important implementation detail.** The oracle does not cause an immediate
return. It only contributes one candidate to the candidate list. The routine
still adds box corners, a scalar bounded minimizer when available, and supplied
candidate controls. The final output is the best candidate after all candidates
are collected and evaluated.

---

### 5.3 Box-corner candidates

If bounds are available, the routine adds all corners of the box

$$
\begin{equation}
A
=
[u_{\min},u_{\max}]
\subset
\mathbb{R}^m.
\end{equation}
$$

For every component $j=1,\dots,m$, a corner chooses either the lower or upper
bound:

$$
\begin{equation}
u_j
\in
\left\{
(u_{\min})_j,\,
(u_{\max})_j
\right\}.
\end{equation}
$$

Therefore, for dimension $m$, there are $2^m$ corners.

In code, these are generated by

```python
for combo in product([0, 1], repeat=m):
    u = np.where(np.array(combo) == 0, u_min, u_max)
    candidates.append(u)
```

For example, if $m=2$ and

$$
\begin{equation}
u_{\min}=(a,b),
\qquad
u_{\max}=(c,d),
\end{equation}
$$

then

$$
\begin{equation}
A=[a,c]\times[b,d],
\end{equation}
$$

and the four corners are

$$
\begin{equation}
(a,b),
\qquad
(c,b),
\qquad
(a,d),
\qquad
(c,d).
\end{equation}
$$

This corner enumeration is useful for bang-bang problems, where the true
Hamiltonian minimizer often lies on the boundary of the control box.

The cost of this step grows like

$$
\begin{equation}
2^m.
\end{equation}
$$

Therefore, this routine is intended for small control dimension $m$ unless the
candidate set is customized elsewhere.

---

### 5.4 Scalar bounded minimization

When bounds are available and the control is scalar, i.e. $m=1$, the routine
also performs a continuous bounded scalar minimization.

Let

$$
\begin{equation}
u_{\min}=a_{\mathrm{lo}},
\qquad
u_{\max}=a_{\mathrm{hi}}.
\end{equation}
$$

The scalar objective is

$$
\begin{equation}
\phi(a)
=
p^\top f(x,[a],t)
+
\ell(x,[a],t),
\qquad
a\in[a_{\mathrm{lo}},a_{\mathrm{hi}}].
\end{equation}
$$

In code:

```python
def obj(a: float) -> float:
    u = np.array([a], dtype=float)
    u = problem.project_control(u)
    if not problem.local_control_feasible(x, u, t, restricted=restricted, dt=dt):
        return 1.0e30
    return float(np.dot(p, problem.f(x, u, t)) + problem.l(x, u, t))
```

The large value `1.0e30` penalizes infeasible scalar controls inside the bounded
minimization.

The minimization is performed by

```python
res = minimize_scalar(
    obj,
    bounds=(a_lo, a_hi),
    method="bounded",
    options={"xatol": 1e-6, "maxiter": 80},
)
```

If the scalar minimization succeeds and returns a finite value, the resulting
control is added to the candidate list:

```python
candidates.append(np.array([float(res.x)], dtype=float))
```

This is an important difference from older versions of this documentation:
`compute_H` is not purely a finite corner-and-bundle minimization in the scalar
bounded case. It enriches the candidate set with a bounded one-dimensional
continuous minimizer.

---

### 5.5 Supplied candidate controls

After oracle, corners, and possible scalar minimization, the routine adds the
user-supplied `candidate_controls`.

These are usually the controls stored in the PA bundle:

```python
for u in candidate_controls:
    if bounds is not None:
        u = problem.project_control(u)
    candidates.append(u)
```

If bounds exist, every supplied candidate is projected to the admissible box
before being added:

$$
\begin{equation}
u
\leftarrow
\Pi_A(u).
\end{equation}
$$

This makes the routine robust to small numerical violations or to bundle entries
that were generated before a projection step.

---

### 5.6 Duplicate removal

Before evaluating candidates, the routine removes duplicates up to a small
Euclidean tolerance.

A candidate $u$ is treated as already present if there is an existing candidate
$v$ such that

$$
\begin{equation}
\|u-v\| < 10^{-10}.
\end{equation}
$$

Only the first representative is kept.

This step prevents repeated evaluations of the same control. It is especially
useful because the same control can enter through multiple paths, for example:

- as an oracle control;
- as a box corner;
- as a scalar minimizer;
- as a PA-bundle control.

---

### 5.7 Feasibility filtering and Hamiltonian evaluation

After duplicate removal, the routine loops through the unique candidate list.

For each control $u$, it first checks local feasibility:

```python
if not problem.local_control_feasible(
    x,
    u,
    t,
    restricted=restricted,
    dt=dt,
):
    continue
```

If the control is accepted, the Hamiltonian integrand is evaluated:

$$
\begin{equation}
\mathcal{H}(p,x,u,t)
=
p^\top f(x,u,t)
+
\ell(x,u,t).
\end{equation}
$$

The code stores the candidate with the smallest value:

$$
\begin{equation}
u^\star
\in
\arg\min_{u\in\mathcal{U}_{cand}^{feas}}
\mathcal{H}(p,x,u,t),
\end{equation}
$$

where $\mathcal{U}_{cand}^{feas}$ is the subset of candidates that passed
`problem.local_control_feasible`.

The corresponding value is

$$
\begin{equation}
H_{cand}(p,x,t)
=
\mathcal{H}(p,x,u^\star,t).
\end{equation}
$$

---

### 5.8 Fallback behavior when no feasible candidate is found

If no feasible candidate is found, the behavior depends on the value of
`restricted`.

If `restricted=True`, the routine returns immediately:

```python
return float("inf"), None
```

This means that, under the requested local restriction, the candidate set did
not contain any admissible control.

If `restricted=False`, the routine performs one final pass over the candidates
without calling `local_control_feasible`. It simply evaluates

$$
\begin{equation}
\mathcal{H}(p,x,u,t)
=
p^\top f(x,u,t)
+
\ell(x,u,t)
\end{equation}
$$

for each candidate and returns the best one.

If even this fallback pass cannot find a control, the routine returns

```python
float("inf"), None
```

This final case usually indicates that the candidate list was empty or that the
candidate data were not usable.

---

### 5.9 Final return

The function returns

```python
best_val, best_control
```

where `best_val` is the smallest Hamiltonian integrand value among accepted
candidates and `best_control` is one corresponding minimizer.

In exact mathematical notation, if the candidate feasible set is nonempty, the
return satisfies

$$
\begin{equation}
\texttt{best\_val}
=
\min_{u\in\mathcal{U}_{cand}^{feas}}
\left\{
p^\top f(x,u,t)+\ell(x,u,t)
\right\},
\end{equation}
$$

and

$$
\begin{equation}
\texttt{best\_control}
\in
\arg\min_{u\in\mathcal{U}_{cand}^{feas}}
\left\{
p^\top f(x,u,t)+\ell(x,u,t)
\right\}.
\end{equation}
$$

If no acceptable minimizer is found, the return is

```python
float("inf"), None
```

---


## 6) Relation with the PA-bundle surrogate

The PA bundle stores a finite set of controls

$$
\begin{equation}
U_{\mathrm{bundle}}
=
\{u_1,\dots,u_M\}.
\end{equation}
$$

The PA-bundle surrogate Hamiltonian is

$$
\begin{equation}
\bar H(p,x,t)
=
\min_{u\in U_{\mathrm{bundle}}}
\left\{
p^\top f(x,u,t)
+
\ell(x,u,t)
\right\}.
\end{equation}
$$

By contrast, `compute_H` builds an enlarged candidate set. In the current
implementation, this set may include:

1. an oracle control, if `use_oracle=True`;
2. control-box corners, if bounds are available;
3. a scalar bounded minimizer, if the control is scalar and bounded;
4. the PA-bundle controls passed through `candidate_controls`.

Thus the effective candidate set has the form

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
U_{\mathrm{bundle}},
\end{equation}
$$

with some of these sets possibly empty.

The returned value from `compute_H` is therefore

$$
\begin{equation}
H_{cand}(p,x,t)
=
\min_{u\in\mathcal{U}_{cand}^{feas}}
\left\{
p^\top f(x,u,t)
+
\ell(x,u,t)
\right\},
\end{equation}
$$

where $\mathcal{U}_{cand}^{feas}$ is the subset of candidates accepted by the
local feasibility logic.

Since $U_{\mathrm{bundle}}\subseteq \mathcal{U}_{cand}$ whenever the bundle
controls are passed in, the candidate Hamiltonian satisfies

$$
\begin{equation}
H_{cand}(p,x,t)
\le
\bar H(p,x,t)
\end{equation}
$$

provided both evaluations use compatible feasibility restrictions.

This is why the adaptive loop can use the gap

$$
\begin{equation}
\bar H(p,x,t)
-
H_{cand}(p,x,t)
\end{equation}
$$

as a PA-bundle approximation indicator.

---

### 6.1 PA indicator in the adaptive loop

In `adaptivity.py`, the PA indicator is computed from nodal gaps. At node
$t_i$, the code forms

$$
\begin{equation}
e_i^{PA}
=
\bar H(P_i,X_i,t_i)
-
H_{cand}(P_i,X_i,t_i).
\end{equation}
$$

The local interval contribution is approximated by the trapezoidal rule:

$$
\begin{equation}
\eta_i^{PA}
=
\frac{1}{2}
\left(
e_i^{PA}
+
e_{i+1}^{PA}
\right)
\Delta t_i.
\end{equation}
$$

The global PA indicator is

$$
\begin{equation}
\eta_{PA}
=
\sum_{i=0}^{N-1}
\eta_i^{PA}.
\end{equation}
$$

A large gap indicates that the current bundle is missing controls that improve
the Hamiltonian minimization near the current trajectory. The adaptive loop then
uses `compute_H` again to obtain candidate controls for PA enrichment.

---

## 7) Mathematical accuracy of `compute_H`

It is important to distinguish three Hamiltonians:

1. the true Hamiltonian $H$;
2. the PA-bundle surrogate $\bar H$;
3. the candidate Hamiltonian $H_{cand}$ returned by `compute_H`.

The true Hamiltonian is

$$
\begin{equation}
H(p,x,t)
=
\min_{u\in A_{\mathrm{true}}}
\mathcal{H}(p,x,u,t).
\end{equation}
$$

The PA surrogate is

$$
\begin{equation}
\bar H(p,x,t)
=
\min_{u\in U_{\mathrm{bundle}}}
\mathcal{H}(p,x,u,t).
\end{equation}
$$

The value returned by `compute_H` is

$$
\begin{equation}
H_{cand}(p,x,t)
=
\min_{u\in\mathcal{U}_{cand}^{feas}}
\mathcal{H}(p,x,u,t).
\end{equation}
$$

If

$$
\begin{equation}
\mathcal{U}_{cand}^{feas}
\subseteq
A_{\mathrm{true}},
\end{equation}
$$

then

$$
\begin{equation}
H(p,x,t)
\le
H_{cand}(p,x,t).
\end{equation}
$$

That is, the candidate Hamiltonian is generally an upper approximation of the
true minimum. It becomes exact only if the true minimizer is included in the
candidate feasible set.

In bang-bang problems, the true minimizer often occurs at a corner of the
control box, so box-corner enumeration can be exact. In scalar smooth problems,
the bounded scalar minimization improves the chance of finding an interior
minimizer. In higher-dimensional smooth problems, accuracy depends more heavily
on the PA-bundle controls and any oracle controls supplied by the problem.

Thus `compute_H` should be interpreted as a **reference-like numerical
Hamiltonian evaluation**, not always as the exact analytic Hamiltonian.

---

## 8) Example interpretations

This section explains how the same routine behaves in common cases.

---

### 8.1 Bang-bang scalar control

Suppose $m=1$ and the Hamiltonian integrand is affine in $u$:

$$
\begin{equation}
\mathcal{H}(p,x,u,t)
=
a(p,x,t)u
+
b(p,x,t),
\qquad
u\in[u_{\min},u_{\max}].
\end{equation}
$$

Then the exact minimizer over the interval is

$$
\begin{equation}
u^\star
=
\begin{cases}
u_{\min}, & a(p,x,t)>0,\\
u_{\max}, & a(p,x,t)<0,\\
\text{any }u\in[u_{\min},u_{\max}], & a(p,x,t)=0.
\end{cases}
\end{equation}
$$

In this case, checking the two box endpoints is enough to recover an exact
minimizer, except at a switching point where many controls may tie.

The scalar bounded minimizer may also return an endpoint or a nearly equivalent
point, but the corner candidates already contain the exact bang-bang controls.

---

### 8.2 Scalar smooth bounded control

For scalar smooth problems, the minimizer may be an interior point. For example,

$$
\begin{equation}
\mathcal{H}(p,x,u,t)
=
a(p,x,t)u
+
r u^2,
\qquad
r>0,
\qquad
u\in[u_{\min},u_{\max}].
\end{equation}
$$

The unconstrained stationary point satisfies

$$
\begin{equation}
a(p,x,t)+2ru=0,
\end{equation}
$$

so

$$
\begin{equation}
u_{\mathrm{uncon}}
=
-\frac{a(p,x,t)}{2r}.
\end{equation}
$$

The bounded minimizer is the projection

$$
\begin{equation}
u^\star
=
\Pi_{[u_{\min},u_{\max}]}
\left(
-\frac{a(p,x,t)}{2r}
\right).
\end{equation}
$$

The current implementation can capture such an interior minimizer because, when
$m=1$, it runs a bounded scalar minimization and adds the resulting control to
the candidate set.

---

### 8.3 LQR-type control cost

For a linear-quadratic integrand of the form

$$
\begin{equation}
f(x,u,t)
=
Ax+Bu,
\qquad
\ell(x,u,t)
=
x^\top Qx
+
u^\top Ru,
\end{equation}
$$

the Hamiltonian integrand is

$$
\begin{equation}
\mathcal{H}(p,x,u,t)
=
p^\top(Ax+Bu)
+
x^\top Qx
+
u^\top Ru.
\end{equation}
$$

If the control were unconstrained and $R$ were positive definite, the analytic
stationarity condition would be

$$
\begin{equation}
B^\top p
+
2Ru
=
0,
\end{equation}
$$

hence

$$
\begin{equation}
u^\star
=
-\frac{1}{2}R^{-1}B^\top p.
\end{equation}
$$

In the current bounded-control implementation, the minimization is performed
over the bounded admissible set. If $m=1$, the scalar bounded minimizer can
recover the projected version of this interior optimum. For $m>1$, the routine
does not perform a multidimensional continuous minimization; it relies on box
corners, oracle controls, and supplied candidates.

---

### 8.4 Restricted control near a state constraint

Suppose the state is constrained to a set $K$, and the current point lies on the
boundary. Then a control may be inside the control box but still be locally
infeasible because it points out of the state constraint set.

Conceptually, the restricted minimization is

$$
\begin{equation}
\min_{\substack{u\in A\\ f(x,u,t)\in T_K(x)}}
\mathcal{H}(p,x,u,t).
\end{equation}
$$

In the code, this restriction is enforced through
`problem.local_control_feasible(...)`. Therefore, a candidate control is skipped
if it fails the local feasibility test, even if it gives a smaller Hamiltonian
value.

This distinction is important in constrained examples: the returned control is
not merely the cheapest control in the box, but the cheapest locally feasible
control among the constructed candidates.

---

## 9) Debugging checklist

When `compute_H` returns unexpected controls, `None`, or `inf`, check the
following points.

---

### 9.1 Candidate set non-emptiness

At least one candidate must be generated. Candidates can come from:

1. `problem.u_star(...)` when `use_oracle=True`;
2. control-box corners when bounds are available;
3. scalar bounded minimization when $m=1$ and bounds are available;
4. supplied `candidate_controls`.

If bounds are absent, `use_oracle=False`, and `candidate_controls` is empty, the
routine has no controls to evaluate.

---

### 9.2 Shapes

Check that

$$
\begin{equation}
p\in\mathbb{R}^n,
\qquad
x\in\mathbb{R}^n,
\qquad
u\in\mathbb{R}^m.
\end{equation}
$$

In NumPy terms, use one-dimensional arrays:

```python
p.shape == (n,)
x.shape == (n,)
u.shape == (m,)
```

Avoid column-shaped controls such as `(m, 1)`, because broadcasting can lead to
incorrect vector operations or duplicate checks.

---

### 9.3 Bounds and projection

If bounds exist, oracle and supplied controls are projected using
`problem.project_control`.

If a control appears to be outside bounds before entering `compute_H`, remember
that it may be clipped before evaluation.

---

### 9.4 Local feasibility filtering

If `restricted=True`, many candidates may be skipped by

```python
problem.local_control_feasible(...)
```

Check whether the problem's tangent, admissibility, or step-feasibility logic is
too strict.

Also check the local time step `dt`. For some problems, changing `dt` changes
which controls are locally feasible.

---

### 9.5 Oracle behavior

If `use_oracle=True`, verify that `problem.u_star(...)` returns both a control
and a meaningful feasibility flag `ok`.

If the oracle returns `ok=False` in restricted mode, the oracle candidate is not
added.

---

### 9.6 Scalar minimization behavior

For scalar bounded controls, the `minimize_scalar` candidate is only added if
the optimization succeeds and returns a finite value.

If the scalar objective is discontinuous, flat, or returns `1.0e30` for most
points because of feasibility filtering, the scalar minimizer may not add a
useful candidate.

---

### 9.7 Non-finite dynamics or costs

If `problem.f` or `problem.l` returns `NaN` or `inf`, Hamiltonian comparison is
not reliable. Check the model functions and the current state/control values.

---

### 9.8 High-dimensional controls

The number of box corners is $2^m$. For large $m$, corner enumeration can become
expensive. Also, for smooth multidimensional problems, corners alone may be a
poor approximation of an interior minimizer.

In that case, consider using problem-specific oracle controls or adding good
candidate controls through the PA bundle.

---

## 10) Where `compute_H` is used

The routine is used by several parts of the repository:

1. **Adaptive PA indicator.** In `adaptivity.py`, it computes the reference-like
   Hamiltonian $H_{cand}$ used in the gap $\bar H-H_{cand}$.

2. **PA enrichment.** In `adaptivity.py`, it provides the candidate controls
   that may be added to the PA bundle when the action is `"add_plane"`.

3. **Control reconstruction.** Experiment scripts call `compute_H` at each node
   to reconstruct a discrete control trajectory from the computed state-costate
   pair $(X,P)$.

4. **Feasibility diagnostics.** For constrained or step-feasible problems,
   `adaptivity.py` calls `compute_H` with different local step sizes to detect
   feasibility sensitivity.

Thus `compute_H` is the main bridge between the abstract Hamiltonian
minimization and the concrete candidate controls used by the adaptive solver.

---

## 6) Code workflow and mathematical interpretation

This section connects the mathematical Hamiltonian minimization with the actual
implementation of `compute_H`.

The routine follows this workflow:

```python
candidates = []
bounds = problem.control_bounds_tuple()

if use_oracle:
    maybe add oracle control

if bounds exist:
    add all box corners
    if scalar control:
        run bounded scalar minimization and add its minimizer

for u in candidate_controls:
    project to bounds if needed
    add to candidates

remove duplicate controls

best_val = np.inf
best_control = None

for u in candidates:
    if not problem.local_control_feasible(...):
        continue
    val = p @ problem.f(x, u, t) + problem.l(x, u, t)
    keep the smallest val

if no best_control:
    if restricted:
        return inf, None
    else:
        try unrestricted fallback pass

return best_val, best_control
```

The rest of this section explains each part of the workflow.

---

### 6.1 `bounds = problem.control_bounds_tuple()`

The first important branch depends on whether control bounds are available.

```python
bounds = problem.control_bounds_tuple()
```

If `bounds is not None`, then

```python
u_min, u_max = bounds
```

and the admissible control box is

$$
\begin{equation}
A
=
[u_{\min},u_{\max}].
\end{equation}
$$

If `bounds is None`, the routine cannot generate box corners and cannot run the
bounded scalar minimization. In that case, the only possible candidates come
from:

1. the optional oracle path;
2. the supplied `candidate_controls`.

This means that, for an unbounded or custom-control problem, the caller must
usually provide meaningful candidates or an oracle.

---

### 6.2 Oracle branch

The oracle branch is controlled by

```python
if use_oracle and hasattr(problem, "u_star"):
    ...
```

The code calls

```python
u_oracle, ok = problem.u_star(
    x,
    p,
    t,
    restricted=restricted,
    dt=dt,
)
```

The intended meaning is that `problem.u_star(...)` returns a problem-specific
candidate for

$$
\begin{equation}
\arg\min_u
\left\{
p^\top f(x,u,t)
+
\ell(x,u,t)
\right\}.
\end{equation}
$$

The returned flag `ok` tells whether the oracle control satisfies the requested
restricted feasibility logic.

The oracle candidate is accepted when

```python
(u_oracle is not None) and (not restricted or ok)
```

That is:

- in unrestricted mode, a non-`None` oracle control is accepted;
- in restricted mode, the oracle control is accepted only if `ok=True`.

If bounds exist, the oracle control is projected before being added:

```python
u_oracle = problem.project_control(u_oracle)
```

Mathematically,

$$
\begin{equation}
u_{\mathrm{oracle}}
\leftarrow
\Pi_A(u_{\mathrm{oracle}}).
\end{equation}
$$

Then the projected oracle control is appended:

```python
candidates.append(u_oracle)
```

**Important code detail.** The oracle branch does not return immediately. It
only adds one candidate to the list. The final answer is still chosen after all
other candidates are added and evaluated.

---

### 6.3 Box-corner generation

If bounds exist, the code generates all corners of the box:

```python
u_min, u_max = bounds
m = u_min.size

for combo in product([0, 1], repeat=m):
    u = np.where(np.array(combo) == 0, u_min, u_max)
    candidates.append(u)
```

The variable `combo` is a tuple of zeros and ones. For each component:

- `0` means choose the lower bound;
- `1` means choose the upper bound.

Thus each generated control satisfies

$$
\begin{equation}
u_j
\in
\left\{
(u_{\min})_j,
(u_{\max})_j
\right\},
\qquad
j=1,\dots,m.
\end{equation}
$$

There are $2^m$ such corners.

For example, if $m=2$,

$$
\begin{equation}
u_{\min}=(a,b),
\qquad
u_{\max}=(c,d),
\end{equation}
$$

then the generated corners are

$$
\begin{equation}
(a,b),
\qquad
(c,b),
\qquad
(a,d),
\qquad
(c,d).
\end{equation}
$$

This branch is important for bang-bang problems, because Hamiltonian minimizers
often occur at the boundary of the control set.

---

### 6.4 Scalar bounded minimization branch

After adding box corners, the code checks whether the control is scalar:

```python
if (m == 1):
    ...
```

If so, it runs a bounded scalar optimization over the interval
$[u_{\min},u_{\max}]$.

The scalar bounds are stored as

```python
a_lo = float(u_min[0])
a_hi = float(u_max[0])
```

For a scalar value $a$, the code builds

```python
u = np.array([a], dtype=float)
u = problem.project_control(u)
```

and evaluates the objective

```python
float(np.dot(p, problem.f(x, u, t)) + problem.l(x, u, t))
```

which is the Hamiltonian integrand

$$
\begin{equation}
\phi(a)
=
p^\top f(x,[a],t)
+
\ell(x,[a],t).
\end{equation}
$$

Before returning the objective value, the code checks feasibility:

```python
if not problem.local_control_feasible(
    x,
    u,
    t,
    restricted=restricted,
    dt=dt,
):
    return 1.0e30
```

Thus infeasible scalar controls are not excluded from the optimizer's search
domain directly. Instead, they are penalized by assigning a very large objective
value.

The optimizer call is

```python
res = minimize_scalar(
    obj,
    bounds=(a_lo, a_hi),
    method="bounded",
    options={"xatol": 1e-6, "maxiter": 80},
)
```

If the optimization succeeds and returns a finite value, the scalar minimizer is
added as another candidate:

```python
if res.success and np.isfinite(res.fun):
    candidates.append(np.array([float(res.x)], dtype=float))
```

This branch is important because it allows `compute_H` to capture interior
minimizers for scalar bounded problems. In older versions, the routine only
checked corners and supplied candidates. The current implementation is richer:
for $m=1$, it adds a continuous scalar minimizer to the discrete candidate list.

---

### 6.5 Adding supplied `candidate_controls`

After the oracle, box corners, and scalar minimizer, the code adds the supplied
candidate controls:

```python
for u in candidate_controls:
    if bounds is not None:
        u = problem.project_control(u)
    candidates.append(u)
```

In most calls, `candidate_controls` is `bundle.controls`, the list of controls
stored in the PA bundle.

If bounds exist, each supplied control is projected:

$$
\begin{equation}
u
\leftarrow
\Pi_A(u).
\end{equation}
$$

This protects the Hamiltonian evaluation from small numerical bound violations
or from bundle controls generated under previous approximations.

At this point, the candidate list may contain controls from four sources:

```python
oracle control
box corners
scalar bounded minimizer
PA-bundle or user-supplied controls
```

---

### 6.6 Duplicate removal

The next block removes duplicate or nearly duplicate controls:

```python
unique = []
for u in candidates:
    is_new = True
    for v in unique:
        if np.linalg.norm(u - v) < 1e-10:
            is_new = False
            break
    if is_new:
        unique.append(u)
candidates = unique
```

The tolerance is

$$
\begin{equation}
\|u-v\| < 10^{-10}.
\end{equation}
$$

This is necessary because the same control can appear in multiple ways. For
example, the oracle control might be equal to a box corner, or the scalar
minimizer might coincide with a bundle control.

Duplicate removal keeps the evaluation loop cleaner and avoids repeated work.

---

### 6.7 Main evaluation loop

After candidate construction and duplicate removal, the code initializes

```python
best_val = np.inf
best_control = None
```

It then loops through the unique candidates:

```python
for u in candidates:
    if not problem.local_control_feasible(
        x,
        u,
        t,
        restricted=restricted,
        dt=dt,
    ):
        continue

    val = float(np.dot(p, problem.f(x, u, t)) + problem.l(x, u, t))

    if val < best_val:
        best_val = val
        best_control = u
```

This loop implements the candidate minimization

$$
\begin{equation}
\min_{u\in\mathcal{U}_{cand}^{feas}}
\left\{
p^\top f(x,u,t)
+
\ell(x,u,t)
\right\}.
\end{equation}
$$

Here

$$
\begin{equation}
\mathcal{U}_{cand}^{feas}
=
\left\{
u\in\mathcal{U}_{cand}
:
\texttt{problem.local\_control\_feasible}(x,u,t,\Delta t)
=
\texttt{True}
\right\}.
\end{equation}
$$

The selected value is

$$
\begin{equation}
\texttt{best\_val}
=
\min_{u\in\mathcal{U}_{cand}^{feas}}
\mathcal{H}(p,x,u,t),
\end{equation}
$$

and the selected control satisfies

$$
\begin{equation}
\texttt{best\_control}
\in
\arg\min_{u\in\mathcal{U}_{cand}^{feas}}
\mathcal{H}(p,x,u,t).
\end{equation}
$$

---

### 6.8 Restricted failure versus unrestricted fallback

After the main loop, it is possible that no feasible candidate was found:

```python
if best_control is None:
    ...
```

The behavior then depends on `restricted`.

If `restricted=True`, the function returns immediately:

```python
return float("inf"), None
```

This means the restricted candidate feasible set was empty:

$$
\begin{equation}
\mathcal{U}_{cand}^{feas}
=
\emptyset.
\end{equation}
$$

If `restricted=False`, the code tries one last fallback pass without the local
feasibility filter:

```python
for u in candidates:
    val = float(np.dot(p, problem.f(x, u, t)) + problem.l(x, u, t))
    if val < best_val:
        best_val = val
        best_control = u
```

This fallback is not used in restricted mode. Therefore, constrained calls fail
clearly with `(inf, None)` if no locally feasible candidate exists.

If even the fallback pass fails to find a control, the routine returns

```python
float("inf"), None
```

This usually means the candidate list was empty.

---

### 6.9 Final return and interpretation

If a minimizing candidate is found, the function returns

```python
return best_val, best_control
```

The value `best_val` is the smallest Hamiltonian integrand among the accepted
candidates, and `best_control` is one corresponding minimizer.

It is useful to name this value

$$
\begin{equation}
H_{cand}(p,x,t)
=
\min_{u\in\mathcal{U}_{cand}^{feas}}
\mathcal{H}(p,x,u,t).
\end{equation}
$$

This is the value returned by `compute_H`. It is a **candidate Hamiltonian**:
a reference-like numerical Hamiltonian obtained from the constructed candidate
set.

It equals the true Hamiltonian only when the true minimizer belongs to the
candidate feasible set.

---

## 7) Relation with the PA-bundle surrogate

The PA bundle stores a finite set of controls

$$
\begin{equation}
U_{\mathrm{bundle}}
=
\{u_1,\dots,u_M\}.
\end{equation}
$$

The PA-bundle surrogate Hamiltonian is

$$
\begin{equation}
\bar H(p,x,t)
=
\min_{u\in U_{\mathrm{bundle}}}
\left\{
p^\top f(x,u,t)
+
\ell(x,u,t)
\right\}.
\end{equation}
$$

When `bundle.controls` is passed as `candidate_controls`, the candidate set used
by `compute_H` contains the PA-bundle controls plus additional controls from the
oracle, box corners, and possibly scalar minimization.

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

Therefore, when the same feasibility convention is used,

$$
\begin{equation}
H_{cand}(p,x,t)
\le
\bar H(p,x,t),
\end{equation}
$$

because `compute_H` minimizes over a larger set of controls.

The adaptive loop uses this fact to estimate PA-bundle error through the gap

$$
\begin{equation}
\bar H(p,x,t)
-
H_{cand}(p,x,t).
\end{equation}
$$

At node $t_i$, this gap is

$$
\begin{equation}
e_i^{PA}
=
\bar H(P_i,X_i,t_i)
-
H_{cand}(P_i,X_i,t_i).
\end{equation}
$$

Then the local PA indicator contribution is approximated by a trapezoidal rule:

$$
\begin{equation}
\eta_i^{PA}
=
\frac{1}{2}
\left(
e_i^{PA}
+
e_{i+1}^{PA}
\right)
\Delta t_i.
\end{equation}
$$

A large PA gap means that the current bundle is missing controls that are useful
for the Hamiltonian minimization near the current trajectory. The adaptive loop
then uses `compute_H` to identify such controls and add them to the PA bundle.

---

## 8) Accuracy interpretation

There are three Hamiltonians to keep conceptually separate.

The true Hamiltonian is

$$
\begin{equation}
H(p,x,t)
=
\min_{u\in A_{\mathrm{true}}}
\mathcal{H}(p,x,u,t).
\end{equation}
$$

The PA-bundle surrogate is

$$
\begin{equation}
\bar H(p,x,t)
=
\min_{u\in U_{\mathrm{bundle}}}
\mathcal{H}(p,x,u,t).
\end{equation}
$$

The candidate Hamiltonian returned by `compute_H` is

$$
\begin{equation}
H_{cand}(p,x,t)
=
\min_{u\in\mathcal{U}_{cand}^{feas}}
\mathcal{H}(p,x,u,t).
\end{equation}
$$

If the candidate feasible set is a subset of the true admissible set,

$$
\begin{equation}
\mathcal{U}_{cand}^{feas}
\subseteq
A_{\mathrm{true}},
\end{equation}
$$

then

$$
\begin{equation}
H(p,x,t)
\le
H_{cand}(p,x,t).
\end{equation}
$$

So `compute_H` is not automatically the exact analytic Hamiltonian. It is exact
when the true minimizer is included in the constructed candidate set.

The current implementation improves this candidate set in several ways:

- box corners help for bang-bang controls;
- scalar bounded minimization helps for scalar interior controls;
- oracle controls help when analytic or problem-specific minimizers are known;
- PA-bundle controls carry information learned during adaptivity.

This is why `compute_H` is best understood as a **reference-like candidate
Hamiltonian evaluator**.