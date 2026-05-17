# `core/hamiltonian.py` — `compute_H`

This module implements `compute_H(...)`, the solver’s **reference Hamiltonian evaluation** routine. It is used for:

1. **Reconstructing an approximate optimal control** from a solved state–costate trajectory $(x(t),p(t))$ (see the `experiments/` scripts).
2. **Computing the PA-bundle indicator** $\eta_{\mathrm{PA}}$ in the adaptive outer loop, via the gap $(\bar H - H)$.

---

## 1) Mathematical definition (minimization convention)

This repository uses the **minimization** convention:

$$
H(p,x,t) \;:=\; \min_{u\in A}\Big\{\, p^\top f(x,u,t) \;+\; \ell(x,u,t)\,\Big\},
$$

where:

- $x\in\mathbb{R}^n$ is the state,
- $p\in\mathbb{R}^n$ is the costate,
- $u\in\mathbb{R}^m$ is the control,
- $A=[u_{\min},u_{\max}]$ is typically a **box** (componentwise bounds),
- $f$ and $\ell$ come from `OCPProblem`.

> Many PMP texts use a *maximization* convention. Here the solver consistently uses **min** because it is aligned with minimizing the cost functional.

---

## 2) Restricted Hamiltonian (state constraints via viability)

When a **state constraint** $x(t)\in K$ is present (in this repo: $K$ is also a box), the solver can compute a *restricted* Hamiltonian:

$$
H_K(p,x,t)\;:=\;\min_{\substack{u\in A\\ f(x,u,t)\in T_K(x)}}\Big\{\, p^\top f(x,u,t) \;+\; \ell(x,u,t)\,\Big\},
$$

where $T_K(x)$ is the tangent cone to the feasible set $K$ at $x$.

### How the repo models $f(x,u,t)\in T_K(x)$
For box constraints $K=[x_{\min},x_{\max}]$, the condition is implemented by a simple sign rule on *active boundaries*:

- if $x_i \approx (x_{\min})_i$, require $\dot x_i \ge 0$,
- if $x_i \approx (x_{\max})_i$, require $\dot x_i \le 0$,

with a small tolerance `tol`.

This is checked through `problem.tangent_ok(x,u,t)`.

---

## 3) Function signature and contract

Conceptually, the function is:

```python
def compute_H(problem, p, x, t, candidate_controls, restricted=False) -> (float, np.ndarray):
    ...
```

### Inputs
- `problem`: `OCPProblem`
- `p`: costate vector of shape `(n,)`
- `x`: state vector of shape `(n,)`
- `t`: time (float)
- `candidate_controls`: list of candidate controls (each shape `(m,)`)
  - typically: `bundle.controls` from `core/pa_bundle.py`
- `restricted`: if `True`, enforce viability via `problem.tangent_ok(...)`

### Outputs
- `best_val`: the minimal Hamiltonian value (float)
- `best_control`: the corresponding minimizing control $u^*$ (np.ndarray of shape `(m,)`)

### Important clarification: 

In `compute_H`, **candidates** refer to *different control vectors* $u \in \mathbb{R}^m$ evaluated at the **same** triple $(p,x,t)$.

For a fixed $(p,x,t)$, the code builds a finite candidate set
$$
\mathcal{U}_{\mathrm{cand}} \subset A
\quad\text{(e.g., box corners + bundle controls),}
$$
and computes
$$
\texttt{best\_val} = \min_{u\in \mathcal{U}_{\mathrm{cand}}}\ \mathcal{H}(p,x,u,t),
\qquad
\texttt{best\_control} \in \arg\min_{u\in \mathcal{U}_{\mathrm{cand}}}\ \mathcal{H}(p,x,u,t),
$$
where
$$
\mathcal{H}(p,x,u,t) := p^\top f(x,u,t) + \ell(x,u,t).
$$

A **control trajectory** $u(t)$ is obtained only after repeating the same pointwise minimization at each time node $t_i$:
$$
u_i := \texttt{best\_control}(p_i,x_i,t_i), \qquad i=0,\dots,N.
$$
Plotting $(t_i,u_i)$ then produces the discrete approximation of $u(t)$ (often piecewise constant / bang–bang when the minimum occurs at box corners).


> Practical requirement: `compute_H` needs **at least one candidate** to return a meaningful `best_control`. In this repo, that is guaranteed by providing `control_bounds` (to generate corners) and/or a non-empty `candidate_controls`.

---

## 4) Implementation walkthrough (minimal mental model)

`compute_H` does **finite candidate minimization**. It does *not* solve a continuous optimization problem over $A$.

### 4.1 Candidate set construction

The function builds the list `candidates` as:

1. **Extreme points of the control box** $A=[u_{\min},u_{\max}]$:
   - it enumerates all $2^m$ “corners” (vertices). Each corner is a vector $u^{(k)}\in\mathbb{R}^m$ obtained by choosing, for every component $j=1,\dots,m$, either the lower or the upper bound:
     $$
     u^{(k)}_j \in \{(u_{\min})_j,\,(u_{\max})_j\},\qquad j=1,\dots,m,\quad k=1,\dots,2^m.
     $$
   - **Visual intuition (2D example).**  
     Let $m=2$ and write
     $$
     u_{\min} = (a,b), \qquad u_{\max} = (c,d), \qquad a<c,\; b<d.
     $$
     This means, componentwise,
     $$
     u_1 \in [a,c], \qquad u_2 \in [b,d],
     $$
     so the control box is the rectangle
     $$
     A = [u_{\min},u_{\max}] = [a,c]\times[b,d].
     $$
     Its **extreme points (corners)** are obtained by choosing, for each component $j\in\{1,2\}$, either the lower or the upper bound:
     $$
     u_j \in \{(u_{\min})_j,\,(u_{\max})_j\}.
     $$
     Concretely, the four corners are:
     $$
     (a,b),\ (c,b),\ (a,d),\ (c,d).
     $$

     ```
     u2
     ^
     |      (a,d) •---------• (c,d)
     |            |         |
     |            |    A    |
     |            |         |
     |      (a,b) •---------• (c,b)
     +-------------------------------> u1
     ```


2. **User-supplied candidate controls** (usually `bundle.controls`):
   - if bounds exist, each candidate is **projected** onto the box $A=[u_{\min},u_{\max}]$ before evaluation:
     $$
     u \leftarrow \Pi_A(u),
     $$
     where $\Pi_A$ is the (componentwise) clipping operator
     $$
     (\Pi_A(u))_j \;=\; \min\Big\{\max\{u_j,(u_{\min})_j\},\ (u_{\max})_j\Big\},\qquad j=1,\dots,m.
     $$
     Intuitively, if a candidate violates the bounds in any component, it is “snapped” back to the nearest boundary value in that component; if it is already feasible, it is unchanged.
     - *Example (2D):* if $u_{\min}=(a,b)=(-1,0)$ and $u_{\max}=(c,d)=(2,3)$, then $\Pi_A(4,-2)=(2,0)$ and $\Pi_A(1,10)=(1,3)$.


3. **Duplicate removal**:
   - candidates closer than a tiny tolerance are deduplicated (norm threshold $\approx 10^{-10}$).

**Complexity note:** the corner enumeration is $O(2^m)$. This is fine for small $m$ (e.g., $m=1,2,3$) but becomes expensive as $m$ grows.

---

### 4.2 Candidate evaluation loop

For each candidate $u$:

1. If `restricted=True`, apply viability filter:
   - skip if `problem.tangent_ok(x,u,t)` is `False`.

2. Check admissibility (box constraints):
   - skip if `not problem.admissible_control(u)`.

3. Evaluate the Hamiltonian integrand:
   $$
   \mathcal{H}(p,x,u,t) := p^\top f(x,u,t) + \ell(x,u,t).
   $$

4. Keep the minimizing control:
   - track the smallest value and store its $u$.

### 4.3 Fallback behavior
If all candidates are filtered out (e.g., due to an overly strict viability test), the code attempts a fallback pass over candidates **without** viability checking. This is meant to be rare in a well-posed setup.

---

## 5) Relationship with the PA-bundle surrogate $\bar H$

The PA-bundle defines a surrogate Hamiltonian:

$$
\bar H(p,x,t)\;:=\;\min_{u\in\mathcal{U}_{\text{bundle}}}\Big\{p^\top f(x,u,t)+\ell(x,u,t)\Big\},
$$

where $\mathcal{U}_{\text{bundle}}$ is the finite set of controls stored in the bundle.

### How `compute_H` is used in the adaptive indicator
The adaptivity loop computes:

$$
\eta_{\mathrm{PA}} \approx \int_0^T \big(\bar H(p(t),x(t),t) - H(p(t),x(t),t)\big)\,dt.
$$

In code:
- $\bar H$ is computed by `bundle.evaluate(...)` (bundle-only minimization),
- $H$ is computed by `compute_H(...)` (bundle controls + box corners).

**Important interpretation:** in this repo, `compute_H` is “more reference-like” than $\bar H$, but it is still a **finite candidate** minimization. It is exact only if the true minimizer is among the enumerated candidates (typical in bang–bang problems). For problems with interior minimizers (e.g., unconstrained LQR), accuracy depends on whether the candidate set contains good interior points (usually coming from the bundle).

---

## 6) Example-based documentation

### 6.1 Example 1 (LQR): what `compute_H` means here

Example 1 defines:

- $f(x,u,t)=Ax+Bu$,
- $\ell(x,u,t)=x^\top Qx + u^\top Ru$.

So the Hamiltonian integrand is:

$$
\mathcal{H}(p,x,u,t) = p^\top(Ax+Bu) + x^\top Qx + u^\top Ru.
$$

If $A$ were unbounded and we minimized exactly over all $u\in\mathbb{R}^m$, the optimal control satisfies:

$$
\nabla_u \mathcal{H} = B^\top p + 2Ru = 0
\quad\Rightarrow\quad
u^*(t)= -\tfrac{1}{2}R^{-1}B^\top p(t),
$$

(using the convention $\ell$ contains $u^\top Ru$ without a $\tfrac{1}{2}$ factor).

**What the repo does instead:** it approximates the minimization over the bounded box $A=[u_{\min},u_{\max}]$ by checking:
- the box corners (for $m=1$, simply $u_{\min}$ and $u_{\max}$),
- plus the bundle’s discrete controls.

This is why Example 1 uses “large bounds” to approximate an unconstrained LQR while keeping the solver architecture consistent.

Typical call (from the example script):

```python
_, u_star = compute_H(prob, P[i], X[i], mesh[i], bundle.controls, restricted=True)
```

(`restricted=True` is harmless in Example 1 because there are no state bounds, so `tangent_ok` returns `True`.)

---

### 6.2 Example 2 (double integrator with $x_1\le 0$): why `restricted=True` matters

Example 2 encodes the state constraint $x_1\le 0$ as bounds:

- $(x_{\max})_1 = 0$,
- other components unbounded via $\pm\infty$.

At points where $x_1$ is on/near the boundary, the viability rule enforces:

$$
\dot x_1 \le 0 \quad\text{if } x_1 \approx 0,
$$

so candidate controls that would push $x_1$ positive get filtered out by `tangent_ok`.

This corresponds to computing the restricted Hamiltonian $H_K$ (finite-candidate version).

---

## 7) Debugging checklist (high-value checks)

When `compute_H` produces “weird” controls or crashes, check:

1. **Candidate set non-emptiness**
   - Ensure `control_bounds` is provided (so corners exist), or `candidate_controls` is non-empty.
   - If both are missing/empty, `best_control` can remain `None`.

2. **Shapes**
   - `p.shape == (n,)`, `x.shape == (n,)`, and each `u.shape == (m,)`.
   - Avoid `(m,1)` controls (broadcasting can silently corrupt comparisons and dot products).

3. **Finite values**
   - If `f` or `l` produces NaNs/Infs, minimization becomes unreliable.

4. **State-constraint viability tolerance**
   - If many candidates get filtered out in constrained problems, inspect the tolerance used in `tangent_ok`.
   - Too strict → no viable candidates; too loose → constraint violation.

5. **High control dimension**
   - Corner enumeration is $2^m$. If $m$ is large, consider customizing candidate generation.

---

## 8) Where `compute_H` is used

- `core/adaptivity.py`: PA error indicator $\eta_{\mathrm{PA}}$ uses $(\bar H - H)$.
- `experiments/ex*.py`: reconstruct approximate controls from the solved $(X,P)$ trajectory.
- Any custom postprocessing: objective approximation, plotting controls, diagnosing switching structure.
