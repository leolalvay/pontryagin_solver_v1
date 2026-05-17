# `core/pa_bundle.py` — PA Bundle (`PABundle`)

This module implements the **PA bundle**, the data structure that stores a finite set of candidate controls. The bundle is central to the adaptive smoothed-PMP method because it defines a **piecewise-affine (PA) surrogate Hamiltonian** (PA in the costate variable $p$) and provides reusable control candidates for Hamiltonian minimization across time.

In the codebase, the bundle is typically passed around as `bundle`, and the candidate set is accessed as `bundle.controls`.

---

## 1) Mathematical role of the PA bundle (PA upper bound in $p$)

For a fixed $(x,t)$, define the Hamiltonian integrand
$$
\mathcal{H}(p,x,u,t) := p^\top f(x,u,t) + \ell(x,u,t).
$$

The **true Hamiltonian** (minimization convention) is
$$
H(p,x,t) := \min_{u\in A}\, \mathcal{H}(p,x,u,t)
= \min_{u\in A}\Big\{\, p^\top f(x,u,t) + \ell(x,u,t)\,\Big\},
$$
where $A=[u_{\min},u_{\max}]$ is the control box.

### Concavity in $p$ and supporting hyperplanes

For each fixed $(x,t)$, the map $p\mapsto H(p,x,t)$ is **concave**, because it is the pointwise minimum of affine functions of $p$. Therefore, for any reference point $\hat p$ and any subgradient $g\in \partial_p H(\hat p,x,t)$,
$$
H(p,x,t)\;\le\; H(\hat p,x,t) + g^\top (p-\hat p),\qquad \forall p.
$$

By defining  
$$
g_i(x,t):=\partial_pH(\hat{p}_i,x,t)\\
d_i(x,t):=H(\hat{p}_i,x,t)-g_i(x,t)\hat{p}_i
$$
We construct a piecewise-affine (PA) surrogate
$$
\bar{H}(p,x,t)=\min_{1\leq i\leq M}\{g_i(x,t)\cdot p+d_i(x,t)\}
$$
which satisfies
$$
H(p,x,t)\leq \bar{H}(p,x,t)
$$
For the Hamiltonian above, any minimizer $\hat u \in \arg\min_{u\in A}\mathcal{H}(\hat p,x,u,t)$ provides a valid subgradient:
$$
g = \partial_p H(\hat p,x,t) = f(x,\hat u,t),
$$
and
$$
H(\hat p,x,t) = \hat p^\top f(x,\hat u,t) + \ell(x,\hat u,t).
$$
Substituting yields the supporting affine upper bound
$$
H(p,x,t)\;\le\; f(x,\hat u,t)^\top p + \ell(x,\hat u,t).
$$
### Bundle representation and surrogate $\bar H$

The PA bundle stores a finite set of controls
$$
\mathcal{U}_{\mathrm{bundle}} = \{u^{(1)},\dots,u^{(K)}\} \subset A.
$$
For each stored control $u^{(i)}$, define the affine function of $p$ (for the current $(x,t)$)
$$
\phi_i(p;x,t) := g_i(x,t)^\top p + d_i(x,t),
\qquad
g_i(x,t):= f(x,u^{(i)},t),\quad d_i(x,t):=\ell(x,u^{(i)},t).
$$
The **bundle surrogate Hamiltonian** is the minimum over these affine functions:
$$
\bar H(p,x,t) := \min_{i=1,\dots,K}\, \phi_i(p;x,t)
= \min_{u\in \mathcal{U}_{\mathrm{bundle}}}\mathcal{H}(p,x,u,t).
$$

Because $\mathcal{U}_{\mathrm{bundle}}\subset A$, minimizing over a smaller set can only increase the minimum value, hence $\bar H$ is an **upper bound**:
$$
H(p,x,t)\;\le\;\bar H(p,x,t)\qquad \forall (p,x,t).
$$

### Why this matters

- $\bar H$ is cheap to evaluate (finite minimum / PA model in $p$).
- The adaptive algorithm monitors the **bundle error indicator**
  $$
  \eta_{\mathrm{PA}} \approx \int_0^T\big(\bar H(p(t),x(t),t) - H(p(t),x(t),t)\big)\,dt,
  $$
  and refines the bundle when this gap is too large.

> Implementation note: the code typically stores **controls** $u^{(i)}$ (i.e., `bundle.controls`) rather than storing the affine coefficients $(g_i,d_i)$ explicitly. The coefficients are computed **on the fly** via
> $$
> g_i(x,t)=f(x,u^{(i)},t),\qquad d_i(x,t)=\ell(x,u^{(i)},t),
> $$
> whenever $\bar H(p,x,t)$ needs to be evaluated.

---

## 2) Example-driven intuition (Example 1: LQR)

In Example 1 (LQR), the unconstrained minimizer is generally **interior**:
$$
u^*(t) = -\tfrac{1}{2}R^{-1}B^\top p(t)
\quad\text{(if }\ell \text{ uses }u^\top Ru \text{ without a } \tfrac12 \text{ factor).}
$$

If you only minimize over **box corners** (extreme points), you often get a poor approximation for $u^*$ unless the bounds are very tight or the solution saturates.

The PA bundle is the mechanism that allows the solver to “discover” and reuse good **interior candidate controls**, so that the discrete minimization can approximate the interior optimum.

---

## 3) What the bundle stores (conceptual view)

The bundle is a small container holding:

- `controls`: a list/array of control vectors $u^{(k)} \in \mathbb{R}^m$
- `max_size` / capacity: upper bound on how many controls are kept
- tolerances for **deduplication**: avoid storing controls that are nearly identical

In practice:
- controls are always intended to be **feasible** ($u\in A$), enforced via projection/clipping when bounds exist.
- controls are stored in a stable format (numpy arrays, float dtype).

---

## 4) Core operations and how to interpret them

### 4.1 Initialization

A bundle must start with a **non-empty candidate set** to be useful. Typical initial sources:

- corners of the control box (via `OCPProblem.control_bounds`)
- a small set of random controls in $A$ (optional, depending on the experiment)
- heuristics or problem-specific initial guesses (rare)

If the bundle is empty and the code does not add corners elsewhere, downstream Hamiltonian minimizations can fail.

---

### 4.2 Insertion / update (adding new candidates)

Whenever the algorithm finds a “useful” control candidate (e.g., from the smoothed argmin or from a local improvement step), it attempts to add it to the bundle:

1. **Project to the box** (if bounds exist):
   $$
   u \leftarrow \Pi_A(u).
   $$
2. **Check near-duplicates**:
   - if $\|u - u^{(k)}\|$ is below a tolerance for some stored $u^{(k)}$, skip insertion.
3. **Insert if capacity allows**:
   - append if current size $< \texttt{max\_size}$.
4. **Replace if full** (bundle management policy):
   - if at capacity, replace an existing control using a simple rule (e.g., remove the “least useful” or oldest).

> The exact replacement policy is implementation-dependent. The key idea is that the bundle maintains a compact, diverse set of candidate controls.

---

### 4.3 Bundle evaluation: computing $\bar H(p,x,t)$

The bundle evaluates the surrogate Hamiltonian by looping over its stored controls:
$$
\bar H(p,x,t) = \min_{u\in \mathcal U_{\mathrm{bundle}}} \mathcal{H}(p,x,u,t).
$$

This is essentially the same minimization pattern as `compute_H`, but restricted to bundle controls only.

### `evaluate` flow (`pa_bundle.py`)

- If there are no controls (`self.controls` is empty), raise `RuntimeError` (cannot evaluate).
- Initialize `best_val = np.inf` and `best_idx = -1` as worst-case placeholders.
- Loop over stored controls with `enumerate` to get both index `i` and control `u`:
  - Compute `val = float(np.dot(p, problem.f(x, u, t)) + problem.l(x, u, t))`.
  - If `val` is smaller than `best_val`, update `best_val = val` and `best_idx = i`.
- After the loop, return `(best_val, best_idx)`: the minimum surrogate Hamiltonian value and the index of the control that achieved it.

---

### 4.4 Where do `bundle.controls` come from in this implementation?

`PABundle` is a **passive container**: it does not generate controls by itself. It only stores controls passed to `add_control(u)` and later evaluates
$$
\bar H(p,x,t)=\min_{u\in \mathcal U_{\mathrm{bundle}}}\{p^\top f(x,u,t)+\ell(x,u,t)\}.
$$

In this repository, controls enter the bundle from the **outer adaptivity loop** (`core/adaptivity.py`):

1. **Initialization (seed control).**  
   At startup the bundle is created empty and seeded with one feasible control `u0`:
   - if bounds exist, the midpoint $u_0=\tfrac12(u_{\min}+u_{\max})$ is added,
   - otherwise $u_0=0$ is added.  
   This prevents `bundle.evaluate(...)` from failing when the bundle is empty.

2. **PA refinement (learning new controls).**  
   When the PA indicator is too large, the outer loop locates the mesh node with the largest gap
   $$
   \bar H(p_i,x_i,t_i) - H(p_i,x_i,t_i),
   $$
   where $\bar H$ is computed by `bundle.evaluate(...)` (bundle-only) and $H$ is computed by
   `compute_H(...)` (finite minimization over **box corners + bundle controls**).
   The argmin control returned by `compute_H(...)` at the worst-gap node is then inserted:
   $$
   u^* \leftarrow \arg\min_{u\in \mathcal U_{\mathrm{cand}}}\{p_i^T f(x_i,u,t_i)+\ell(x_i,u,t_i)\},
   \qquad
   \texttt{bundle.add_control}(u^*).
   $$

**Key takeaway.** The bundle is populated by controls that are discovered as pointwise Hamiltonian minimizers
during the adaptive loop (plus an initial seed control). The corners of the control box are not necessarily stored
from the start; instead, they are always available as candidates inside `compute_H(...)` and may be added to the
bundle later if they become active minimizers.


## 5) Relationship to `compute_H(...)`

`compute_H(problem, p, x, t, candidate_controls, ...)` typically uses:

- all **control-box corners**, plus
- `candidate_controls` (usually `bundle.controls`) projected into the box,

and then returns:
- `best_val`: the minimum value found among candidates (a finite approximation of $H(p,x,t)$)
- `best_control`: the candidate achieving that minimum (a finite approximation of an argmin).

So the bundle acts as the “memory” that provides interior controls and stabilizes the Hamiltonian minimization across time.

---

## 6) Debugging checklist (high-value checks)

When you see odd behavior (no improvement in $\eta_{\mathrm{PA}}$, weird controls, crashes), check:

1. **Bundle is non-empty**
   - `len(bundle.controls) > 0` after initialization.

2. **Correct control dimension**
   - each stored control has shape `(m,)` and consistent `dtype=float`.

3. **Bounds consistency**
   - if bounds exist, ensure controls are actually within bounds (projection applied).
   - avoid infinite bounds for corners (can generate `inf` candidates).

4. **Dedup tolerance**
   - too strict: bundle fills with near-duplicates (low diversity).
   - too loose: useful distinct controls might be rejected.

5. **Capacity / replacement policy**
   - if `max_size` is too small, the bundle may not keep enough diversity to reduce $\eta_{\mathrm{PA}}$.
   - if replacement removes good interior candidates too aggressively, Example 1 quality may degrade.

---

## 7) Where the bundle is used

- `core/adaptivity.py`: bundle refinement is triggered by the PA indicator $\eta_{\mathrm{PA}}$.
- `core/hamiltonian.py`: bundle controls are included as candidate controls for finite minimization.
- `experiments/ex*.py`: reconstructed controls (plots) often depend on bundle-supported candidate sets.

---

## 8) Key takeaway

The PA bundle stores controls $\{u^{(i)}\}$ and induces a **piecewise-affine upper bound** $\bar H$ of the true Hamiltonian $H$ (upper bound because the minimization is restricted to a subset of admissible controls). It is the main mechanism that makes Hamiltonian evaluation practical and reusable across time, especially in problems where the true minimizer is not captured well by box corners alone (e.g., interior solutions such as LQR).


# Bundle vs. Supergradient Planes — Key Conclusions (from our discussion)

## 1) Two different objects: pre-Hamiltonian vs. reduced Hamiltonian
- Define the **pre-Hamiltonian** (control-dependent):
  \[
  g_u(p,x,t) := p^\top f(x,u,t) + \ell(x,u,t).
  \]
- Define the **reduced Hamiltonian** (control eliminated):
  \[
  H(p,x,t) := \min_{u\in\mathcal U(x,t)} g_u(p,x,t),
  \]
  where \(\mathcal U(x,t)\) encodes control bounds and (if used) viability/tangent constraints.

## 2) What the current PA-bundle in the repo actually stores and evaluates
- The bundle stores a **finite set of controls** \(\mathcal U_B=\{u_1,\dots,u_M\}\).
- The surrogate is:
  \[
  \bar H(p,x,t) := \min_{u\in\mathcal U_B} g_u(p,x,t).
  \]
- When the evaluation point changes \((p,x,t)\to(p',x',t')\), the bundle does **not** recompute derivatives or solve a new minimization problem; it simply **re-evaluates** \(f(x',u_i,t')\) and \(\ell(x',u_i,t')\) for the stored \(u_i\) and takes the minimum.

## 3) Why the control-based bundle is a (global) upper bound
- Since \(H\) is a minimum over all admissible controls:
  \[
  H(p,x,t) \le g_u(p,x,t)\quad \forall u\in\mathcal U(x,t).
  \]
- If each stored \(u_i\) is admissible at the evaluation point, then:
  \[
  H(p,x,t) \le \bar H(p,x,t)\quad \forall (p,x,t),
  \]
  i.e., the surrogate is an **upper bound globally in \((p,x,t)\)**.
- **Envelope theorem is not what guarantees the upper bound**. The upper bound comes from “minimum over a subset.”  
  Envelope/Danskin helps in **enrichment**: it tells you how the active control relates to \(\partial_p H\), and motivates adding good controls.

## 4) Why the bundle can be a good or bad approximation at a new point
- Evaluating \(\bar H(p,x,t)\) at a new point is **not** “applying envelope theorem.”
- A stored control \(u_i\) may have been optimal at \((\hat p_i,\hat x_i,\hat t_i)\), but it may be good or bad at a different \((p,x,t)\).
- The approximation quality is measured by the gap:
  \[
  \bar H(p,x,t) - H(p,x,t) \ge 0,
  \]
  which is small only if \(\mathcal U_B\) contains a control near-optimal for that point.

## 5) Role of adaptivity (enrichment)
- If you identify a point where the gap is large, you enrich by computing a “good” (ideally optimal) control \(u^\*(p,x,t)\) for that point and adding it to \(\mathcal U_B\).
- Adding the true active control at \((\hat p,\hat x,\hat t)\) makes the surrogate **exact at that point** and often improves a neighborhood (unless you are near switching/ties).

## 6) Supergradient-based planes: valid, but not globally reusable in \((x,t)\)
- For fixed \((x,t)\), \(p\mapsto H(p,x,t)\) is concave, so for \(s\in\partial_p H(\hat p,x,t)\):
  \[
  H(p,x,t) \le H(\hat p,x,t) + s^\top(p-\hat p)\quad \forall p.
  \]
- If you **freeze** the coefficients (store the plane as a function of \(p\) only) and then reuse it at different \((x',t')\), the upper-bound guarantee generally **breaks**, because the concavity argument is only in \(p\) with \((x,t)\) fixed.
- A consistent “supergradient bundle” would be **local in \((x,t)\)**: when \((x,t)\) changes, you must recompute \(H(\hat p,x,t)\) and \(\partial_p H(\hat p,x,t)\) (i.e., planes are not reusable across \((x,t)\) unless you can evaluate these quantities as functions).

## 7) Constraints and viability (restricted Hamiltonian)
- If using a restricted Hamiltonian
  \[
  H_K(p,x,t)=\min_{u:\ f(x,u,t)\in T_K(x)} g_u(p,x,t),
  \]
  then to preserve the upper-bound property the surrogate must minimize only over **viable** controls at that \((x,t)\), i.e., apply the same feasibility filter (e.g., `tangent_ok`) consistently.
- Control bounds are handled by projection/clipping; viability depends on the state and may vary with \((x,t)\).

## 8) “Explicit Hamiltonian” as an optional oracle in a universal solver
- Knowing a closed form for \(H(p,x,t)\) typically implies you can recover the minimizer \(u^\*(p,x,t)\) (possibly by cases / clipping under bounds).
- In a universal solver, the clean idea is: **optionally** provide an explicit/oracle evaluation for \(H\) (and ideally \(u^\*\)) when available; otherwise fall back to the current numerical routine.
- This oracle is most useful to:
  - speed up computing “true” \(H\) for \(\eta_{PA}\),
  - provide accurate enrichment controls \(u^\*\),
  - serve as a benchmark/reference example (e.g., hypersensitive control Example 3.1).
