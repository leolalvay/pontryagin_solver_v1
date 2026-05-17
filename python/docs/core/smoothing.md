# `core/smoothing.py` — Smoothed surrogate Hamiltonian (`eval_H_smooth`)

This module implements the **δ-smoothed** version of the **piecewise-affine (PA) surrogate Hamiltonian** induced by a PA bundle.

The goal here is **not** to document the syntax, but to explain what the routine computes **mathematically**, and why it is central to the solver (Newton + symplectic discretization + adaptivity).

---

## 1) What we are smoothing: the bundle-restricted Hamiltonian surrogate

Given a problem in Bolza form (see `core/problem.py`) and a PA bundle (see `core/pa_bundle.py`) containing a finite set of controls
$$
\mathcal{U}_{\mathrm{bundle}}=\{u^{(1)},\dots,u^{(m)}\}\subset A,
$$
define, for each candidate control $$u^{(i)}$$ the “plane value”
$$
g_i(p,x,t)
:=
p^\top f(x,u^{(i)},t)+\ell(x,u^{(i)},t).
$$

The PA surrogate (bundle-restricted) Hamiltonian is the **minimum over planes**
$$
\bar H(p,x,t)
:=
\min_{1\le i\le m}\, g_i(p,x,t).
$$

This object is **non-smooth** whenever there are ties or near-ties between planes (switching between minimizers). That non-smoothness is what makes Newton-type TPBVP solvers fragile: they want consistent gradients.

---

## 2) δ-smoothing via log-sum-exp (soft-min)

`smoothing.py` replaces the hard minimum by a **soft-min / log-sum-exp** smooth approximation:
$$
H_\delta(p,x,t)
:=
-\delta \log\!\left(\sum_{i=1}^m \exp\!\left(-\frac{g_i(p,x,t)}{\delta}\right)\right),
\qquad \delta>0.
$$

Key properties:

- **Lower envelope / monotone limit**
$$
H_\delta(p,x,t)\le \bar H(p,x,t),
\qquad
H_\delta(p,x,t)\xrightarrow[\delta\downarrow 0]{}\bar H(p,x,t).
$$

- **Quantitative approximation bound** (letting $g_\star=\min_i g_i$)
$$
g_\star-\delta\log m \;\le\; H_\delta \;\le\; g_\star.
$$

Interpretation: $H_\delta$ is an entropically-regularized minimum. Smaller $\delta$ makes the approximation sharper (closer to $\bar H$) but less smooth numerically.

---

## 3) Soft-min weights and exact gradient identities

Define the soft-min (Gibbs) weights
$$
w_i(p,x,t;\delta)
:=
\frac{\exp\!\left(-g_i(p,x,t)/\delta\right)}
{\sum_{j=1}^m \exp\!\left(-g_j(p,x,t)/\delta\right)},
\qquad
\sum_i w_i = 1,\quad w_i\ge 0.
$$

Then log-sum-exp smoothing satisfies the exact identity
$$
\nabla H_\delta(p,x,t)
=
\sum_{i=1}^m w_i \,\nabla g_i(p,x,t).
$$

Two components matter for the PMP discretization:

### (a) Gradient with respect to the costate $p$

Since
$$
g_i(p,x,t)=p^\top f(x,u^{(i)},t)+\ell(x,u^{(i)},t),
\qquad
\nabla_p g_i = f(x,u^{(i)},t),
$$
we get
$$
\nabla_p H_\delta(p,x,t)
=
\sum_{i=1}^m w_i\, f(x,u^{(i)},t).
$$

So the smoothed state direction is a **convex combination** of the candidate dynamics.

In the sharp limit $\delta\downarrow 0$, the weights concentrate on minimizers of $g_i$; when the minimizer is unique,
$$
\nabla_p H_\delta \to f(x,u^*,t),
\qquad
u^*\in\arg\min_i g_i.
$$
When minimizers are multiple, the limit naturally yields a **subgradient-like convex combination**, which is exactly what we want near switching.

### (b) Gradient with respect to the state $x$

Formally,
$$
\nabla_x g_i
=
\left(\nabla_x f(x,u^{(i)},t)\right)^\top p
+
\nabla_x \ell(x,u^{(i)},t),
$$
hence
$$
\nabla_x H_\delta(p,x,t)
=
\sum_{i=1}^m w_i\,\nabla_x g_i(p,x,t).
$$

This is the quantity that drives the costate equation via $$-\dot p = \nabla_x H_\delta$$

---

## 4) What the implementation actually does

The function implemented here is conceptually:

- `eval_H_smooth(problem, bundle, p, x, t, delta) -> (H_delta, grad_p, grad_x)`.

### Step 1: Evaluate all planes $g_i$ and store dynamics $f_i$

Loop over `bundle.controls` (size $m$):

- compute $$f_i = f(x,u^{(i)},t)$$
- compute $$g_i = p^\top f_i + \ell(x,u^{(i)},t)$$
- store $f_i$ into `f_vals` for later use in $$\nabla_p H_\delta$$.

### Step 2: Stable log-sum-exp

### Why we shift by $g_{\min}$ in the log-sum-exp smoothing

The smoothed (soft-min) Hamiltonian is defined as

$$
H_\delta(p,x,t)
=
-\delta \log\!\left(\sum_{i=1}^m \exp\!\left(-\frac{g_i(p,x,t)}{\delta}\right)\right),
\qquad \delta>0.
$$

In practice, when $\delta$ is small, the terms $\exp(-g_i/\delta)$ can **underflow to $0$** or **overflow to $\infty$** in floating-point arithmetic, even though the mathematical expression is well-defined. The shift fixes this by keeping all exponentials in a safe range.

### Stable rewrite (exactly equivalent)

Let $g_{\min}:=\min_i g_i(p,x,t)$. Then

$$
\sum_{i=1}^m \exp\!\left(-\frac{g_i}{\delta}\right)
=
\sum_{i=1}^m
\exp\!\left(-\frac{g_{\min}}{\delta}\right)
\exp\!\left(-\frac{g_i-g_{\min}}{\delta}\right)
=
\exp\!\left(-\frac{g_{\min}}{\delta}\right)
\sum_{i=1}^m
\exp\!\left(-\frac{g_i-g_{\min}}{\delta}\right).
$$

Taking the logarithm:

$$
\log\!\left(\sum_{i=1}^m \exp\!\left(-\frac{g_i}{\delta}\right)\right)
=
-\frac{g_{\min}}{\delta}
+
\log\!\left(\sum_{i=1}^m \exp\!\left(-\frac{g_i-g_{\min}}{\delta}\right)\right).
$$

Multiplying by $-\delta$ gives the implementation form:

$$
H_\delta(p,x,t)
=
g_{\min}
-\delta\log\!\left(\sum_{i=1}^m \exp\!\left(-\frac{g_i-g_{\min}}{\delta}\right)\right).
$$

### Why this helps (and does not change the method)

- **Numerical stability:** since $g_i-g_{\min}\ge 0$, all exponents satisfy $-(g_i-g_{\min})/\delta\le 0$, so the terms $\exp(-(g_i-g_{\min})/\delta)\in(0,1]$. This avoids overflow and reduces catastrophic underflow.
- **No conceptual change:** the rewrite is **algebraically identical**, so it produces the same $H_\delta$ as the original definition.
- **Weights unchanged:** the soft-min weights are invariant under the same shift:
  $w_i \propto \exp(-g_i/\delta)$ is equivalent to $w_i \propto \exp(-(g_i-g_{\min})/\delta)$, because the common factor cancels in normalization.


Typical numerical safeguards used in practice:

- replace $\delta$ by $\max(\delta,10^{-12})$ to avoid division by zero,
- add a tiny constant inside the log (e.g. $10^{-300}$) to avoid $\log(0)$.

### Step 3: Soft-min weights

Weights are normalized:
$$
w_i = \frac{\mathrm{exps}_i}{\sum_j \mathrm{exps}_j}.
$$

### Step 4: Compute $\nabla_p H_\delta$ exactly

Using stored dynamics values:
$$
\nabla_p H_\delta = \sum_i w_i f_i.
$$

### Step 5: Compute $\nabla_x H_\delta$ by finite differences on $g_i$

## How the code computes the gradient w.r.t. $x$ (and the exact math behind it)

We define the bundle planes
$$
g_i(p,x,t)=p^\top f(x,u^{(i)},t)+\ell(x,u^{(i)},t), \qquad i=1,\dots,m,
$$
and the log-sum-exp (soft-min) smoothing
$$
H_\delta(p,x,t)
=
-\delta \log\!\left(\sum_{i=1}^m \exp\!\left(-\frac{g_i(p,x,t)}{\delta}\right)\right).
$$

Let the soft-min weights be
$$
w_i(p,x,t;\delta)
=
\frac{\exp\!\left(-g_i(p,x,t)/\delta\right)}
{\sum_{j=1}^m \exp\!\left(-g_j(p,x,t)/\delta\right)},
\qquad
\sum_i w_i=1,\; w_i\ge 0.
$$

### Key identity (exact)

For each component $x_k$, the gradient of the smoothed Hamiltonian satisfies
$$
\frac{\partial H_\delta}{\partial x_k}(p,x,t)
=
\sum_{i=1}^m w_i(p,x,t;\delta)\;
\frac{\partial g_i}{\partial x_k}(p,x,t).
$$

Even though $w_i$ depends on $x$, this dependence is already accounted for in the log-sum-exp derivative, so there is no separate “$\partial w_i/\partial x_k$ term” in the final expression.

### What the code does (finite differences on $g_i$ + weighted sum)

1) Compute all $g_i(p,x,t)$ and the weights $w_i(p,x,t;\delta)$ once at the base point $(p,x,t)$.

2) For each coordinate $k$ (with step $\varepsilon$), form perturbed states
$x^+=x+\varepsilon e_k$ and $x^-=x-\varepsilon e_k$, and recompute each plane value:
$$
g_i^+ = g_i(p,x+\varepsilon e_k,t),
\qquad
g_i^- = g_i(p,x-\varepsilon e_k,t).
$$

3) Approximate the plane derivative by a central difference:
$$
\frac{\partial g_i}{\partial x_k}(p,x,t)
\approx
\frac{g_i^+-g_i^-}{2\varepsilon}.
$$

4) Combine using the exact identity:
$$
\frac{\partial H_\delta}{\partial x_k}(p,x,t)
\approx
\sum_{i=1}^m w_i(p,x,t;\delta)\;
\frac{g_i^+-g_i^-}{2\varepsilon}.
$$

Stacking these components for $k=1,\dots,n$ yields the vector $\nabla_x H_\delta(p,x,t)$.


Computational cost: approximating $\nabla_x H_\delta$ costs about $2mn$ extra evaluations of $f$ and $\ell$ (two-sided difference, across all $m$ planes, for each of the $n$ state components).

---

## 5) Why the solver needs this routine (where it plugs in)

This function is used in two core places.

### (a) Symplectic Euler TPBVP residual (`core/integrators.py`)

The discretization uses the PMP form
$$
\dot x = \nabla_p H_\delta(p,x,t),
\qquad
-\dot p = \nabla_x H_\delta(p,x,t),
$$
and a symplectic Euler update (one common choice) can be written as
$$
x_{i+1} = x_i + \Delta t_i\, \nabla_p H_\delta(p_i,x_i,t_i),
$$
$$
p_i = p_{i+1} + \Delta t_i\, \nabla_x H_\delta(p_{i+1},x_{i+1},t_{i+1}).
$$

That is why residual assembly calls:

- `grad_p` at $$(p_i,x_i,t_i)$$
- `grad_x` at $$(p_{i+1},x_{i+1},t_{i+1})$$

### (b) Adaptive indicators (`core/adaptivity.py`)

The adaptivity loop needs consistent evaluations of:

- gradients across time nodes (for a time discretization indicator),
- smoothed values $$H_\delta$$ (for smoothing-related checks).

Smoothing makes these indicators behave continuously under plane switching.

---

## 6) Practical numerical notes (what to watch)

- **δ is a “temperature”**: smaller $\delta$ makes the approximation closer to $$\bar H$$ but can lead to nearly one-hot weights, which may:
  - improve fidelity,
  - increase stiffness / ill-conditioning for Newton,
  - underflow without stabilization (handled here by shifting with $g_{\min}$).

- **Fixed finite difference step $$\varepsilon=10^{-6}$$** is simple but not scale-aware:
  - if components of $$x$$ have very different magnitudes, a relative step could be better,
  - the current choice prioritizes robustness with minimal user input.

- **Bundle must be non-empty**: if the PA bundle has no planes (no candidate controls), neither $$\bar H$$ nor $$H_\delta$$ is defined.

---

## 7) Summary: what `eval_H_smooth` means

`eval_H_smooth` computes a **smooth Hamiltonian** obtained by replacing the hard minimum over bundle planes with a **log-sum-exp soft-min**.

- $H_\delta$ approximates $\bar H$ with error controlled by $\delta\log m$.
- $\nabla_p H_\delta$ is a weighted blend of candidate dynamics.
- $\nabla_x H_\delta$ is computed as a weighted blend of finite-difference derivatives of plane values.

This is the computational bridge that lets a PA-bundle approximation (piecewise and switching) be used inside a **Newton-based** TPBVP solve on a **symplectic** discretization, while remaining stable near control switching.
