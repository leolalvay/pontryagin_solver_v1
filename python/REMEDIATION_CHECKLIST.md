# Python Archive Remediation Checklist

This checklist is ordered by manuscript-consistency risk, not by coding convenience.

## Priority 1: Solver contracts that must match the manuscript

- [x] Make restricted Hamiltonian evaluation strict.
  - `compute_H(..., restricted=True)` must not silently fall back to the unrestricted minimum when no viable control exists.
  - Current status: patched in `src/core/hamiltonian.py`.

- [x] Restore manuscript-aligned outer-loop defaults in the reusable solver path.
  - `s_time = 0.5`
  - `K_time = 1e-6`
  - Newton defaults `tol = 1e-10`, `max_iter = 50`
  - Current status: patched in `src/core/adaptivity.py` and `src/core/newton.py`.

- [x] Make outer-loop logs reproducible enough for manuscript tables.
  - Log the chosen action at each outer iteration.
  - Keep `rho`, `rho_bar`, `r_bar`, and `t_nodes_iter` on the final resolve entry.
  - Echo the effective solver settings in the result dictionary.
  - Current status: patched in `src/core/adaptivity.py`.

## Priority 2: Keep Example 2 variants explicit instead of conflated

- [x] Preserve the archived fixed-horizon constrained variant for comparison.
  - Name: `archive_fixed_box`
  - Meaning: fixed `T=2`, running cost `1 + 1e-2 u^2`, penalty weight `100`, state constraint `x1 <= 0`

- [x] Preserve the archived fixed-horizon unconstrained variant for comparison.
  - Name: `archive_fixed_unconstrained`
  - Meaning: same as above, but with no state-space restriction

- [x] Declare the manuscript target variant separately.
  - Name: `manuscript_tau_box`
  - Meaning: tau-augmented normalized-time Section 4.2 problem with `K=[-2,2]^2`, `rho = 1e4`
  - Current status: configuration is present, but execution still raises `NotImplementedError` until the tau-augmented solver path exists.

## Priority 3: Replace archive heuristics with manuscript algorithms

- [ ] Implement the tau-augmented unknown vector and residual/Jacobian path for `manuscript_tau_box`.
  - Add scalar `tau` as an unknown.
  - Solve on normalized time `s in [0,1]`.
  - Enforce the tau transversality condition in the residual system.
  - Report both normalized-time and physical-time quantities.

- [ ] Separate manuscript bundle logic from generic archive heuristics.
  - Bootstrap and plane-addition steps should be named and implemented to match the manuscript pseudocode.
  - Remove hidden tuning constants from inline code paths when they belong in the example specification.

- [ ] Make the indicator policy explicit for each example.
  - Fixed-horizon examples should use the standard policy.
  - Tau-augmented Example 2 should be marked as an extension path with its own documented scope note.
  - State-constrained examples should document exactly when restricted Hamiltonians are used.

## Priority 4: Eliminate structural mismatches with the corrected solver core

- [ ] Replace the legacy unknown-vector layout that still carries `p_0` as a primary Newton unknown.
  - Current archive layout: `x_1,...,x_N,p_0,...,p_N`
  - Desired direction: align with the corrected full-space fixed-horizon core used in the manuscript-oriented rebuild.

- [ ] Stop relying on finite-difference gradients where the manuscript implementation expects exact derivatives.
  - This is especially important for comparisons across languages and for cleaner Jacobian structure tests.

## Priority 5: Reproducibility and reporting

- [ ] Add dedicated unit tests for the tau-augmented Example 2 residual size, switch-time reporting, and terminal residual behavior.
- [ ] Add cross-variant comparison tests so `archive_fixed_box` and `archive_fixed_unconstrained` can be run side-by-side without changing code.
- [ ] Emit machine-readable summary files directly from the archive example runners instead of reconstructing them ad hoc.

## Tests added in this patch set

- `tests/test_manuscript_consistency.py`
  - strict restricted-Hamiltonian semantics
  - final-resolve log completeness
  - Example 2 manuscript/comparison variant preservation
