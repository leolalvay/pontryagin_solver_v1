# Adaptive Pontryagin Solver (Python)

This repository implements a generic solver for optimal control problems based on
Pontryagin's Maximum Principle (PMP) with an adaptive discretisation
and smoothing strategy.  The solver is designed to handle problems with
nonsmooth Hamiltonians through a piecewise‑affine (PA) bundle surrogate and
a log‑sum‑exp smoothing function.  An outer adaptivity loop refines the time
mesh, adds new controls to the bundle, and reduces the smoothing parameter
until a posteriori error indicators fall below user‑specified tolerances.

## Repository Structure

```
python/
  README.md            # this file
  docs/                # .md files documenting functions and files in the repo.
    core/...
    experiments/...
  literature/
    error_estimate_optimal_controlo_ode.pdf  # Karlsson et al. article
    pontryagin_pmp_smoothed_adaptive_....pdf # Tempone Pontryagin solver working draft
    Pontryagin_Solver_Documentation.pdf      # Tempone solver documentation (Not Updated)
    slides/...                               # Slides of Regular Meetings 
  src/
    benchmarks
    core/
      problem.py       # problem definition (dynamics, cost, constraints)
      pa_bundle.py     # PA bundle surrogate (control candidates)
      hamiltonian.py   # Hamiltonian evaluation routines
      smoothing.py     # smoothed Hamiltonian and gradients via log-sum-exp
      integrators.py   # symplectic Euler residual/Jacobian assembly
      shooting.py      # wrapper for residual/Jacobian using packed vectors
      newton.py        # damped Newton solver with Armijo line search
      adaptivity.py    # adaptive outer loop implementing Algorithm 3.1
      constraints.py   # clipping utilities for states and controls
      utils.py         # miscellaneous helper functions
    experiments/
      ex1_lqr.py       # Example 1: LQR problem
      ex2_double_integrator.py  # Example 2: minimum‑time double integrator
      ex3_dubins.py    # Example 3: Dubins car
      ex4_simpleint.py # Example 4: Simple Integration OC Problem
      ex5_hypersensitive.py # Example 5: Hypersensitive Problem
      ex6_nonsmoothham.py # Example 6: Problem with nonsmooth Hamiltonian
    figures
    logs
    tests
      ex5_hypersensitive_test.py
      ex6_nonsmoothham_test.py  
```

## Experiments usage

All dependencies are part of the Python standard library except for
NumPy, which is required.  To run an example from the command line:

```bash
cd python/src
python3 -m experiments.ex1_lqr
python3 -m experiments.ex2_double_integrator
python3 -m experiments.ex3_dubins
python3 -m experiments.ex4_simpleint
python3 -m experiments.ex5_hypersensitive
python3 -m experiments.ex6_nonsmoothham
```

Each example constructs an `OCPProblem`, sets up an initial time mesh and
tolerances, calls the adaptive solver, and prints a summary of the solution
including the number of time nodes, number of control planes in the bundle,
and the history of error indicators per outer iteration.

## Tests usage

To run a test from the command line:

```bash
cd python/src
python3 -m tests.ex5_hypersensitive_test
python3 -m tests.ex6_nonsmoothham_test
```

Currently status: A test can either 
- use the solver structure -> constructs an `OCPProblem`, sets up an initial time mesh and
tolerances, use flags setting explicit hamitonian and its gradients and prints a summary of the solution 
- Or use a self contained structure for the test.

## Notes

- This implementation follows a version control divided by branches as follows
```
main
  pa-init-hyper-search
  pa-init-hyper-test
  pa-init-hypersensitive
  paper-time-adapt  
```
- The `main` branch is defined as the deliverable implementation (branch to monitor for updates) and the secondary branches contain modifications of the main branch that could or could not be integrated later in the main branch.

