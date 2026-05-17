# Legacy 2014 Reproduction

This note describes how to rerun the executable comparisons against the
legacy manuscript stored at
`/Users/temponrf/Dropbox/2026/MANUSCRIPTS/2025_Pontryagin_adaptive_OC/1407.8330v1.pdf`.

## Executable legacy examples

The current archive Python code contains runnable implementations for:

- Legacy Example 3.1: `experiments.ex5_hypersensitive`
- Legacy Example 3.2: `experiments.ex6_nonsmoothham`
- Legacy Example 3.3: `experiments.ex7_singular`

## Run the regression tests

From `archive_python/python/src`:

```bash
MPLBACKEND=Agg \
MPLCONFIGDIR="/Users/temponrf/Documents/Codex/2026-04-23-files-mentioned-by-the-user-frozen/.mplconfig" \
PYTHONPATH="/Users/temponrf/Documents/Codex/2026-04-23-files-mentioned-by-the-user-frozen/.python_vendor:/Users/temponrf/Documents/Codex/2026-04-23-files-mentioned-by-the-user-frozen/archive_python/python/src" \
/Users/temponrf/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
-m unittest tests.test_manuscript_consistency tests.test_legacy_2014_examples
```

The legacy regression checks live in:

- `archive_python/python/src/tests/test_legacy_2014_examples.py`

## Export plots and tables

From `archive_python/python/src`:

```bash
MPLBACKEND=Agg \
MPLCONFIGDIR="/Users/temponrf/Documents/Codex/2026-04-23-files-mentioned-by-the-user-frozen/.mplconfig" \
PYTHONPATH="/Users/temponrf/Documents/Codex/2026-04-23-files-mentioned-by-the-user-frozen/.python_vendor:/Users/temponrf/Documents/Codex/2026-04-23-files-mentioned-by-the-user-frozen/archive_python/python/src" \
/Users/temponrf/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
experiments/export_legacy_2014_results.py
```

This writes:

- Machine-readable summaries:
  - `archive_runs/legacy_2014/summary.json`
  - `archive_runs/legacy_2014/summary.csv`
- Per-example run data:
  - `archive_runs/legacy_2014/ex31_hypersensitive`
  - `archive_runs/legacy_2014/ex32_nonsmooth`
  - `archive_runs/legacy_2014/ex33_singular`
- Per-example figures:
  - `figures/legacy_2014/ex31_hypersensitive`
  - `figures/legacy_2014/ex32_nonsmooth`
  - `figures/legacy_2014/ex33_singular`

## Current interpretation

- Legacy Example 3.1 is executable and matches the legacy problem
  definition, but the current adaptive loop does not satisfy the
  PA/smoothing stopping criteria within the present outer-iteration cap.
- Legacy Example 3.2 is executable and reproduces the nonsmooth problem
  qualitatively well; the objective is close to the exact value, but the
  final Newton solve becomes stressed at very small smoothing levels.
- Legacy Example 3.3 is now executable in explicit-gradient mode; the
  current archive run drives the objective to a small value and clusters
  the mesh strongly near the regularized singularity, but the strict
  time-indicator stopping criterion is not yet satisfied under the
  present outer-iteration cap.
