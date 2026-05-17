import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")

from experiments.ex6_nonsmoothham import (
    export_ex32_deep_report_artifacts,
    run_nonsmooth_solver,
)


class Example32DeepReportTests(unittest.TestCase):
    def test_ex32_report_export_writes_tables_figures_and_tex(self):
        result, prob = run_nonsmooth_solver(
            n_init=12,
            tol_time=5.0e-2,
            tol_PA=1.0e-1,
            tol_delta=1.0e-1,
            max_iters=2,
            delta0=0.05,
            store_iterates=True,
            verbose=False,
        )

        with TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "ex32_deep_report"
            artifacts = export_ex32_deep_report_artifacts(result, prob, out_dir=out_dir)

            report_path = Path(artifacts["report_tex"])
            summary_path = out_dir / "tables" / "final_summary.json"
            outer_history_path = out_dir / "tables" / "outer_loop_history.csv"
            input_path = out_dir / "tables" / "input_parameters.csv"
            reference_path = out_dir / "tables" / "reference_values.csv"
            history_figure = out_dir / "figures" / "error_indicator_history.pdf"
            iter0_dir = out_dir / "iterations" / "iter_00"

            self.assertTrue(report_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertTrue(outer_history_path.exists())
            self.assertTrue(input_path.exists())
            self.assertTrue(reference_path.exists())
            self.assertTrue(history_figure.exists())
            self.assertTrue((iter0_dir / "iter_00_state_costate.pdf").exists())
            self.assertTrue((iter0_dir / "iter_00_bundle_support_points.pdf").exists())
            self.assertTrue((iter0_dir / "iter_00_control.pdf").exists())
            self.assertTrue((iter0_dir / "iter_00_rho_density.pdf").exists())
            self.assertTrue((iter0_dir / "iter_00_mesh_and_indicator.pdf").exists())
            self.assertTrue((iter0_dir / "iter_00_pa_delta_contributions.pdf").exists())

            summary = json.loads(summary_path.read_text())
            self.assertIn("objective_mesh_approx", summary)
            self.assertIn("exact_objective_J", summary)
            self.assertIn("all_indicators_within_tolerance", summary)
            self.assertIn("final_action", summary)

            with outer_history_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertGreaterEqual(len(rows), 1)
            self.assertIn("iteration", rows[0])
            self.assertIn("eta_time_sum", rows[0])
            self.assertIn("action", rows[0])

            with input_path.open(newline="") as handle:
                input_rows = list(csv.DictReader(handle))
            parameter_names = {row["parameter"] for row in input_rows}
            self.assertIn("pa_add_fraction", parameter_names)
            self.assertIn("pa_time_separation_factor", parameter_names)
            self.assertIn("pa_gap_floor_ratio", parameter_names)
            self.assertIn("initial_guess_label", parameter_names)

            tex = report_path.read_text()
            self.assertIn("Legacy 2014 Example 3.2", tex)
            self.assertIn("Reference values", tex)
            self.assertIn("Error-estimate history", tex)
            self.assertIn(r"\subsection{Iteration 0}", tex)
            if any(entry.get("action") == "add_plane" for entry in result["log"]):
                self.assertIn("PA enrichment summary", tex)
                self.assertIn("Selected PA enrichment candidates", tex)


if __name__ == "__main__":
    unittest.main()
