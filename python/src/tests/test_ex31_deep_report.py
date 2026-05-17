import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")

from experiments.ex5_hypersensitive import (
    export_ex31_deep_report_artifacts,
    run_hypersensitive_solver,
)


class Example31DeepReportTests(unittest.TestCase):
    def test_ex31_report_export_writes_tables_figures_and_tex(self):
        result, prob = run_hypersensitive_solver(
            n_init=12,
            tol_time=5.0e-2,
            tol_PA=5.0e-2,
            tol_delta=5.0e-2,
            max_iters=1,
            delta0=0.02,
            store_iterates=True,
        )

        with TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "ex31_deep_report"
            artifacts = export_ex31_deep_report_artifacts(result, prob, out_dir=out_dir)

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
            self.assertTrue((iter0_dir / "iter_00_control.pdf").exists())
            self.assertTrue((iter0_dir / "iter_00_rho_density.pdf").exists())
            self.assertTrue((iter0_dir / "iter_00_mesh_and_indicator.pdf").exists())
            self.assertTrue((iter0_dir / "iter_00_pa_delta_contributions.pdf").exists())
            self.assertTrue((iter0_dir / "iter_00_bundle_support_points.pdf").exists())

            summary = json.loads(summary_path.read_text())
            self.assertIn("terminal_state", summary)
            self.assertIn("all_indicators_within_tolerance", summary)
            self.assertIn("final_action", summary)

            with outer_history_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertGreaterEqual(len(rows), 1)
            self.assertIn("iteration", rows[0])
            self.assertIn("action", rows[0])
            self.assertIn("all_indicators_within_tolerance", rows[0])

            tex = report_path.read_text()
            self.assertIn("Legacy 2014 Example 3.1", tex)
            self.assertIn("Input parameters", tex)
            self.assertIn("Reference values", tex)
            self.assertIn("Outer-loop summary", tex)
            self.assertIn("Error-estimate history", tex)
            self.assertIn(r"\subsection{Iteration 0}", tex)
            if any(entry.get("action") == "add_plane" for entry in result["log"]):
                self.assertIn("PA enrichment summary", tex)
                self.assertIn("Selected PA enrichment candidates", tex)


if __name__ == "__main__":
    unittest.main()
