import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")

from experiments.ex7_singular import export_ex33_deep_report_artifacts, run_singular_solver


class Example33DeepReportTests(unittest.TestCase):
    def test_ex33_report_export_writes_tables_figures_and_tex(self):
        result, prob = run_singular_solver(
            n_init=16,
            tol_time=5.0e-3,
            tol_PA=5.0e-3,
            tol_delta=5.0e-3,
            max_iters=3,
            delta0=1.0e-2,
            store_iterates=True,
            verbose=False,
        )

        with TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "ex33_deep_report"
            artifacts = export_ex33_deep_report_artifacts(result, prob, out_dir)

            report_path = Path(artifacts["report_tex"])
            summary_path = out_dir / "tables" / "final_summary.json"
            input_path = out_dir / "tables" / "input_parameters.csv"
            history_path = out_dir / "tables" / "outer_loop_history.csv"
            history_figure = out_dir / "figures" / "error_indicator_history.pdf"
            iter0_dir = out_dir / "iterations" / "iter_00"

            self.assertTrue(report_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertTrue(input_path.exists())
            self.assertTrue(history_path.exists())
            self.assertTrue(history_figure.exists())
            self.assertTrue((iter0_dir / "iter_00_state_costate.pdf").exists())
            self.assertTrue((iter0_dir / "iter_00_control.pdf").exists())
            self.assertTrue((iter0_dir / "iter_00_rho_density.pdf").exists())
            self.assertTrue((iter0_dir / "iter_00_mesh_and_indicator.pdf").exists())

            summary = json.loads(summary_path.read_text())
            self.assertIn("objective_mesh_approx", summary)
            self.assertIn("true_objective", summary)
            self.assertIn("max_state_error", summary)

            with input_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            params = {row["parameter"] for row in rows}
            self.assertIn("epsilon", params)
            self.assertIn("beta", params)
            self.assertIn("t0", params)
            self.assertIn("use_explicit_hamiltonian_gradients", params)

            tex = report_path.read_text()
            self.assertIn("Legacy 2014 Example 3.3", tex)
            self.assertIn("Singular Tracking Problem", tex)
            self.assertIn(r"\subsection{Iteration 0}", tex)


if __name__ == "__main__":
    unittest.main()
