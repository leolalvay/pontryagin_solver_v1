import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.ex8_goddard_fixedtime import (
    export_goddard_deep_report_artifacts,
    run_goddard_solver,
)


class GoddardDeepReportTests(unittest.TestCase):
    def test_export_goddard_deep_report_artifacts(self):
        result, prob = run_goddard_solver(
            T=0.15,
            q_max=12.75,
            rho_m=1.0e4,
            n_init=8,
            tol_time=1.0e-3,
            tol_PA=1.0e-3,
            tol_delta=1.0e-3,
            max_iters=2,
            delta0=5.0e-2,
            store_iterates=True,
            verbose=False,
        )

        with TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "goddard_deep_report"
            artifacts = export_goddard_deep_report_artifacts(result, prob, out_dir)
            self.assertTrue((out_dir / "report.tex").exists())
            self.assertTrue((out_dir / "tables" / "final_summary.json").exists())
            self.assertTrue((out_dir / "tables" / "input_parameters.csv").exists())
            self.assertTrue((out_dir / "figures" / "error_indicator_history.pdf").exists())
            self.assertTrue((out_dir / "iterations" / "iter_00" / "iter_00_state_costate.pdf").exists())
            self.assertTrue((out_dir / "iterations" / "iter_00" / "iter_00_control.pdf").exists())
            self.assertTrue((out_dir / "iterations" / "iter_00" / "iter_00_pressure.pdf").exists())
            self.assertEqual(Path(artifacts["report_tex"]), out_dir / "report.tex")


if __name__ == "__main__":
    unittest.main()
