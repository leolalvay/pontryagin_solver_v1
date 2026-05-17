import unittest

from experiments.ex8_goddard_fixedtime import run_goddard_qmax_continuation


class GoddardContinuationTests(unittest.TestCase):
    def test_qmax_continuation_returns_stage_log(self):
        continuation = run_goddard_qmax_continuation(
            q_schedule=[100.0, 40.0],
            T=0.15,
            rho_m=1.0e4,
            n_init=8,
            tol_time=1.0e-4,
            tol_PA=1.0e-4,
            tol_delta=1.0e-4,
            delta0=5.0e-2,
            stage_max_iters=1,
            verbose=False,
        )
        self.assertEqual(len(continuation["stages"]), 2)
        self.assertEqual(continuation["stages"][0]["status"], "ok")
        self.assertIsNotNone(continuation["last_successful_result"])


if __name__ == "__main__":
    unittest.main()
