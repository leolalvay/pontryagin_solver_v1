import unittest

import numpy as np

from experiments.ex8_goddard_fixedtime import (
    goddard_dynamic_pressure,
    make_goddard_pressure_feasible_guess,
)


class GoddardPressureFeasibleGuessTests(unittest.TestCase):
    def test_velocity_clip_makes_pressure_feasible_nodewise(self):
        X_guess = np.array(
            [
                [1.0, 0.0, 1.0],
                [1.0, 0.12, 0.9],
                [1.01, 0.11, 0.8],
            ],
            dtype=float,
        )
        q_max = 12.0
        adjusted = make_goddard_pressure_feasible_guess(
            X_guess,
            q_max=q_max,
            b=6200.0,
            beta=500.0,
            safety_factor=0.98,
        )

        self.assertTrue(np.allclose(adjusted[:, 0], X_guess[:, 0]))
        self.assertTrue(np.allclose(adjusted[:, 2], X_guess[:, 2]))
        for x in adjusted[1:]:
            q_val = goddard_dynamic_pressure(x[0], x[1], b=6200.0, beta=500.0)
            self.assertLessEqual(q_val, 0.98 * q_max + 1.0e-9)


if __name__ == "__main__":
    unittest.main()
