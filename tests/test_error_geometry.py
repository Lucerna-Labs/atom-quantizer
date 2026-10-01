from pathlib import Path
import tempfile
import unittest
import numpy as np
from experiments.error_geometry import functional_error_profile, save_new


class ErrorGeometryTests(unittest.TestCase):
    def test_cancellation_and_reinforcement_are_distinguished(self):
        w = np.array([[1., 0.]], np.float32)
        x = np.array([[1., 1.], [-1., -1.]], np.float32)
        cancel = functional_error_profile(w, np.array([[0., 1.]]), x, x, block=1)
        reinforce = functional_error_profile(w, np.array([[0., -1.]]), x, x, block=1)
        self.assertEqual(cancel["diagonal_error_prediction"], 2)
        self.assertEqual(cancel["selection_output_error_energy"], 0)
        self.assertEqual(cancel["correlation_cross_term"], -2)
        self.assertEqual(reinforce["selection_output_error_energy"], 4)
        self.assertEqual(reinforce["correlation_cross_term"], 2)
        self.assertEqual(reinforce["cross_block_interference"], 2)

    def test_fit_directions_do_not_claim_unseen_selection_error(self):
        w, q = np.eye(2), np.zeros((2, 2))
        fit, selection = np.array([[1., 0.]]), np.array([[0., 1.]])
        profile = functional_error_profile(w, q, fit, selection, block=1)
        first = profile["shared_error_modes"][0]
        self.assertEqual(first["fit_error_energy_fraction"], 1)
        self.assertEqual(first["selection_error_energy_fraction"], 0)

    def test_rank_one_error_is_detected_and_zero_error_is_defined(self):
        w = np.array([[1., 2.], [2., 4.]], np.float32)
        x = np.eye(2, dtype=np.float32)
        profile = functional_error_profile(w, np.zeros_like(w), x, x)
        self.assertEqual(profile["fit_error_rank"], 1)
        self.assertAlmostEqual(profile["shared_error_modes"][0]["selection_error_energy_fraction"], 1)
        exact = functional_error_profile(w, w, x, x)
        self.assertEqual(exact["selection_output_error_energy"], 0)
        self.assertIsNone(exact["output_to_diagonal_ratio"])

    def test_invalid_inputs_and_existing_destinations_fail_closed(self):
        w, x = np.ones((2, 2)), np.eye(2)
        for bad in [np.zeros((3, 2)), np.full((2, 2), np.nan)]:
            with self.assertRaises(ValueError):
                functional_error_profile(w, bad, x, x)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"result.json"
            path.write_text("keep me")
            with self.assertRaises(ValueError):
                save_new(path, {})
            self.assertEqual(path.read_text(), "keep me")

    def test_finite_inputs_that_overflow_the_operator_are_rejected(self):
        huge = np.full((1, 1), np.finfo(np.float32).max, np.float32)
        with self.assertRaisesRegex(ValueError, "nonfinite operator"):
            functional_error_profile(huge, np.zeros((1, 1)), huge, huge)


if __name__ == "__main__":
    unittest.main()
