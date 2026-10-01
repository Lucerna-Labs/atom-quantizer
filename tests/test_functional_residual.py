import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import gguf
import numpy as np

from experiments.full22 import archive, codec, harness, residual
from experiments.full22.data import load_calibration
from experiments.full22.records import Record
from test_full22_integrity import fixture


class FunctionalResidualTests(unittest.TestCase):
    def test_stored_factors_follow_the_operator_svd_identity(self):
        rng = np.random.default_rng(17)
        error, inputs = rng.normal(size=(12, 9)), rng.normal(size=(7, 9))
        singular = np.linalg.svd(error @ inputs.T, compute_uv=False)
        left, right = residual.functional_factors(error, inputs, 2)
        remaining = np.linalg.norm((error-left.astype(float)@right.astype(float)) @ inputs.T)**2
        expected = np.sum(singular[2:]**2)
        self.assertLess(abs(remaining-expected), expected*1e-3)
        self.assertEqual(left.nbytes+right.nbytes, 84)

    def test_zero_measured_rank_does_not_invent_nullspace_corrections(self):
        error = np.array([[0., 1.], [0., 2.]])
        inputs = np.array([[1., 0.], [2., 0.]])
        left, right = residual.functional_factors(error, inputs, 2)
        self.assertEqual((left.shape, right.shape), ((2, 0), (0, 2)))
        np.testing.assert_array_equal(left.astype(float)@right.astype(float), np.zeros((2, 2)))

    def test_shuffled_features_do_not_receive_credit_for_correct_geometry(self):
        error, inputs = np.diag([1., 2., 3.]), np.diag([8., 1., 0.5])
        left, right = residual.functional_factors(error, inputs, 1)
        wrong_left, wrong_right = residual.functional_factors(error, inputs, 1, shuffled=True)
        aligned = np.linalg.norm((error-left.astype(float)@right.astype(float))@inputs.T)**2
        shuffled = np.linalg.norm((error-wrong_left.astype(float)@wrong_right.astype(float))@inputs.T)**2
        self.assertLess(aligned, shuffled)

    def test_serialized_correction_preserves_an_observed_rank_one_operator(self):
        weights = np.random.default_rng(91).normal(size=(16, 32)).astype(np.float32)
        inputs = np.tile(np.linspace(-1, 1, 32, dtype=np.float32), (8, 1))
        base = codec.decode(codec.encode(weights, inputs, {"bits": 3}))
        record = codec.encode(weights, inputs, {"bits": 3, "rank": 1, "rank_geometry": "activation"})
        reconstructed = codec.decode(Record.loads(record.dumps(True)))
        before = codec.distortion(weights, base, inputs)["output_nmse"]
        after = codec.distortion(weights, reconstructed, inputs)["output_nmse"]
        self.assertLess(after, before*1e-4)
        np.testing.assert_array_equal(reconstructed, codec.decode(record.roundtrip(True)))

    def test_invalid_inputs_and_decoder_factor_state_are_rejected(self):
        for inputs, rank in [(None, 1), (np.eye(3), 1), (np.full((2, 2), np.nan), 1),
                             (np.eye(2), -1), (np.eye(2), True)]:
            with self.subTest(rank=rank), self.assertRaises(ValueError):
                residual.functional_factors(np.eye(2), inputs, rank)
        values = np.eye(2, dtype=np.float32)
        for arrays in [{"lowrank_u": values}, {"lowrank_u": values, "lowrank_v": np.ones((3, 2))},
                       {"lowrank_u": np.full((2, 2), np.inf), "lowrank_v": values}]:
            with self.assertRaises(ValueError):
                residual.add_stored(values, arrays, {})

    def test_complete_archive_needs_neither_original_weights_nor_calibration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, names, weights = fixture(root)
            samples = load_calibration(root/"fit.acal", source)
            config = {"bits": 3, "rank": 2, "rank_geometry": "activation"}
            expected = [codec.decode(codec.encode(w, samples[name], config)) for name,w in zip(names,weights)]
            with contextlib.redirect_stdout(io.StringIO()):
                harness.build_model(config, root/"model.a22", source, root/"fit.acal")
            source.unlink()
            (root/"fit.acal").unlink()
            archive.export_model(root/"model.a22", root/"decoded.gguf")
            actual = gguf.GGUFReader(root/"decoded.gguf").tensors
            for tensor, values in zip(actual, expected):
                np.testing.assert_array_equal(tensor.data, values)


if __name__ == "__main__":
    unittest.main()
