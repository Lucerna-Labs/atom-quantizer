"""Test the native-to-codec boundary independently of the discovery score."""
from pathlib import Path
import contextlib
import io
import tempfile
import unittest

import gguf
import numpy as np

from experiments.basis_discovery import native_basis
from experiments.local_basis_discovery import measure_inputs, native_basis as local_basis
from experiments.full22 import archive, codec, harness, transforms
from experiments.full22.data import load_calibration
from experiments.full22.records import Record
from test_full22_integrity import fixture


class BasisBoundaryTests(unittest.TestCase):
    def test_local_input_measurements_and_native_boundary(self):
        weights = np.ones((4, 32), np.float32)
        inputs = np.zeros((4, 32), np.float32)
        inputs[:, 0] = [1, -1, 2, -2]
        inputs[:, 1] = [2, -2, 1, -1]
        signal, coupling = measure_inputs(weights, inputs)
        self.assertAlmostEqual(coupling[0, 1], 0.8)
        np.testing.assert_array_equal(coupling[2], np.zeros(32))
        matrix, diagnostics = local_basis(signal, coupling, 0, 1, 0)
        self.assertLess(diagnostics[0], 1e-10)
        restored = transforms.basis_apply(transforms.basis_apply(weights, matrix), matrix, True)
        self.assertLess(codec.distortion(weights, restored)["weight_nmse"], 1e-11)
        identity, _ = local_basis(signal, coupling, 1, 3, 510)
        np.testing.assert_array_equal(identity, np.eye(32))

    def test_native_matrix_matches_forward_inverse_and_operator_coordinates(self):
        matrix, diagnostics = native_basis(np.arange(1, 33, dtype=float), 0, 1, 0)
        self.assertLess(diagnostics[0], 1e-10)
        rng = np.random.default_rng(31)
        weights = rng.normal(size=(4, 64)).astype(np.float32)
        inputs = rng.normal(size=(8, 64)).astype(np.float32)
        transformed = transforms.basis_apply(weights, matrix)
        restored = transforms.basis_apply(transformed, matrix, True)
        self.assertLess(codec.distortion(weights, restored)["weight_nmse"], 1e-11)
        np.testing.assert_allclose(transformed @ transforms.basis_apply(inputs, matrix).T,
                                   weights @ inputs.T, rtol=2e-5, atol=4e-6)

    def test_saved_record_decodes_after_native_basis_file_is_removed(self):
        matrix, _ = native_basis(np.arange(1, 33, dtype=float), 1, 2, 0)
        weights = np.random.default_rng(12).normal(size=(4, 64)).astype(np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "basis.npy"
            np.save(path, matrix)
            record = codec.encode(weights, config={"bits": 4, "basis_path": str(path)})
            data = record.dumps(True)
            expected = codec.decode(record)
            path.unlink()
            loaded = Record.loads(data)
            np.testing.assert_array_equal(expected, codec.decode(loaded))
            self.assertEqual(loaded.arrays["pre_basis"].nbytes, 32 * 32 * 4)

    def test_invalid_transport_is_rejected_without_repair(self):
        weight = np.ones((2, 32), np.float32)
        for matrix in [np.eye(32, dtype=np.float32) * 2, np.full((32, 32), np.nan), np.eye(16)]:
            with self.subTest(shape=matrix.shape), self.assertRaises(ValueError):
                transforms.basis_apply(weight, matrix)

    def test_local_basis_full_archive_decodes_without_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, names, weights = fixture(root)
            samples = load_calibration(root / "fit.acal", source)
            signal, coupling = measure_inputs(weights[0], samples[names[0]])
            matrix, _ = local_basis(signal, coupling, 0, 2, 0)
            basis_path = root / "basis.npy"
            np.save(basis_path, matrix)
            config = {"bits": 4, "basis_path": str(basis_path)}
            expected = [codec.decode(codec.encode(w, samples[name], config)) for name, w in zip(names, weights)]
            model, destination = root / "native.a22", root / "decoded.gguf"
            with contextlib.redirect_stdout(io.StringIO()):
                harness.build_model(config, model, source, root / "fit.acal")
            basis_path.unlink()
            source.unlink()
            (root / "fit.acal").unlink()
            archive.export_model(model, destination)
            tensors = gguf.GGUFReader(destination).tensors
            self.assertEqual([t.name for t in tensors], names)
            for tensor, values in zip(tensors, expected):
                np.testing.assert_array_equal(tensor.data, values)


if __name__ == "__main__":
    unittest.main()
