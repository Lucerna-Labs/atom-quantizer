"""Actual native resource and self-contained archive boundaries for B03."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import gguf
import numpy as np

from experiments.resource_discovery import measure_fields, native_resources
from experiments.full22 import archive, codec, harness, transforms
from experiments.full22.data import load_calibration
from test_full22_integrity import fixture


class ResourceBoundaryTests(unittest.TestCase):
    def test_block_measurements_and_native_resource_bounds(self):
        weights = np.ones((4, 64), np.float32)
        inputs = np.tile(np.arange(1, 65, dtype=np.float32), (4, 1))
        signals, coupling, folded, _ = measure_fields(weights, inputs)
        np.testing.assert_array_equal(signals.reshape(-1), inputs[0])
        self.assertFalse(np.array_equal(signals[0], signals[1]))
        self.assertEqual(folded.shape, (32,))
        resources, diagnostics = native_resources(signals, coupling, 0, 1, 0)
        self.assertTrue(np.all((resources >= 1/3) & (resources <= 1)))
        self.assertLess(diagnostics[:, 0].max(), 1e-10)
        identity, _ = native_resources(signals, coupling, 1, 3, 510)
        np.testing.assert_array_equal(identity, np.ones(64))

    def test_scales_preserve_operator_coordinates_and_reject_invalid_values(self):
        w = np.random.default_rng(11).normal(size=(4, 32)).astype(np.float32)
        x = np.random.default_rng(12).normal(size=(8, 32)).astype(np.float32)
        scale = np.linspace(1/3, 3, 32).astype(np.float32)
        np.testing.assert_allclose((w*scale) @ (x/scale).T, w @ x.T, rtol=1e-5, atol=4e-6)
        for invalid in [np.zeros(32), np.full(32, np.nan), np.ones(16), np.full(32, 5.)]:
            with self.assertRaises(ValueError):
                transforms.checked_native_scale(invalid, 32)

    def test_native_resource_archive_is_independent_of_source_files(self):
        for readout in ["conductance", "resistance"]:
            with self.subTest(readout=readout), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source, names, weights = fixture(root)
                samples = load_calibration(root / "fit.acal", source)
                signal, coupling, _, _ = measure_fields(weights[0], samples[names[0]])
                resource, _ = native_resources(signal, coupling, 1, 2, 0)
                scale = (resource if readout == "conductance" else 1/resource).astype(np.float32)
                scale_path = root / "scale.npy"
                np.save(scale_path, scale)
                config = {"bits": 4, "native_scale_path": str(scale_path)}
                expected = [codec.decode(codec.encode(w, samples[name], config)) for name, w in zip(names, weights)]
                with contextlib.redirect_stdout(io.StringIO()):
                    harness.build_model(config, root/"native.a22", source, root/"fit.acal")
                scale_path.unlink()
                source.unlink()
                (root/"fit.acal").unlink()
                archive.export_model(root/"native.a22", root/"decoded.gguf")
                actual = gguf.GGUFReader(root/"decoded.gguf").tensors
                for tensor, values in zip(actual, expected):
                    np.testing.assert_array_equal(tensor.data, values)


if __name__ == "__main__":
    unittest.main()
