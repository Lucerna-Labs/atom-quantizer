import contextlib
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import gguf
import numpy as np

from experiments.full22 import archive, codec, harness
from experiments.full22.data import load_calibration, sha256
from test_full22_integrity import fixture


class CalibrationRoleTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(703)
        self.weight = rng.normal(size=(16, 32)).astype(np.float32)
        self.base = rng.normal(size=(8, 32)).astype(np.float32)
        self.correction = rng.normal(size=(8, 32)).astype(np.float32)
        self.config = {"bits": 3, "rank": 2, "rank_geometry": "activation"}

    def test_same_role_reproduces_default_bytes_and_separate_role_changes_only_factors(self):
        old = codec.encode(self.weight, self.base, self.config)
        explicit = codec.encode(self.weight, self.base, self.config, residual_inputs=self.base)
        separate = codec.encode(self.weight, self.base, self.config, residual_inputs=self.correction)
        scalar = codec.encode(self.weight, self.base, {"bits": 3})
        self.assertEqual(old.dumps(), explicit.dumps())
        for key, value in scalar.arrays.items():
            np.testing.assert_array_equal(value, separate.arrays[key])
        self.assertFalse(np.array_equal(old.arrays["lowrank_v"], separate.arrays["lowrank_v"]))
        reverse = codec.encode(self.weight, self.correction, self.config, residual_inputs=self.base)
        self.assertFalse(np.array_equal(separate.arrays["lowrank_v"], reverse.arrays["lowrank_v"]))

    def test_explicit_invalid_observations_cannot_fall_back(self):
        for observations in (None, np.ones((3, 31)), np.empty((0, 32)),
                             np.full((8, 32), np.nan), self.base.astype(complex)):
            with self.subTest(shape=np.shape(observations)), self.assertRaises(ValueError):
                codec.encode(self.weight, self.base, self.config, residual_inputs=observations)
        with self.assertRaises(ValueError):
            codec.encode(self.weight, self.base, {"bits": 3}, residual_inputs=self.correction)

    def test_archive_binds_both_roles_and_decodes_without_fitting_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, names, weights = fixture(root)
            fit, correction = root/"fit.acal", root/"selection.acal"
            base, extra = load_calibration(fit, source), load_calibration(correction, source)
            expected = [codec.decode(codec.encode(w, base[n], self.config, residual_inputs=extra[n]))
                        for n, w in zip(names, weights)]
            with contextlib.redirect_stdout(io.StringIO()):
                harness.build_model(self.config, root/"model.a22", source, fit, residual_calibration=correction)
            with zipfile.ZipFile(root/"model.a22") as saved:
                manifest = json.loads(saved.read("manifest.json"))
                self.assertEqual(manifest["calibration_sha256"], sha256(fit))
                self.assertEqual(manifest["residual_calibration_sha256"], sha256(correction))
            for path in (source, fit, correction):
                path.unlink()
            archive.export_model(root/"model.a22", root/"decoded.gguf")
            for tensor, values in zip(gguf.GGUFReader(root/"decoded.gguf").tensors, expected):
                np.testing.assert_array_equal(tensor.data, values)

    def test_wrong_source_and_missing_operator_fail_without_publication(self):
        for mode in ("source", "missing"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source, _, _ = fixture(root)
                path = root/"selection.acal"
                data = path.read_bytes()
                if mode == "source":
                    data = data[:4] + bytes(32) + data[36:-32]
                else:
                    data = data[:68] + struct.pack("<I", 0)
                path.write_bytes(data + hashlib.sha256(data).digest())
                with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
                    harness.build_model(self.config, root/"bad.a22", source, root/"fit.acal", residual_calibration=path)
                self.assertFalse((root/"bad.a22").exists())

    def test_changed_role_file_is_rejected_before_publication(self):
        for role in ("fit.acal", "selection.acal"):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source, _, _ = fixture(root)
                actual_encode = codec.encode
                def mutate(*args, **kwargs):
                    record = actual_encode(*args, **kwargs)
                    with (root/role).open("ab") as file:
                        file.write(b"changed")
                    return record
                with patch.object(codec, "encode", side_effect=mutate), self.assertRaisesRegex(ValueError, "calibration changed"):
                    harness.build_model(self.config, root/"bad.a22", source, root/"fit.acal",
                                        residual_calibration=root/"selection.acal")
                self.assertFalse((root/"bad.a22").exists())


if __name__ == "__main__":
    unittest.main()
