import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest

import numpy as np

from experiments.full22 import codec
from experiments.full22.data import ROOT
from experiments.full22.records import Record, raw_record
from test_native_a22 import fixture, change_manifest

BINARY = ROOT/"target/packed-runtime-current/release/atom-quantizer"
PROBE = ROOT/"target/packed-runtime-current/release/examples/packed_probe"


class PackedRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(BINARY.is_file())
        self.assertTrue(PROBE.is_file())
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def call(self, archive, tensor, values, output, extra=()):
        source = self.root/"inputs.f32"
        source.write_bytes(np.asarray(values, dtype="<f4").tobytes())
        args = [str(BINARY), "apply-packed", str(archive), "--tensor", tensor,
                "--input", str(source), "--batch", str(len(values)), "--out", str(output), *extra]
        return subprocess.run(args, capture_output=True, text=True)

    def test_every_supported_width_rank_padding_and_raw_form_has_exact_outputs(self):
        rng = np.random.default_rng(102122)
        records = []
        for bits in (2, 3, 4, 6, 8):
            for rank in (0, 1, 2):
                config = {"bits": bits}
                if rank: config.update(rank=rank, rank_geometry="activation")
                records.append((f"b{bits}-r{rank}", codec.encode(rng.normal(size=(5, 35)).astype(np.float32),
                    rng.normal(size=(8, 35)).astype(np.float32), config)))
        zero = codec.encode(rng.normal(size=(5, 35)).astype(np.float32), config={"bits": 6})
        zero.arrays.update(lowrank_u=np.empty((5, 0), np.float16), lowrank_v=np.empty((0, 35), np.float16))
        zero.meta["lowrank_effective_rank"] = 0
        records.extend([("rank-zero-arrays", zero),
            ("raw-f32", raw_record(rng.normal(size=(5, 35)).astype(np.float32))),
            ("raw-bf16", raw_record(np.arange(175, dtype=np.float32).reshape(5, 35))),
            ("constant", Record({"kind": "constant", "shape": [5, 35]}, {"value": np.array([-0.0], np.float32)}))])
        archive = self.root/"models.a22"
        fixture(archive, records)
        values = rng.normal(size=(3, 35)).astype(np.float32)
        source = self.root/"inputs.f32"
        source.write_bytes(values.tobytes())
        for index, (name, record) in enumerate(records):
            with self.subTest(tensor=name):
                weight = codec.decode(record)
                expected = np.zeros((3, 5), np.float64)
                for col in range(35):
                    expected += values[:, col].astype(np.float64)[:, None] * weight[:, col].astype(np.float64)[None, :]
                output = self.root/f"out-{index}.f32"
                result = self.call(archive, name, values, output)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(output.read_bytes(), expected.astype("<f4").tobytes())
                receipt = json.loads(result.stdout)
                self.assertEqual(receipt["scope"], "selected_tensor")
                self.assertEqual(receipt["output_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
                directory = self.root/f"probe-{index}"
                tested = subprocess.run([str(PROBE), "operator", str(archive), name, str(source), "3", str(directory)], capture_output=True, text=True)
                self.assertEqual(tested.returncode, 0, tested.stderr)
                for mode in ("fused", "row", "dense_shared", "dense_existing"):
                    self.assertEqual((directory/f"{mode}.f32").read_bytes(), output.read_bytes())

    def test_bad_inputs_and_arguments_fail_without_output(self):
        archive = self.root/"model.a22"
        fixture(archive, [("matrix", raw_record(np.ones((3, 5), np.float32))), ("vector", raw_record(np.ones(5, np.float32)))])
        cases = [("matrix", np.ones((1, 4)), ()), ("matrix", np.full((1, 5), np.nan), ()),
                 ("matrix", np.full((1, 5), np.inf), ()), ("vector", np.ones((1, 5)), ()),
                 ("missing", np.ones((1, 5)), ()), ("matrix", np.ones((1, 5)), ("--batch", "1")),
                 ("matrix", np.ones((1, 5)), ("--unknown", "1"))]
        for index, (name, values, extra) in enumerate(cases):
            output = self.root/f"bad-{index}.f32"
            result = self.call(archive, name, values, output, extra)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob(".*.packed.tmp")))

    def test_existing_and_dangling_outputs_are_preserved(self):
        archive = self.root/"model.a22"
        fixture(archive, [("matrix", raw_record(np.ones((3, 5), np.float32)))])
        existing = self.root/"existing"
        existing.write_bytes(b"retain")
        link = self.root/"link"
        link.symlink_to(self.root/"missing-target")
        for output in (existing, link):
            self.assertNotEqual(self.call(archive, "matrix", np.ones((1, 5)), output).returncode, 0)
        self.assertEqual(existing.read_bytes(), b"retain")
        self.assertTrue(link.is_symlink())
        self.assertFalse((self.root/"missing-target").exists())

    def test_finite_inputs_that_overflow_the_output_are_rejected(self):
        archive = self.root/"overflow.a22"
        fixture(archive, [("matrix", raw_record(np.full((1, 2), np.finfo(np.float32).max, np.float32)))])
        output = self.root/"overflow.f32"
        result = self.call(archive, "matrix", np.ones((1, 2)), output)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("nonfinite output", result.stderr)
        self.assertFalse(output.exists())

    def test_commitment_failure_rejects_selected_tensor_and_full_model(self):
        archive = self.root/"model.a22"
        fixture(archive, [("first", raw_record(np.ones((3, 5), np.float32))), ("other", raw_record(np.ones((3, 5), np.float32)))])
        bad = self.root/"bad.a22"
        change_manifest(archive, bad, lambda m: m["tensors"][1].update(decoded_sha256="0"*64))
        good = self.call(bad, "first", np.ones((1, 5)), self.root/"selected.f32")
        self.assertEqual(good.returncode, 0, good.stderr)
        self.assertEqual(json.loads(good.stdout)["scope"], "selected_tensor")
        result = self.call(bad, "other", np.ones((1, 5)), self.root/"rejected.f32")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root/"rejected.f32").exists())
        verified = subprocess.run([str(BINARY), "verify", str(bad)], capture_output=True, text=True)
        self.assertNotEqual(verified.returncode, 0)

    def test_staging_collision_preserves_the_existing_staging_file(self):
        archive = self.root/"model.a22"
        fixture(archive, [("matrix", raw_record(np.ones((3, 5), np.float32)))])
        inputs = self.root/"inputs.f32"
        inputs.write_bytes(np.ones(5, np.float32).tobytes())
        output = self.root/"result.f32"
        command = [str(BINARY), "apply-packed", str(archive), "--tensor", "matrix", "--input", str(inputs),
                   "--batch", "1", "--out", str(output)]
        process = subprocess.Popen(["/bin/sh", "-c", 'kill -STOP $$; exec "$@"', "staging-control", *command],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            _, status = os.waitpid(process.pid, os.WUNTRACED)
            self.assertTrue(os.WIFSTOPPED(status))
            staged = self.root/f".{output.name}.{process.pid}-0.packed.tmp"
            staged.write_bytes(b"preserve staging owner")
            os.kill(process.pid, signal.SIGCONT)
            process.communicate(timeout=30)
            self.assertNotEqual(process.returncode, 0)
            self.assertFalse(output.exists())
            self.assertEqual(staged.read_bytes(), b"preserve staging owner")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


if __name__ == "__main__": unittest.main()
