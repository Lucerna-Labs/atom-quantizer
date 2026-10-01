"""Exercise real records, cache resumes and GGUF publication on disposable inputs."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from experiments.full22 import archive, codec, harness
from experiments.full22.records import Record, raw_record
from experiments.full22.screen_cache import ScreenCache


def digest(data):
    return hashlib.sha256(data).hexdigest()


def frame(header, body=b""):
    header = json.dumps(header, separators=(",", ":")).encode()
    data = b"AR01" + struct.pack("<I", len(header)) + header + body
    return data + hashlib.sha256(data).digest()


def fixture(root):
    """Two F32 tensors and separate, source-bound synthetic calibration files."""
    names = ["blk.0.attn_q.weight", "blk.1.attn_q.weight"]
    tensors = [np.random.default_rng(i).normal(size=(4, 32)).astype("<f4") for i in range(2)]
    prefix = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 2, 0))
    for index, name in enumerate(names):
        prefix.extend(struct.pack("<Q", len(name)) + name.encode())
        prefix.extend(struct.pack("<IQQIQ", 2, 32, 4, 0, index * 512))
    prefix.extend(b"\0" * (-len(prefix) % 32))
    source = root / "source.gguf"
    source.write_bytes(prefix + b"".join(t.tobytes() for t in tensors))
    for filename, seed in [("fit.acal", 11), ("selection.acal", 12)]:
        data = bytearray(b"AC01" + hashlib.sha256(source.read_bytes()).digest() + bytes(32))
        data.extend(struct.pack("<I", len(names)))
        for name in names:
            inputs = np.random.default_rng(seed).normal(size=(8, 32)).astype("<f4")
            data.extend(struct.pack("<I", len(name)) + name.encode())
            data.extend(struct.pack("<BII", 0, 8, 32) + inputs.tobytes())
        data.extend(hashlib.sha256(data).digest())
        (root / filename).write_bytes(data)
    return source, names, tensors


class RecordTests(unittest.TestCase):
    def test_raw_and_compressed_records_decode_identically(self):
        for values in [np.arange(12, dtype=np.float32).reshape(3, 4),
                       np.random.default_rng(8).normal(size=(3, 5)).astype(np.float32)]:
            for entropy in [False, True]:
                with self.subTest(dtype=raw_record(values).meta["kind"], entropy=entropy):
                    np.testing.assert_array_equal(codec.decode(raw_record(values).roundtrip(entropy)), values)

    def test_corrupt_compression_has_a_validation_error(self):
        with self.assertRaises(ValueError):
            Record.loads(b"ACZ1" + struct.pack("<Q", 40) + b"broken zlib stream")

    def test_compressed_length_truncation_and_trailing_streams_are_rejected(self):
        data = raw_record(np.ones((2, 4), np.float32)).dumps(True)
        size = struct.unpack_from("<Q", data, 4)[0]
        for damaged in [data[:-1], data + b"suffix", data + data[12:],
                        data[:4] + struct.pack("<Q", size - 1) + data[12:],
                        data[:4] + struct.pack("<Q", size + 1) + data[12:]]:
            with self.subTest(data=damaged), self.assertRaises(ValueError):
                Record.loads(damaged)

    def test_every_truncation_and_payload_mutation_is_rejected(self):
        data = raw_record(np.array([[0.123, 0.456]], np.float32)).dumps()
        for end in range(len(data)):
            with self.subTest(end=end), self.assertRaises(ValueError):
                Record.loads(data[:end])
        for offset in range(len(data)):
            changed = bytearray(data)
            changed[offset] ^= 1
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                Record.loads(changed)

    def test_checksum_valid_invalid_metadata_is_rejected(self):
        base = {"version": 1, "meta": {"kind": "raw_f32", "shape": [1]}, "arrays": [
            {"name": "values", "dtype": "<f4", "shape": [1], "offset": 0, "bytes": 4}]}
        invalid = [[], None, {}, {**base, "version": True}, {**base, "meta": []}, {**base, "arrays": {}}]
        for field, value in [("name", []), ("name", ""), ("dtype", []), ("shape", [True]),
                             ("shape", [-1]), ("offset", False), ("bytes", 4.0)]:
            header = copy.deepcopy(base)
            header["arrays"][0][field] = value
            invalid.append(header)
        for header in invalid:
            with self.subTest(header=header), self.assertRaises(ValueError):
                Record.loads(frame(header, bytes(4)))


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.cache = ScreenCache(self.temporary.name, {"source": "test"})
        self.result, self.artifact = self.cache.paths("probe", {"bits": 3})
        self.expected = {"shape": [2, 4], "config": {"bits": 3}}
        self.artifact.write_bytes(raw_record(np.ones((2, 4), np.float32)).dumps(True))
        self.save_result()

    def save_result(self):
        self.result.write_text(json.dumps(dict(self.expected, status="measured",
            archive_bytes=self.artifact.stat().st_size, artifact_sha256=digest(self.artifact.read_bytes()))))

    def test_valid_resume_and_changed_configuration(self):
        self.assertTrue(self.cache.resume(self.result, self.artifact, self.expected))
        self.assertFalse(self.cache.resume(self.result, self.artifact, {**self.expected, "config": {"bits": 4}}))

    def test_malformed_result_is_a_cache_miss(self):
        for text in ["[]", "null", '"string"', "{", '{"status":"failed"}']:
            with self.subTest(text=text):
                self.result.write_text(text)
                self.assertFalse(self.cache.resume(self.result, self.artifact, self.expected))

    def test_damaged_record_is_a_cache_miss_even_with_updated_hash(self):
        for data in [b"ACZ1" + struct.pack("<Q", 40) + b"broken zlib stream", frame([])]:
            with self.subTest(data=data):
                self.artifact.write_bytes(data)
                self.save_result()
                self.assertFalse(self.cache.resume(self.result, self.artifact, self.expected))

    def test_wrong_shape_is_rejected_before_decode(self):
        self.artifact.write_bytes(Record({"kind": "constant", "shape": [999999999]},
            {"value": np.array([1], np.float32)}).dumps())
        self.save_result()
        with patch.object(codec, "decode", side_effect=AssertionError("must validate shape first")):
            self.assertFalse(self.cache.resume(self.result, self.artifact, self.expected))

    def test_invalidated_runs_cannot_resume(self):
        (self.cache.directory / "invalidated.json").write_text('{}')
        with self.assertRaisesRegex(ValueError, "invalidated"):
            ScreenCache(self.temporary.name, {"source": "test"})

    def test_replaced_evidence_is_retained(self):
        before = {p.name: p.read_bytes() for p in [self.result, self.artifact]}
        self.cache.retain_previous(self.result, self.artifact)
        self.assertFalse(self.result.exists())
        self.assertFalse(self.artifact.exists())
        history = list((self.cache.directory / "history").iterdir())
        self.assertEqual(len(history), 1)
        self.assertEqual({p.name: p.read_bytes() for p in history[0].iterdir()}, before)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source, self.names, self.tensors = fixture(self.root)

    def build(self):
        output = self.root / "model.a22"
        with contextlib.redirect_stdout(io.StringIO()):
            harness.build_model({"family": "lossless"}, output, self.source, self.root / "fit.acal")
        return output

    def test_complete_archive_roundtrip_is_byte_identical(self):
        model = self.build()
        destination = self.root / "decoded.gguf"
        result = archive.export_model(model, destination)
        self.assertEqual(destination.read_bytes(), self.source.read_bytes())
        self.assertEqual(result["sha256"], digest(self.source.read_bytes()))
        self.assertEqual(sorted(p.name for p in self.root.iterdir()),
            ["decoded.gguf", "fit.acal", "model.a22", "model.a22.json", "selection.acal", "source.gguf"])

    def test_existing_destination_and_source_alias_are_preserved(self):
        model = self.build()
        existing = self.root / "existing.gguf"
        existing.write_bytes(b"user data")
        for destination in [existing, self.source]:
            original = destination.read_bytes()
            with self.assertRaisesRegex(ValueError, "destination exists"):
                archive.export_model(model, destination)
            self.assertEqual(destination.read_bytes(), original)

    def test_archive_record_shape_is_checked_before_decoder_allocation(self):
        model = self.build()
        with zipfile.ZipFile(model) as z:
            members = {name: z.read(name) for name in z.namelist()}
        manifest = json.loads(members["manifest.json"])
        entry = manifest["tensors"][0]
        data = Record({"kind": "constant", "shape": [999999999]},
            {"value": np.array([1], np.float32)}).dumps()
        members[entry["entry"]] = data
        entry.update(bytes=len(data), sha256=digest(data))
        members["manifest.json"] = json.dumps(manifest).encode()
        damaged = self.root / "wrong-shape.a22"
        with zipfile.ZipFile(damaged, "w") as z:
            for name, data in members.items():
                z.writestr(name, data)
        destination = self.root / "wrong-shape.gguf"
        with patch.object(codec, "decode", side_effect=AssertionError("must validate shape first")):
            with self.assertRaisesRegex(ValueError, "record shape differs"):
                archive.export_model(damaged, destination)
        self.assertFalse(destination.exists())
        self.assertEqual(list(self.root.glob(".*.tmp")), [])

    def test_cli_build_and_standalone_decode(self):
        output, destination = self.root / "cli.a22", self.root / "cli.gguf"
        commands = [["build", "--config", "lossless", "--source", str(self.source),
                     "--calibration", str(self.root / "fit.acal"), "--out", str(output)],
                    ["decode", str(output), str(destination)]]
        for command in commands:
            result = subprocess.run([sys.executable, "-m", "experiments.full22.harness", *command],
                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            if command[0] == "build":
                original = self.source.read_bytes()
                self.source.unlink()
        self.assertEqual(destination.read_bytes(), original)

    def test_checksum_valid_layout_corruption_never_publishes(self):
        model = self.build()
        with zipfile.ZipFile(model) as z:
            members = {name: z.read(name) for name in z.namelist()}
        mutations = [lambda m: m["tensors"][1].update(gguf_offset=m["tensors"][0]["gguf_offset"]),
                     lambda m: m["tensors"][0].update(shape=[32, 4]),
                     lambda m: m["tensors"][0].update(name="invented.weight"),
                     lambda m: m.update(source_bytes=m["source_bytes"] + 32),
                     lambda m: m["tensors"][1].update(entry=m["tensors"][0]["entry"])]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                manifest = json.loads(members["manifest.json"])
                mutate(manifest)
                damaged = self.root / f"damaged-{index}.a22"
                with zipfile.ZipFile(damaged, "w") as z:
                    for name, data in members.items():
                        z.writestr(name, json.dumps(manifest).encode() if name == "manifest.json" else data)
                destination = self.root / f"damaged-{index}.gguf"
                with self.assertRaises(ValueError):
                    archive.export_model(damaged, destination)
                self.assertFalse(destination.exists())
                self.assertEqual(list(self.root.glob(".*.tmp")), [])

    def test_screen_resume_is_bound_to_rows_and_config_and_recovers_corruption(self):
        experiments = [{"name": "probe", "principles": [5], "config": {"bits": 3}}]
        with patch.object(harness, "OUT", self.root), patch.object(harness, "SOURCE", self.source), \
                patch.object(harness, "configurations", return_value=experiments), \
                contextlib.redirect_stdout(io.StringIO()):
            first = harness.screen(1, self.names[:1])
            second = harness.screen(1, self.names[:1])
            self.assertEqual((first["measured"], second["resumed"]), (1, 1))
            changed = harness.screen(4, self.names[:1])
            self.assertNotEqual(first["run_fingerprint"], changed["run_fingerprint"])
            experiments[0]["config"]["bits"] = 4
            changed_config = harness.screen(4, self.names[:1])
            self.assertNotEqual(changed["run_fingerprint"], changed_config["run_fingerprint"])
            directory = Path(changed_config["directory"])
            artifact = next(directory.glob("*.ar"))
            result_path = artifact.with_suffix(".json")
            artifact.write_bytes(b"ACZ1" + struct.pack("<Q", 40) + b"broken zlib stream")
            result = json.loads(result_path.read_text())
            result.update(archive_bytes=artifact.stat().st_size, artifact_sha256=digest(artifact.read_bytes()))
            result_path.write_text(json.dumps(result))
            recovered = harness.screen(4, self.names[:1])
            self.assertEqual((recovered["measured"], recovered["resumed"], recovered["failed"]), (1, 0, 0))
            self.assertEqual(len(list((directory / "history").iterdir())), 1)
            self.assertEqual(harness.screen(4, self.names[:1])["resumed"], 1)


if __name__ == "__main__":
    unittest.main()
