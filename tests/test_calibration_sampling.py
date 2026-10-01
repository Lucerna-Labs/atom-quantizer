import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.calibration_sampling import stratified_indices, write_capture


class CalibrationSamplingTests(unittest.TestCase):
    def test_each_stratum_has_one_distinct_reproducible_position(self):
        for length in [1, 2, 7, 31, 128, 511, 512]:
            for count in [1, 2, 3, 16, 600]:
                selected = stratified_indices(length, count, 0, "corpus:doc")
                n = min(length, count)
                self.assertEqual(len(selected), n)
                self.assertEqual(selected, sorted(set(selected)))
                self.assertEqual(selected, stratified_indices(length, count, 0, "corpus:doc"))
                for i, position in enumerate(selected):
                    self.assertTrue(i*length//n <= position < (i+1)*length//n)

    def test_document_and_seed_change_coverage_without_endpoint_concentration(self):
        positions = [p for d in range(1024) for p in stratified_indices(512, 2, 0, f"probe:{d}")]
        self.assertGreater(len(set(positions)), 490)
        self.assertLess(sum(p in (0, 511) for p in positions), 30)
        self.assertNotEqual(stratified_indices(512, 2, 0, "doc"), stratified_indices(512, 2, 1, "doc"))

    def test_invalid_sampling_arguments_fail_closed(self):
        for values in [(0, 2, 0, "d"), (3, 0, 0, "d"), (3, 2, -1, "d"), (True, 2, 0, "d")]:
            with self.assertRaises(ValueError):
                stratified_indices(*values)

    def test_publication_is_new_only_and_matches_its_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"capture.acal"
            body = b"calibration fixture"
            write_capture(path, body, {"sampling": "stratified"})
            metadata = json.loads(path.with_suffix(".acal.json").read_text())
            self.assertEqual(path.read_bytes(), body)
            self.assertEqual(metadata["file_sha256"], hashlib.sha256(body).hexdigest())
            with self.assertRaises(ValueError):
                write_capture(path, b"changed", {})
            self.assertEqual(path.read_bytes(), body)
            other = Path(temporary)/"other.acal"
            other.with_suffix(".acal.json").write_text("user metadata")
            with self.assertRaises(ValueError):
                write_capture(other, body, {})
            self.assertFalse(other.exists())


if __name__ == "__main__":
    unittest.main()
