import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from experiments.mamba_evaluate import benefit_gates
from experiments.mamba_models import ORDER
from experiments import mamba_models


class MambaBenefitTests(unittest.TestCase):
    def fixture(self):
        models = {n: {"archive_bytes": 1000} for n in ORDER}
        scores = {n: {"status": "MEASURED", "perplexity": 20.} for n in ORDER}
        traces = {n: {"status": "MEASURED", "summary": {"teacher_kl": .1,
                  "tail_256_ssm_nmse": 1., "finite": True}} for n in {"source-reset", *ORDER[1:]}}
        traces["q6-functional"]["summary"]["tail_256_ssm_nmse"] = .9
        traces["q6-functional-selectors-exact"]["summary"]["tail_256_ssm_nmse"] = .7
        return models, scores, {"status": "MEASURED", "models": traces}

    def test_both_storage_and_temporal_quality_are_required(self):
        models, scores, traces = self.fixture()
        result = benefit_gates(models, scores, traces)
        self.assertEqual(result["G2"]["status"], "MEASURED")
        self.assertEqual(result["G3"]["status"], "MEASURED")
        models["q6-functional"]["archive_bytes"] = 1051
        self.assertEqual(benefit_gates(models, scores, traces)["G2"]["status"], "FAILED")
        models["q6-functional"]["archive_bytes"] = 1000
        traces["models"]["q6-functional"]["summary"]["tail_256_ssm_nmse"] = .96
        self.assertEqual(benefit_gates(models, scores, traces)["G2"]["status"], "FAILED")
        traces["models"]["q6-functional"]["summary"]["tail_256_ssm_nmse"] = .9
        scores["q6-functional-selectors-exact"]["perplexity"] = 20.1
        self.assertEqual(benefit_gates(models, scores, traces)["G3"]["status"], "FAILED")

    def test_partial_model_or_trajectory_scope_is_inconclusive(self):
        models, scores, traces = self.fixture()
        scores["q4-scalar"]["status"] = "FAILED"
        self.assertEqual(benefit_gates(models, scores, traces)["status"], "INCONCLUSIVE")
        scores["q4-scalar"]["status"] = "MEASURED"
        traces["models"].pop("source-reset")
        self.assertEqual(benefit_gates(models, scores, traces)["status"], "INCONCLUSIVE")

    def test_build_rejects_existing_and_dangling_destinations_without_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing = root/"existing"
            existing.mkdir()
            (existing/"keep").write_text("unchanged")
            link = root/"linked"
            link.symlink_to(root/"absent", target_is_directory=True)
            with patch.object(mamba_models, "_run", side_effect=AssertionError("must not run")):
                for destination in (existing, link):
                    with self.assertRaises(FileExistsError): mamba_models.run(destination)
            self.assertEqual(sorted(p.name for p in existing.iterdir()), ["keep"])
            self.assertTrue(link.is_symlink())
            self.assertFalse((root/"absent").exists())

    def test_owned_failed_build_keeps_its_diagnostic_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)/"new"
            with patch.object(mamba_models, "_run", side_effect=ValueError("observed failure")):
                with self.assertRaisesRegex(ValueError, "observed failure"): mamba_models.run(out)
            receipt = json.loads((out/"failure.json").read_text())
            self.assertEqual(receipt["status"], "FAILED")
            self.assertIn("observed failure", receipt["error"])


if __name__ == "__main__": unittest.main()
