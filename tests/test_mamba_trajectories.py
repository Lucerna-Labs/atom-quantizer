from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.mamba_trajectories import energy, summaries, write_arrays, run


class MambaTrajectoryTests(unittest.TestCase):
    def test_energy_keeps_absolute_per_layer_observations(self):
        source = torch.ones(2, 3, 4)
        candidate = source * 2
        e, r, q = energy(source, candidate)
        np.testing.assert_array_equal(e, [12, 12])
        np.testing.assert_array_equal(r, [12, 12])
        np.testing.assert_array_equal(q, [48, 48])
        with self.assertRaises(ValueError): energy(source, torch.ones(2, 3, 5))
        with self.assertRaises(ValueError): energy(source, torch.full_like(source, float("nan")))

    def test_prefix_and_tail_summaries_use_summed_energies(self):
        trace = {}
        for kind in ("ssm", "conv"):
            trace[kind+"_reference"] = np.ones((4, 1024, 2))
            trace[kind+"_candidate"] = np.ones((4, 1024, 2))
            trace[kind+"_error"] = np.full((4, 1024, 2), .25)
        for key, value in (("teacher_kl", .1), ("source_nll", np.log(2)), ("candidate_nll", np.log(4)), ("top1_match", .5)):
            trace[key] = np.full((4, 1024), value)
        result = summaries(trace)
        self.assertTrue(result["finite"])
        self.assertEqual(result["tail_256_ssm_nmse"], .25)
        for length in ("64", "256", "1024"):
            self.assertEqual(result[length]["ssm_nmse"], .25)
            self.assertEqual(result[length]["ssm_per_layer_nmse"], [.25, .25])
            self.assertEqual(result[length]["ssm_per_sequence_nmse"], [.25]*4)
            self.assertAlmostEqual(result[length]["candidate_ppl"], 4)
        trace["ssm_reference"][..., 0] *= 100
        self.assertAlmostEqual(summaries(trace)["tail_256_ssm_nmse"], .5/101)

    def test_existing_outputs_and_dangling_destinations_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            existing = parent/"existing"
            existing.mkdir()
            (existing/"keep").write_text("retained")
            with self.assertRaises(FileExistsError): run("missing", "missing", existing)
            self.assertEqual(sorted(p.name for p in existing.iterdir()), ["keep"])
            linked = parent/"linked"
            linked.symlink_to(parent/"absent", target_is_directory=True)
            with self.assertRaises(FileExistsError): run("missing", "missing", linked)
            self.assertTrue(linked.is_symlink())
            self.assertFalse((parent/"absent").exists())
            archive = parent/"state.npz"
            write_arrays(archive, {"state": np.ones(3)})
            before = archive.read_bytes()
            with self.assertRaises(FileExistsError): write_arrays(archive, {"state": np.zeros(3)})
            self.assertEqual(archive.read_bytes(), before)


if __name__ == "__main__": unittest.main()
