from pathlib import Path
import tempfile
import unittest

import gguf
import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from experiments.full22.gguf_weights import install, inverse_weight
from experiments.sequential_functional import gates, CONTROLS, ORDER
from scripts.capture_calibration import converted_weight


class DecodedWeightsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2201)
        self.model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16,
            intermediate_size=24, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, tie_word_embeddings=True)).float().eval()

    def write(self, path, mutation=None, architecture="llama", metadata_mutation=None):
        mapping = gguf.get_tensor_name_map(gguf.MODEL_ARCH.LLAMA, 2)
        arrays = {}
        for name, parameter in self.model.named_parameters():
            mapped = mapping.get_name(name, try_suffixes=(".weight", ".bias"))
            weight = parameter.detach().numpy().copy()
            converted = converted_weight(weight, mapped, self.model.config)
            arrays[mapped] = weight if converted is None else converted
        if mutation:
            mutation(arrays)
        writer = gguf.GGUFWriter(path, architecture)
        config = self.model.config
        metadata = {
            "llama.block_count": 2, "llama.context_length": config.max_position_embeddings,
            "llama.embedding_length": 16, "llama.feed_forward_length": 24,
            "llama.attention.head_count": 4, "llama.attention.head_count_kv": 2,
            "llama.rope.freq_base": float(config.rope_parameters["rope_theta"]),
            "llama.attention.layer_norm_rms_epsilon": float(config.rms_norm_eps),
            "llama.attention.key_length": 4, "llama.attention.value_length": 4,
            "llama.vocab_size": 32, "llama.rope.dimension_count": 4,
        }
        if metadata_mutation:
            metadata_mutation(metadata)
        for key, value in metadata.items():
            if isinstance(value, float): writer.add_float32(key, value)
            else: writer.add_uint32(key, value)
        for name, weight in arrays.items():
            writer.add_tensor(name, weight)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        return arrays

    def test_complete_roundtrip_preserves_actual_logits_and_tied_head(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.gguf"
            arrays = self.write(path)
            ids = torch.tensor([[1, 4, 3, 7, 2]])
            with torch.inference_mode():
                expected = self.model(ids).logits.clone()
            other = LlamaForCausalLM(self.model.config).float().eval()
            receipt = install(other, gguf.GGUFReader(path))
            self.assertEqual(receipt["verified_tensors"], len(arrays))
            self.assertEqual(receipt["tied_aliases"], ["lm_head.weight"])
            self.assertEqual(other.lm_head.weight.data_ptr(), other.model.embed_tokens.weight.data_ptr())
            with torch.inference_mode():
                self.assertTrue(torch.equal(expected, other(ids).logits))

    def test_inverse_query_key_permutations(self):
        for name, rows in (("blk.0.attn_q.weight", 16), ("blk.0.attn_k.weight", 8)):
            weight = np.arange(rows * 16, dtype=np.float32).reshape(rows, 16)
            permuted = converted_weight(weight, name, self.model.config)
            self.assertFalse(np.array_equal(permuted, weight))
            np.testing.assert_array_equal(inverse_weight(permuted, name, self.model.config), weight)
        with self.assertRaises(ValueError):
            inverse_weight(np.ones((3, 4)), "blk.0.attn_q.weight", self.model.config)

    def test_incomplete_invalid_and_extra_sources_reject_before_mutation(self):
        def missing(a): del a["output_norm.weight"]
        def extra(a): a["extra.weight"] = np.ones(3, dtype=np.float32)
        def shape(a): a["output_norm.weight"] = np.ones(3, dtype=np.float32)
        def half(a): a["output_norm.weight"] = a["output_norm.weight"].astype(np.float16)
        def nonfinite(a): a["output_norm.weight"][0] = np.nan
        for mutation in (missing, extra, shape, half, nonfinite):
            with self.subTest(case=mutation.__name__), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.gguf"
                self.write(path, mutation)
                before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
                with self.assertRaises(ValueError):
                    install(self.model, gguf.GGUFReader(path))
                self.assertTrue(all(torch.equal(before[n], p) for n, p in self.model.named_parameters()))

    def test_wrong_architecture_and_broken_alias_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.gguf"
            self.write(path, architecture="gpt2")
            with self.assertRaises(ValueError): install(self.model, gguf.GGUFReader(path))
            correct = Path(directory) / "correct.gguf"
            self.write(correct)
            self.model.lm_head.weight = torch.nn.Parameter(self.model.lm_head.weight.detach().clone())
            with self.assertRaisesRegex(ValueError, "unmapped"):
                install(self.model, gguf.GGUFReader(correct))

    def test_same_weight_shapes_with_different_attention_heads_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.gguf"
            self.write(path)
            wrong = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16,
                intermediate_size=24, num_hidden_layers=2, num_attention_heads=8,
                num_key_value_heads=4, tie_word_embeddings=True)).float().eval()
            self.assertEqual([tuple(p.shape) for p in self.model.parameters()],
                             [tuple(p.shape) for p in wrong.parameters()])
            before = {n: p.detach().clone() for n, p in wrong.named_parameters()}
            with self.assertRaisesRegex(ValueError, "architecture mismatch"):
                install(wrong, gguf.GGUFReader(path))
            self.assertTrue(all(torch.equal(before[n], p) for n, p in wrong.named_parameters()))

    def test_missing_different_and_unsupported_architecture_state_rejected(self):
        def missing(m): del m["llama.attention.head_count"]
        def epsilon(m): m["llama.attention.layer_norm_rms_epsilon"] = 0.1
        def extra(m): m["llama.attention.sliding_window"] = 8
        for mutation in (missing, epsilon, extra):
            with self.subTest(case=mutation.__name__), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.gguf"
                self.write(path, metadata_mutation=mutation)
                before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
                with self.assertRaises(ValueError): install(self.model, gguf.GGUFReader(path))
                self.assertTrue(all(torch.equal(before[n], p) for n, p in self.model.named_parameters()))


class SequentialGatesTests(unittest.TestCase):
    def fixture(self):
        models = {n: {"archive_bytes": 1000} for n in (*CONTROLS, *ORDER)}
        models["sequential-activation"]["archive_bytes"] = 850
        scores = {n: {"status": "MEASURED", "perplexity": 20.} for n in models}
        reports = {"sequential-activation": {
            f"{ref}-{split}-{i}": {"report": {"all_passed": True}}
            for ref in ("W", "W1") for split in ("fit", "selection") for i in range(2)}}
        return models, scores, reports

    def test_both_references_and_all_views_are_mandatory(self):
        models, scores, reports = self.fixture()
        self.assertEqual(gates(models, scores, reports)["status"], "QUALIFIED")
        for key in list(reports["sequential-activation"]):
            value = reports["sequential-activation"].pop(key)
            self.assertEqual(gates(models, scores, reports)["status"], "FAILED")
            reports["sequential-activation"][key] = value
        reports["sequential-activation"]["W-selection-1"]["report"]["all_passed"] = False
        self.assertEqual(gates(models, scores, reports)["status"], "FAILED")

    def test_historical_benefit_does_not_replace_direct_or_scalar_comparison(self):
        models, scores, reports = self.fixture()
        models["six-activation"]["archive_bytes"] = 850
        result = gates(models, scores, reports)
        self.assertEqual(result["G3"]["status"], "MEASURED")
        self.assertEqual(result["G4"]["status"], "FAILED")
        models["six-activation"]["archive_bytes"] = 1000
        scores["sequential-scalar"]["perplexity"] = 19.9
        self.assertEqual(gates(models, scores, reports)["status"], "FAILED")
        scores["source-f32"]["status"] = "FAILED"
        self.assertEqual(gates(models, scores, reports)["status"], "INCONCLUSIVE")


if __name__ == "__main__": unittest.main()
