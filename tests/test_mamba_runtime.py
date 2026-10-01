from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import gguf
import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import MambaConfig, MambaForCausalLM

from experiments.mamba_runtime import converted, load, states, transition_preimage, verify


def fixture(path, metadata_change=None, tensor_change=None):
    torch.manual_seed(1824)
    config = MambaConfig(vocab_size=32, hidden_size=16, num_hidden_layers=2,
                         state_size=4, expand=2, conv_kernel=4, time_step_rank=2,
                         use_bias=False, use_conv_bias=True, tie_word_embeddings=True)
    model = MambaForCausalLM(config).float().eval()
    metadata = {"mamba.context_length": 1048576, "mamba.embedding_length": 16,
        "mamba.feed_forward_length": 0, "mamba.attention.head_count": 0, "mamba.block_count": 2,
        "mamba.ssm.conv_kernel": 4, "mamba.ssm.inner_size": 32, "mamba.ssm.state_size": 4,
        "mamba.ssm.time_step_rank": 2, "mamba.attention.layer_norm_rms_epsilon": float(config.layer_norm_epsilon),
        "mamba.ssm.dt_b_c_rms": False}
    if metadata_change: metadata_change(metadata)
    mapping = gguf.get_tensor_name_map(gguf.MODEL_ARCH.MAMBA, 2)
    arrays = {mapping.get_name(n, try_suffixes=(".weight", ".bias")): converted(p, n).copy()
              for n, p in model.named_parameters()}
    if tensor_change: tensor_change(arrays)
    writer = gguf.GGUFWriter(path, "mamba")
    for key, value in metadata.items():
        if type(value) is bool: writer.add_bool(key, value)
        elif type(value) is float: writer.add_float32(key, value)
        else: writer.add_uint32(key, value)
    for name, values in arrays.items(): writer.add_tensor(name, values)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    return model


class MambaRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(4)

    def test_transition_preimage_preserves_the_actual_exponential(self):
        original = torch.linspace(-8, 4, 1024, dtype=torch.float32).reshape(64, 16)
        target = -original.exp()
        recovered = torch.from_numpy(transition_preimage(target.numpy()))
        self.assertTrue(torch.equal(-recovered.exp(), target))
        impossible = -torch.nextafter(torch.tensor(10., dtype=torch.float32).exp(), torch.tensor(float("inf"))).item()
        for values in (np.array([0.]), np.array([1.]), np.array([np.nan]), np.array([impossible], dtype=np.float32)):
            with self.subTest(values=values), self.assertRaises(ValueError): transition_preimage(values)

    def test_complete_stored_operator_loading_preserves_real_logits_and_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.gguf"
            original = fixture(path)
            restored, receipt = load(path)
            self.assertEqual(receipt["verified_tensors"], 22)
            self.assertEqual(restored.backbone.embeddings.weight.data_ptr(), restored.lm_head.weight.data_ptr())
            ids = torch.tensor([[1, 5, 3, 7, 4, 9]])
            with torch.inference_mode():
                before = original(ids, use_cache=True)
                after = restored(ids, use_cache=True)
                self.assertTrue(torch.equal(before.logits, after.logits))
                for kind, values in states(before.cache_params, original.config).items():
                    self.assertTrue(torch.equal(values, states(after.cache_params, restored.config)[kind]))
            self.assertEqual(verify(original, gguf.GGUFReader(path))["operator_sha256"], receipt["operator_sha256"])

    def test_incomplete_or_incompatible_stored_models_reject(self):
        changes = [
            (lambda m: m.pop("mamba.ssm.time_step_rank"), None),
            (lambda m: m.update({"mamba.ssm.inner_size": 48}), None),
            (lambda m: m.update({"mamba.ssm.dt_b_c_rms": True}), None),
            (None, lambda t: t.pop("blk.0.ssm_dt.weight")),
            (None, lambda t: t.update({"blk.0.ssm_a": np.ones((32, 4), np.float32)})),
            (None, lambda t: t.update({"blk.0.ssm_in.weight": np.full((32, 16), np.nan, np.float32)})),
        ]
        for index, (metadata, tensors) in enumerate(changes):
            with self.subTest(case=index), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "model.gguf"
                fixture(path, metadata, tensors)
                with self.assertRaises(ValueError): load(path)

    def test_cache_dictionary_values_are_used_and_metadata_is_not_a_tensor(self):
        config = MambaConfig(hidden_size=16, state_size=4, num_hidden_layers=2, conv_kernel=4)
        layers = [SimpleNamespace(number_of_states=1,
            recurrent_states={0: torch.ones(1, 32, 4)}, conv_states={0: torch.zeros(1, 32, 4)}) for _ in range(2)]
        cache = SimpleNamespace(layers=layers)
        self.assertEqual(tuple(states(cache, config)["recurrent_states"].shape), (2, 32, 4))
        for replacement in ({0: 1}, {0: torch.ones(32, 4)}, {0: torch.full((1, 32, 4), float("nan"))}, {1: torch.ones(1, 32, 4)}):
            with self.subTest(replacement=type(replacement[ next(iter(replacement)) ]).__name__):
                cache.layers[0].recurrent_states = replacement
                with self.assertRaises(ValueError): states(cache, config)

    def test_dt_input_is_the_actual_selector_output_prefix(self):
        config = MambaConfig(vocab_size=32, hidden_size=16, num_hidden_layers=2,
                             state_size=4, expand=2, conv_kernel=4, time_step_rank=2)
        model = MambaForCausalLM(config).float().eval()
        selected, actual_inputs, pre_calls, handles = [], [], [], []
        dt_pointers = {layer.mixer.dt_proj.weight.data_ptr() for layer in model.backbone.layers}
        class ObserveMatmul(TorchDispatchMode):
            def __torch_dispatch__(self, function, types, args=(), kwargs=None):
                if (str(function) in ("aten.matmul.default", "aten.mm.default", "aten.bmm.default")
                        and args[0].data_ptr() in dt_pointers):
                    actual_inputs.append(args[1].transpose(-1, -2).reshape(-1, 2).clone())
                return function(*args, **(kwargs or {}))
        for layer in model.backbone.layers:
            handles.append(layer.mixer.x_proj.register_forward_hook(lambda mod, inp, out: selected.append(out[..., :2].clone())))
            handles.append(layer.mixer.dt_proj.register_forward_pre_hook(lambda mod, inp: pre_calls.append(inp)))
        with torch.inference_mode(), ObserveMatmul():
            model(torch.tensor([[1, 2, 5, 3, 8, 4]]), use_cache=False)
        for handle in handles: handle.remove()
        self.assertEqual(len(selected), 2)
        self.assertEqual(len(actual_inputs), 2)
        self.assertFalse(pre_calls)
        for expected, actual in zip(selected, actual_inputs):
            self.assertTrue(torch.equal(expected.reshape(-1, 2), actual))


if __name__ == "__main__": unittest.main()
