"""Architecture and information-availability tests; no real-data training."""
import importlib.util
from pathlib import Path
import sys
import unittest

import torch

V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))
from shared.models import BPNet, EXPECTED_PARAMETERS, VARIANTS, count_parameters, make_model


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        spec = importlib.util.spec_from_file_location("v1_network_for_test", V2_ROOT.parent / "PulseDB-based Model-V1/network.py")
        cls.v1 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.v1)

    def test_exact_v1_state_keys_and_outputs(self):
        torch.manual_seed(42)
        old = self.v1.BPNet(pool=1).eval()
        new = make_model("baseline", 42).eval()
        self.assertEqual(list(old.state_dict()), list(new.state_dict()))
        for name, value in old.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[name], rtol=0, atol=0)
        x = torch.randn(3, 2, 1250)
        torch.testing.assert_close(old(x), new(x), rtol=0, atol=0)
        new.load_state_dict(old.state_dict(), strict=True)
        self.assertEqual(count_parameters(new), 37666)

    def test_variants_shapes_counts_gradients_and_equal_backbone(self):
        baseline = make_model("baseline", 43)
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = make_model(variant, 43)
                self.assertEqual(count_parameters(model), EXPECTED_PARAMETERS[variant])
                for name, value in baseline.state_dict().items():
                    torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
                out = model(torch.randn(2, 2, 1250))
                self.assertEqual(out.shape, (2, 2))
                self.assertTrue(torch.isfinite(out).all())
                out.square().sum().backward()
                for name, param in model.named_parameters():
                    self.assertIsNotNone(param.grad, name)
                    self.assertTrue(torch.isfinite(param.grad).all(), name)

    def test_gates_preserve_block_shape(self):
        model = make_model("se_all", 42).eval()
        shapes = []
        hooks = [gate.register_forward_hook(lambda module, args, output: shapes.append(tuple(output.shape)))
                 for gate in model.gates.values()]
        model(torch.randn(2, 2, 1250))
        self.assertEqual(shapes, [(2, 16, 313), (2, 32, 157), (2, 48, 79)])
        for hook in hooks:
            hook.remove()

    def test_modality_mask_is_in_model(self):
        x = torch.randn(2, 2, 1250)
        for variant, missing in (("baseline_ecg", 1), ("baseline_ppg", 0)):
            model = make_model(variant, 42).eval()
            changed = x.clone()
            changed[:, missing] = torch.randn_like(changed[:, missing]) * 100
            torch.testing.assert_close(model(x), model(changed), rtol=0, atol=0)

    def test_factory_preserves_rng_and_rejects_nonfrozen_shapes(self):
        before = torch.get_rng_state().clone()
        make_model("cbam_last", 44)
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            BPNet(pool=5)
        with self.assertRaises(ValueError):
            BPNet(variant="not-a-model")
        with self.assertRaises(ValueError):
            BPNet()(torch.randn(2, 2, 250))


if __name__ == "__main__":
    unittest.main()
