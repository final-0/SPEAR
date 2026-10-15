"""Fast CPU contracts; small synthetic tensors are TESTS, not resized training."""
import copy
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import h5py
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from .attention import attention
from .data import NATIVE, NativeSpring, check_native, choose_condition, read_flow, sample_indices
from .losses import clean_regret, sequence_loss
from .model import ThreeFrameRobustWAFT
from .runtime import load_training, save_training


class ToyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.iter_dim, self.factor = 8, 4
        self.args = SimpleNamespace(iters=2)
        self.encoder = SimpleNamespace(embed_dim=16)
        self.hidden_conv = nn.Conv2d(16, 8, 1)
        self.warp_linear = nn.Conv2d(26, 8, 1)
        self.refine_net = ToyRefine()
        self.refine_transform = nn.Conv2d(12, 8, 1)
        self.flow_head = nn.Conv2d(8, 6, 1)
        self.upsample_weight = nn.Conv2d(8, 36, 1)

    def normalize_image(self, image):
        return image.float() / 255

    def upsample_data(self, flow, info, mask):
        # Exercise mask gradients too, unlike a detached interpolation mock.
        return F.interpolate(flow * 2 + mask[:, :2] * .01, scale_factor=2), F.interpolate(info, scale_factor=2)


class ToyRefine(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 4, 1)

    def forward(self, image):
        return {"out": self.conv(image)}


class ToyThreeFrame(ThreeFrameRobustWAFT):
    def __init__(self, checkpointing=True):
        super().__init__(ToyBase(), checkpointing, temporal_dim=8, weather_dim=8)
        self.require_native = False  # ONLY synthetic unit tests may disable this.

    @torch.no_grad()
    def _encode(self, image, tokens_only=False):
        feature = F.avg_pool2d(image.mean(1, keepdim=True), 2).repeat(1, 8, 1, 1)
        summary = image.mean((1, 2, 3))[:, None, None].repeat(1, 4, 64)
        return (None if tokens_only else feature), summary


class NativeContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_native_shape(self):
        check_native(np.zeros((*NATIVE, 3), np.uint8))
        with self.assertRaises(ValueError):
            check_native(np.zeros((540, 960, 3), np.uint8))

    def test_4k_label_grid_keeps_flow_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "label.flo5"
            with h5py.File(path, "w") as f:
                f.create_dataset("flow", shape=(2160, 3840, 2), dtype="f4", fillvalue=8)
            flow, valid = read_flow(path)
            self.assertEqual(flow.shape, (*NATIVE, 2))
            self.assertTrue((flow == 8).all())
            self.assertTrue(valid.all())

    def test_sampler_equivalent_2gpu_8gpu_and_resume(self):
        for world in (1, 2, 8):
            samples = sorted(i for rank in range(world) for i in sample_indices(0, 7, 8, 1, rank, world))
            self.assertEqual(samples, list(range(56)))
            for rank in range(world):
                all_indices = list(sample_indices(0, 7, 8, 1, rank, world))
                resumed = list(sample_indices(3, 7, 8, 1, rank, world))
                self.assertEqual(all_indices[3 * 8 // world:], resumed)
        with self.assertRaises(ValueError):
            list(sample_indices(0, 1, 8, 1, 3, 3))

    def test_sampling_covers_all20_and_clean(self):
        rng = random.Random(91)
        self.assertEqual(len({choose_condition(rng) for _ in range(10000)}), 21)

    def test_only_train_directory_and_no_val_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "val", "test"):
                camera = root / split / "0042/frame_left"
                camera.mkdir(parents=True)
                labels = camera.parent / "flow_FW_left"
                labels.mkdir()
                for i in range(1, 8):
                    (camera / f"frame_left_{i:04d}.png").touch()
                    (labels / f"flow_FW_left_{i:04d}.flo5").touch()
            train = NativeSpring(root, cameras=("left",), directions=("FW",))
            diagnostic = NativeSpring(root, "diagnostic", cameras=("left",), directions=("FW",))
            self.assertEqual(len(train), 3)
            self.assertEqual(len(diagnostic), 3)
            self.assertFalse({r["id"] for r in train.records} & {r["id"] for r in diagnostic.records})
            self.assertTrue(all("/train/" in str(r["gt"]) for r in train.records + diagnostic.records))
            self.assertEqual(diagnostic.records[0]["times"], (1, 1, 2))
            with self.assertRaises(ValueError):
                NativeSpring(root, "test")

    def test_chunked_attention_output_and_gradients(self):
        q, k, v = [torch.randn(1, 2, 11, 4, requires_grad=True) for _ in range(3)]
        expected = F.scaled_dot_product_attention(q, k, v)
        grads = torch.autograd.grad(expected.sum(), (q, k, v))
        actual = attention(q, k, v, "chunked", chunk=3)
        torch.testing.assert_close(actual, expected)
        for a, b in zip(torch.autograd.grad(actual.sum(), (q, k, v)), grads):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)

    def test_native_boundary_duplicates_corrupted_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            camera = root / "train/0042/frame_left"
            labels = camera.parent / "flow_FW_left"
            camera.mkdir(parents=True)
            labels.mkdir()
            yy, xx = np.mgrid[:NATIVE[0], :NATIVE[1]]
            rgb = np.stack((xx % 256, yy % 256, (xx+yy) % 256), -1).astype(np.uint8)
            for i in range(1, 4):
                Image.fromarray(rgb).save(camera / f"frame_left_{i:04d}.png")
                with h5py.File(labels / f"flow_FW_left_{i:04d}.flo5", "w") as f:
                    f.create_dataset("flow", shape=(*NATIVE, 2), dtype="f4", fillvalue=4)
            data = NativeSpring(root, "diagnostic", cameras=("left",), directions=("FW",), prefix=1)
            for condition in ("gaussian_noise", "elastic_transform"):
                item = data.fetch(0, condition, 3, 91)
                self.assertTrue(torch.equal(item["corrupt"][0], item["corrupt"][1]))
                self.assertEqual(tuple(item["corrupt"].shape), (3, 3, *NATIVE))

    def test_identity_and_third_frame_has_effect_after_update(self):
        model = ToyThreeFrame().eval()
        frames = [torch.rand(1, 3, 16, 24) * 255 for _ in range(3)]
        with torch.no_grad():
            original = model(*frames, baseline=True)["flow"][-1]
            adapted = model(*frames)["flow"][-1]
        torch.testing.assert_close(adapted, original, atol=0, rtol=0)
        with torch.no_grad():
            model.temporal_context.out_proj.weight.normal_(std=.1)
            model.temporal_recurrent.out_proj.weight.normal_(std=.1)
            a = model(*frames)["flow"][-1]
            b = model(frames[0] + 100, *frames[1:])["flow"][-1]
        self.assertGreater(float((a-b).abs().max()), 1e-7)

    def test_checkpointing_forward_and_gradient_equivalence(self):
        first = ToyThreeFrame().train()
        second = copy.deepcopy(first)
        second.checkpointing = False
        frames = [torch.rand(1, 3, 16, 24) * 255 for _ in range(3)]
        losses = []
        for model in (first, second):
            out = model(*frames)
            loss = sequence_loss(out, torch.ones(1, 2, 16, 24), torch.ones(1, 16, 24))
            loss.backward()
            losses.append(loss)
            self.assertTrue(all(p.grad is None for p in model.base.parameters()))
            self.assertIsNotNone(model.temporal_context.out_proj.weight.grad)
            self.assertIsNotNone(model.temporal_recurrent.out_proj.weight.grad)
            self.assertIsNotNone(model.experts[0].out_proj.weight.grad)
        torch.testing.assert_close(*losses)
        for a, b in zip(first.parameters(), second.parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad)

    def test_stage_freezing_and_router_features_only(self):
        model = ToyThreeFrame()
        for stage in ("router", "adapters"):
            model.set_stage(stage)
            model.train()
            self.assertFalse(model.base.training)
            self.assertTrue(all(not p.requires_grad for p in model.base.parameters()))
            self.assertEqual(model.router.classifier[-1].weight.requires_grad, stage == "router")
            self.assertEqual(model.temporal_context.out_proj.weight.requires_grad, stage == "adapters")
        self.assertFalse(any(isinstance(m, nn.Conv2d) for m in model.router.modules()))
        a = [torch.randn(2, 4, 64) for _ in range(3)]
        self.assertEqual(model.router(*a).shape, (2, model.num_expert_groups))
        with self.assertRaises(TypeError):
            model.router(*a, corruption_name="rain")

    def test_clean_regret_only_penalizes_worse(self):
        gt = torch.zeros(1, 2, 2, 2)
        valid = torch.ones(1, 2, 2)
        self.assertEqual(clean_regret(gt, gt + 1, gt, valid).item(), 0)
        self.assertGreater(clean_regret(gt + 2, gt + 1, gt, valid).item(), 0)

    def test_portable_checkpoint_roundtrip_and_reject_invalid(self):
        model = ToyThreeFrame()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "step-1.pt"
            save_training(path, model, optimizer, scaler, 1, {"stage": "adapters"}, 0, 1)
            other = ToyThreeFrame()
            payload = load_training(path, other)
            self.assertEqual(payload["step"], 1)
            for k, v in model.delta_state().items():
                torch.testing.assert_close(v, other.delta_state()[k], atol=0, rtol=0)
            with self.assertRaises(FileExistsError):
                save_training(path, model, optimizer, scaler, 1, {}, 0, 1)
            with self.assertRaises(ValueError):
                model.load_delta({})


if __name__ == "__main__":
    unittest.main()
