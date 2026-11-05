"""Numerical and coverage checks for local, batched axial SVD.

Run from the repository root: python -m unittest discover -s svd/tests -v
"""
from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_svd_core import analyze_local, reconstruct_block


def matrix_for(stack, bounds):
    z0, y0, x0, z1, y1, x1 = map(int, bounds)
    return stack[z0:z1, y0:y1, x0:x1].reshape(z1-z0, -1).T.astype(np.float64)


class LocalSVDTests(unittest.TestCase):
    def analyze(self, stack, **kwargs):
        options = dict(resolution=3, slab=4, ranks=(0, 1, 2, 4),
                       device="cpu", batch_size=3, dtype="float64")
        options.update(kwargs)
        return analyze_local(stack, **options)

    def test_odd_edges_and_tail_slabs_cover_every_voxel_once(self):
        stack = np.arange(7*5*8, dtype=np.float64).reshape(7, 5, 8) / 300
        result = self.analyze(stack)
        blocks = result["analyses"]["raw"]
        coverage = np.zeros(stack.shape, dtype=np.int32)
        for bound, count, full in zip(blocks["bounds"], blocks["voxel_count"], blocks["full_block"]):
            z0, y0, x0, z1, y1, x1 = map(int, bound)
            coverage[z0:z1, y0:y1, x0:x1] += 1
            self.assertEqual(int(count), (z1-z0)*(y1-y0)*(x1-x0))
            self.assertEqual(bool(full), (z1-z0 == 4 and y1-y0 == 3 and x1-x0 == 3))
        np.testing.assert_array_equal(coverage, np.ones(stack.shape))
        self.assertEqual(int(np.sum(blocks["voxel_count"])), stack.size)
        self.assertTrue(np.any(blocks["bounds"][:, 3] - blocks["bounds"][:, 0] == 3))

    def test_zero_volume_has_zero_ranks_and_errors(self):
        result = self.analyze(np.zeros((5, 4, 7), dtype=np.float32))
        for blocks in result["analyses"].values():
            self.assertTrue(np.all(blocks["near_zero"]))
            np.testing.assert_array_equal(blocks["effective_ranks"], 0)
            np.testing.assert_array_equal(blocks["eigenvalues"], 0)
            np.testing.assert_array_equal(blocks["sse"], 0)

    def test_repeated_slices_have_no_axial_residual(self):
        image = np.random.default_rng(9).random((5, 8))
        stack = np.repeat(image[None], 7, axis=0)
        result = self.analyze(stack)
        raw = result["analyses"]["raw"]
        centered = result["analyses"]["axial_centered"]
        np.testing.assert_array_equal(raw["effective_ranks"], 1)
        self.assertTrue(np.all(centered["near_zero"]))
        np.testing.assert_allclose(centered["energy"], 0, atol=1e-25)
        np.testing.assert_allclose(centered["original_energy"], centered["mean_energy"], rtol=1e-12)

    def test_known_separable_field_is_rank_one(self):
        image = np.arange(1, 41, dtype=np.float64).reshape(5, 8) / 40
        profile = np.array([.2, .5, .8, .1, .6, .9, .3])
        result = self.analyze(profile[:, None, None] * image[None])
        ranks = result["metadata"]["ranks"]
        one = list(ranks).index(1)
        for blocks in result["analyses"].values():
            np.testing.assert_array_equal(blocks["effective_ranks"], 1)
            np.testing.assert_allclose(blocks["sse"][:, one], 0, atol=1e-12)

    def test_spectra_and_tail_errors_match_direct_numpy_svd(self):
        stack = np.random.default_rng(20).random((7, 5, 8))
        result = self.analyze(stack)
        ranks = result["metadata"]["ranks"]
        for name, blocks in result["analyses"].items():
            for i, bound in enumerate(blocks["bounds"]):
                a = matrix_for(stack, bound)
                if name == "axial_centered":
                    a -= a.mean(axis=1, keepdims=True)
                sigma = np.linalg.svd(a, compute_uv=False)
                expected = np.zeros(blocks["eigenvalues"].shape[1])
                expected[:min(len(sigma), len(expected))] = sigma[:len(expected)]**2
                np.testing.assert_allclose(blocks["eigenvalues"][i], expected, rtol=2e-10, atol=1e-12)
                for j, rank in enumerate(ranks):
                    effective = min(int(rank), int(blocks["max_rank"][i]))
                    self.assertEqual(int(blocks["retained_ranks"][i, j]), effective)
                    np.testing.assert_allclose(blocks["sse"][i, j], np.sum(sigma[effective:]**2), rtol=2e-10, atol=1e-12)

    def test_single_pixel_and_single_slice_centering(self):
        stack = np.random.default_rng(1).random((5, 3, 3))
        result = self.analyze(stack, resolution=2, slab=4)
        for name, blocks in result["analyses"].items():
            for i, bound in enumerate(blocks["bounds"]):
                z0, y0, x0, z1, y1, x1 = map(int, bound)
                expected = min((y1-y0)*(x1-x0), z1-z0-(name == "axial_centered"))
                self.assertEqual(int(blocks["max_rank"][i]), expected)
                if name == "axial_centered" and z1-z0 == 1:
                    self.assertEqual(int(blocks["effective_ranks"][i, 0]), 0)

    def test_batch_size_does_not_change_results(self):
        stack = np.random.default_rng(12).random((7, 8, 9))
        a = self.analyze(stack, batch_size=1)
        b = self.analyze(stack, batch_size=23)
        for name in a["analyses"]:
            for key in a["analyses"][name]:
                np.testing.assert_allclose(a["analyses"][name][key], b["analyses"][name][key], rtol=1e-10, atol=1e-12,
                                           err_msg=f"{name}/{key}")

    def test_explicit_centered_reconstruction_restores_mean(self):
        stack = np.random.default_rng(14).random((7, 5, 8))
        result = self.analyze(stack)
        data = result["analyses"]["axial_centered"]
        for index in (0, len(data["bounds"])-1):
            bound = data["bounds"][index]
            for j, rank in enumerate(result["metadata"]["ranks"]):
                example = reconstruct_block(stack, bound, int(rank), "axial_centered")
                actual_sse = np.square(example["original"] - example["reconstructed"]).sum()
                np.testing.assert_allclose(actual_sse, data["sse"][index, j], atol=1e-12, rtol=1e-10)
                if rank == 0:
                    expected = example["original"].mean(axis=0)
                    np.testing.assert_allclose(example["reconstructed"], np.broadcast_to(expected, example["original"].shape))

    def test_reporting_weights_voxels_and_counts_centered_mean_capacity(self):
        from svd_reporting import save_config_results, save_comparison
        stack = np.random.default_rng(42).random((7, 5, 8))
        result = self.analyze(stack)
        with tempfile.TemporaryDirectory(prefix="svd_test_", dir=Path(__file__).resolve().parent) as directory:
            out = Path(directory)
            rows = save_config_results(result, out / "one_config", make_plots=False)
            save_comparison(rows, out, make_plots=False)
            self.assertTrue((out / "comparison.csv").is_file())
            for analysis, data in result["analyses"].items():
                for j, rank in enumerate(result["metadata"]["ranks"]):
                    row = next(r for r in rows if r["analysis"] == analysis and r["scope"] == "all" and r["rank"] == rank)
                    expected_mse = data["sse"][:, j].sum() / stack.size
                    self.assertAlmostEqual(row["rmse"]**2, expected_mse, places=13)
                    dims = data["bounds"][:, 3:] - data["bounds"][:, :3]
                    pixels = dims[:, 1] * dims[:, 2]
                    expected_params = np.sum(data["retained_ranks"][:, j] * (pixels + dims[:, 0] + 1))
                    if analysis == "axial_centered":
                        expected_params += pixels.sum()
                    self.assertEqual(row["parameter_count"], expected_params)
                text = (out / "one_config" / analysis / "summary.json").read_text(encoding="utf-8")
                self.assertNotIn("NaN", text)
                self.assertNotIn("Infinity", text)
                json.loads(text)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_and_cpu_agree(self):
        try:
            (torch.ones(2, device="cuda") + 1).cpu()
            torch.cuda.synchronize()
        except RuntimeError as exc:
            if "no kernel image" in str(exc) or "not compatible" in str(exc):
                self.skipTest(f"Installed torch cannot execute on {torch.cuda.get_device_name()}: {str(exc).splitlines()[0]}")
            raise
        stack = np.random.default_rng(41).random((8, 9, 11))
        cpu = self.analyze(stack, device="cpu")
        gpu = self.analyze(stack, device="cuda")
        for name in cpu["analyses"]:
            for key in cpu["analyses"][name]:
                np.testing.assert_allclose(cpu["analyses"][name][key], gpu["analyses"][name][key], rtol=1e-9, atol=1e-11,
                                           err_msg=f"{name}/{key}")


if __name__ == "__main__":
    unittest.main()
