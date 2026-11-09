"""Check actual reconstruction, capacity accounting, and Pillow exports."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_svd_core import analyze_local
from svd_reconstruction import reconstruct_slices, block_modes
from svd_presentation import summarize, main


class PresentationTests(unittest.TestCase):
    def test_reconstruction_matches_spectral_errors_and_restores_centered_means(self):
        rng=np.random.default_rng(32)
        stack=rng.random((5,7,8),dtype=np.float32)
        ranks=[0,1,2,4]
        result=analyze_local(stack,3,4,ranks=ranks,device="cpu")
        for analysis in ("raw","axial_centered"):
            out=reconstruct_slices(stack,result,analysis,ranks,list(range(5)),"cpu",3)
            data=result["analyses"][analysis]
            for ri,rank in enumerate(ranks):
                j=result["metadata"]["ranks"].index(rank)
                self.assertAlmostEqual(out["slice_sse"][ri].sum(),data["sse"][:,j].sum(),places=10)
                self.assertAlmostEqual(out["slice_sse"][ri].sum(),
                                       np.square(stack.astype(float)-out["images"][ri]).sum(),places=6)
            np.testing.assert_array_equal(out["images"][-1],stack)

    def test_global_and_selected_slice_paths(self):
        stack=np.random.default_rng(4).random((4,5,6),dtype=np.float32)
        result=analyze_local(stack,6,4,ranks=[1,2,4],device="cpu")
        out=reconstruct_slices(stack,result,"raw",[1,2,4],[1,3],"cpu")
        self.assertEqual(out["images"].shape,(3,2,5,6))
        np.testing.assert_array_equal(out["images"][-1],stack[[1,3]])
        modes=block_modes(stack,[0,0,0,4,5,6],"raw",4)
        self.assertEqual(modes["spatial"].shape,(4,5,6))
        reconstructed=(modes["spatial"].reshape(4,-1).T*modes["sigma"])@modes["axial"]
        np.testing.assert_allclose(reconstructed.T.reshape(stack.shape),stack,atol=1e-12)

    def test_uint8_float32_capacity_counts_mean(self):
        stack=np.ones((4,4,4),np.float32)*.5
        result=analyze_local(stack,4,4,ranks=[0,1],device="cpu")
        row=summarize(result,"axial_centered",0,np.zeros(4),0,"block004x004__slab004")
        self.assertEqual(row["factor_scalar_count"],16)
        self.assertEqual(row["scalar_compression_ratio"],4)
        self.assertEqual(row["uint8_source_to_float32_factor_ratio"],1)

    def test_cli_exports_native_images_global_rows_and_fixed_legend(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root=Path(temp)
            source=root/"input"
            source.mkdir()
            for z in range(3):
                Image.fromarray(np.uint8(np.arange(30).reshape(5,6)+z*5)).save(source/f"slice_{z}.png")
            output=root/"report_results"
            main(["--input",str(source),"--output",str(output),"--resolution","3",
                  "--slab","2","3","--ranks","1","2","--slices","0","2",
                  "--device","cpu","--template-blocks","1","--template-modes","2"])
            with (output/"cross_comparison.csv").open(encoding="utf-8-sig") as handle:
                rows=list(csv.DictReader(handle))
            self.assertEqual(len(set(r["configuration"] for r in rows)),4)
            self.assertTrue(any(r["kind"]=="global_xy" for r in rows))
            images=list(output.glob("configurations/**/**reconstruction_native.png"))
            self.assertTrue(images)
            for path in images:
                with Image.open(path) as image:
                    self.assertEqual(image.size,(6,5))
            self.assertTrue((output/"00_error_color_scale.png").exists())
            self.assertTrue(list(output.glob("tables/*.csv")))


if __name__=="__main__":
    unittest.main()
