import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "training"))
sys.path.insert(0, str(PROJECT_ROOT / "utils"))

from ddppo import FeatureBuildWorkspace


class FeatureBuildWorkspaceTests(unittest.TestCase):
    def test_arange_cache_is_bounded_by_device_and_dtype(self):
        workspace = FeatureBuildWorkspace()
        device = torch.device("cpu")

        large = workspace.arange(128, device)
        small = workspace.arange(7, device)
        grown = workspace.arange(256, device)

        self.assertEqual(len(workspace._arange_cache), 1)
        self.assertEqual(large.numel(), 128)
        self.assertEqual(small.tolist(), list(range(7)))
        self.assertEqual(grown.numel(), 256)
        self.assertEqual(next(iter(workspace._arange_cache.values())).numel(), 256)

    def test_clear_scratch_releases_all_shape_dependent_storage(self):
        workspace = FeatureBuildWorkspace()
        workspace.arange(32, torch.device("cpu"))
        workspace.scratch("temporary", (4, 8), torch.device("cpu"), torch.float32)

        workspace.clear_scratch()

        self.assertEqual(workspace._arange_cache, {})
        self.assertEqual(workspace._scratch, {})

if __name__ == "__main__":
    unittest.main()
