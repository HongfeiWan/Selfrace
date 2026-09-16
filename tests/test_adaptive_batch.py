import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from adaptive_batch import AdaptiveBatchSizer


class AdaptiveBatchSizerTests(unittest.TestCase):
    def test_available_samples_are_a_ceiling_not_a_requirement(self):
        sizer = AdaptiveBatchSizer(maximum=32000, minimum=256)

        self.assertEqual(sizer.choose(32000), 32000)
        self.assertEqual(sizer.choose(7000), 7000)
        self.assertEqual(sizer.choose(0), 0)

    def test_oom_backoff_is_persistent_and_stops_at_floor(self):
        sizer = AdaptiveBatchSizer(maximum=32000, minimum=1000)

        self.assertEqual(sizer.backoff(32000), 16000)
        self.assertEqual(sizer.choose(32000), 16000)
        self.assertEqual(sizer.backoff(16000), 8000)
        self.assertEqual(sizer.backoff(2000), 1000)
        self.assertIsNone(sizer.backoff(1000))

    def test_successful_updates_cautiously_restore_capacity(self):
        sizer = AdaptiveBatchSizer(maximum=8000, minimum=500, growth_interval=2)
        self.assertEqual(sizer.backoff(8000), 4000)

        sizer.record_success(attempted=4000, available=8000)
        self.assertEqual(sizer.current, 4000)
        sizer.record_success(attempted=4000, available=8000)
        self.assertEqual(sizer.current, 8000)

    def test_initial_limit_and_state_round_trip(self):
        source = AdaptiveBatchSizer(maximum=8000, minimum=500, growth_interval=3, initial=2000)
        self.assertEqual(source.current, 2000)
        source.backoff(2000)
        source.record_success(attempted=1000, available=8000)

        restored = AdaptiveBatchSizer(maximum=8000, minimum=500, growth_interval=3)
        restored.load_state_dict(source.state_dict())

        self.assertEqual(restored.current, source.current)
        self.assertEqual(restored.state_dict(), source.state_dict())


if __name__ == "__main__":
    unittest.main()
