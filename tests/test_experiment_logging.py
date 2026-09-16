import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from experiment_logging import SwanLabTracker, _clean_metrics


class _Run:
    def __init__(self):
        self.logged = []
        self.finished = False

    def log(self, metrics, step):
        self.logged.append((metrics, step))

    def finish(self):
        self.finished = True

    def get_url(self):
        return "https://example.invalid/run"


class ExperimentLoggingTests(unittest.TestCase):
    def test_clean_metrics_drops_none_non_finite_and_non_scalar_values(self):
        cleaned = _clean_metrics(
            {
                "integer": 3,
                "boolean": True,
                "finite": 1.25,
                "infinite": float("inf"),
                "missing": None,
                "text": "skip",
            }
        )

        self.assertEqual(cleaned, {"integer": 3, "boolean": 1, "finite": 1.25})

    def test_tracker_respects_interval_and_finishes(self):
        run = _Run()
        tracker = SwanLabTracker(object(), run, log_interval=2)

        tracker.log({"loss": 2.0}, step=1)
        tracker.log({"loss": 1.0}, step=2)
        tracker.finish()

        self.assertEqual(run.logged, [({"loss": 1.0}, 2)])
        self.assertTrue(run.finished)
        self.assertEqual(tracker.url, "https://example.invalid/run")


if __name__ == "__main__":
    unittest.main()
