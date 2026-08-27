"""Unit tests for TrainingQueueManager and atomic queue file operations."""

import unittest
import tempfile
from pathlib import Path
import sys

src_dir = str(Path(__file__).resolve().parent.parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from queue_manager import TrainingQueueManager, TrainingJob


class TestQueueManager(unittest.TestCase):
    def test_queue_operations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_file = Path(tmpdir) / "training_queue.json"
            qm = TrainingQueueManager(str(queue_file))

            job1 = TrainingJob(exp_name="Exp-1", version=3, epochs=10)
            job2 = TrainingJob(exp_name="Exp-2", version=2, epochs=20)

            qm.add_job(job1)
            qm.add_job(job2)

            self.assertEqual(len(qm.jobs), 2)
            self.assertEqual(qm.get_next_pending().exp_name, "Exp-1")

            # Move job down
            qm.move_job(job1.job_id, 1)
            self.assertEqual(qm.jobs[0].exp_name, "Exp-2")

            # Update job status
            qm.update_job(job2.job_id, status="completed", best_val_loss=1.23)
            self.assertEqual(qm.get_job(job2.job_id).status, "completed")

            # Reload manager from disk
            qm_reloaded = TrainingQueueManager(str(queue_file))
            self.assertEqual(len(qm_reloaded.jobs), 2)
            self.assertEqual(qm_reloaded.get_job(job2.job_id).best_val_loss, 1.23)


if __name__ == "__main__":
    unittest.main()

