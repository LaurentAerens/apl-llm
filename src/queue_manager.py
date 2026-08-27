"""
Training Queue Manager for APL SLM.
Manages persistent training jobs in `training_queue.json` with atomic file operations.
Supports scheduling, prioritizing, reordering, and updating batch training experiments.
"""

import os
import json
import time
import uuid
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

DEFAULT_QUEUE_PATH = "training_queue.json"


class TrainingJob:
    """Represents a scheduled training experiment in the queue."""

    def __init__(
        self,
        exp_name: str,
        version: int = 3,
        model_preset: Optional[str] = "small",
        epochs: int = 30,
        batch_size: int = 16,
        lr: float = 5e-4,
        warmup_epochs: int = 2,
        grad_accum_steps: int = 1,
        early_stopping_patience: int = 6,
        precision: str = "auto",
        dataset_path: str = "data/apl_corpus.txt",
        finetune_from: Optional[str] = None,
        depth_loss_weight: float = 0.2,
        resume: bool = False,
        device: Optional[str] = None,
        job_id: Optional[str] = None,
        status: str = "pending",  # pending, running, completed, failed, cancelled
        created_at: Optional[float] = None,
        started_at: Optional[float] = None,
        finished_at: Optional[float] = None,
        best_val_loss: Optional[float] = None,
        error_message: Optional[str] = None,
    ):
        self.job_id = job_id or str(uuid.uuid4())[:8]
        self.exp_name = exp_name
        self.version = int(version)
        self.model_preset = model_preset or "(none)"
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.warmup_epochs = int(warmup_epochs)
        self.grad_accum_steps = int(grad_accum_steps)
        self.early_stopping_patience = int(early_stopping_patience)
        self.precision = precision
        self.dataset_path = dataset_path
        self.finetune_from = finetune_from
        self.depth_loss_weight = float(depth_loss_weight)
        self.resume = bool(resume)
        self.device = device

        self.status = status
        self.created_at = created_at or time.time()
        self.started_at = started_at
        self.finished_at = finished_at
        self.best_val_loss = best_val_loss
        self.error_message = error_message

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "exp_name": self.exp_name,
            "version": self.version,
            "model_preset": self.model_preset,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "warmup_epochs": self.warmup_epochs,
            "grad_accum_steps": self.grad_accum_steps,
            "early_stopping_patience": self.early_stopping_patience,
            "precision": self.precision,
            "dataset_path": self.dataset_path,
            "finetune_from": self.finetune_from,
            "depth_loss_weight": self.depth_loss_weight,
            "resume": self.resume,
            "device": self.device,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "best_val_loss": self.best_val_loss,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingJob":
        return cls(**data)

    def to_command_args(self) -> List[str]:
        """Converts the training job into CLI arguments for python src/train.py."""
        cmd = [
            "src/train.py",
            "--model_version", str(self.version),
            "--exp_name", self.exp_name,
            "--epochs", str(self.epochs),
            "--batch_size", str(self.batch_size),
            "--lr", str(self.lr),
            "--warmup_epochs", str(self.warmup_epochs),
            "--grad_accum_steps", str(self.grad_accum_steps),
            "--early_stopping_patience", str(self.early_stopping_patience),
            "--data_file", self.dataset_path,
        ]

        if self.model_preset and self.model_preset != "(none)":
            cmd += ["--model_preset", self.model_preset]

        if self.precision and self.precision != "auto":
            cmd += ["--precision", self.precision]

        if self.finetune_from:
            cmd += ["--finetune_from", self.finetune_from]

        if self.depth_loss_weight is not None and self.version in (2, 3):
            cmd += ["--depth_loss_weight", str(self.depth_loss_weight)]

        if self.device and self.device not in ("(auto)", "auto"):
            cmd += ["--device", self.device]

        if self.resume:
            cmd.append("--resume")

        return cmd


class TrainingQueueManager:
    """Handles loading, atomic saving, and mutating the training queue."""

    def __init__(self, queue_path: str = DEFAULT_QUEUE_PATH):
        self.queue_path = Path(queue_path)
        self.jobs: List[TrainingJob] = []
        self.load()

    def load(self) -> List[TrainingJob]:
        if not self.queue_path.exists():
            self.jobs = []
            return self.jobs

        try:
            with open(self.queue_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    raw_jobs = data.get("jobs", [])
                elif isinstance(data, list):
                    raw_jobs = data
                else:
                    raw_jobs = []
                self.jobs = [TrainingJob.from_dict(j) for j in raw_jobs]
        except Exception as e:
            print(f"[!] Warning: Failed to load queue from {self.queue_path}: {e}")
            self.jobs = []
        return self.jobs

    def save(self) -> None:
        """Atomically saves jobs to JSON using a temporary file to prevent corruption."""
        try:
            self.queue_path.parent.mkdir(parents=True, exist_ok=True)
            temp_file = self.queue_path.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump({"jobs": [j.to_dict() for j in self.jobs]}, f, indent=2)
            os.replace(temp_file, self.queue_path)
        except Exception as e:
            print(f"[!] Warning: Failed to save queue to {self.queue_path}: {e}")

    def add_job(self, job: TrainingJob) -> TrainingJob:
        self.jobs.append(job)
        self.save()
        return job

    def remove_job(self, job_id: str) -> bool:
        initial_len = len(self.jobs)
        self.jobs = [j for j in self.jobs if j.job_id != job_id]
        if len(self.jobs) < initial_len:
            self.save()
            return True
        return False

    def move_job(self, job_id: str, direction: int) -> bool:
        """Move job up (direction=-1) or down (direction=1) in queue."""
        for i, j in enumerate(self.jobs):
            if j.job_id == job_id:
                new_idx = i + direction
                if 0 <= new_idx < len(self.jobs):
                    self.jobs[i], self.jobs[new_idx] = self.jobs[new_idx], self.jobs[i]
                    self.save()
                    return True
                break
        return False

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        for j in self.jobs:
            if j.job_id == job_id:
                return j
        return None

    def get_next_pending(self) -> Optional[TrainingJob]:
        for j in self.jobs:
            if j.status == "pending":
                return j
        return None

    def update_job(
        self,
        job_id: str,
        status: Optional[str] = None,
        started_at: Optional[float] = None,
        finished_at: Optional[float] = None,
        best_val_loss: Optional[float] = None,
        error_message: Optional[str] = None,
    ) -> bool:
        job = self.get_job(job_id)
        if not job:
            return False
        if status is not None:
            job.status = status
        if started_at is not None:
            job.started_at = started_at
        if finished_at is not None:
            job.finished_at = finished_at
        if best_val_loss is not None:
            job.best_val_loss = best_val_loss
        if error_message is not None:
            job.error_message = error_message
        self.save()
        return True

    def clear_completed(self) -> int:
        initial_len = len(self.jobs)
        self.jobs = [j for j in self.jobs if j.status in ("pending", "running")]
        cleared = initial_len - len(self.jobs)
        if cleared > 0:
            self.save()
        return cleared

    def reset_failed(self) -> int:
        count = 0
        for j in self.jobs:
            if j.status in ("failed", "cancelled"):
                j.status = "pending"
                j.error_message = None
                count += 1
        if count > 0:
            self.save()
        return count
