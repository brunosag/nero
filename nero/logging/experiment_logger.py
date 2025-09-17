"""Basic experiment logger implementation."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from nero.domain.interfaces import ExperimentLogger, ExperimentTracker
from nero.domain.models import TrainingMetrics


class BasicExperimentLogger(ExperimentLogger):
    """Basic implementation of experiment logger."""

    def __init__(
        self,
        experiment_id: str,
        run_id: int,
        external_tracker: ExperimentTracker | None = None,
    ):
        super().__init__(experiment_id, run_id, external_tracker)
        self.hyperparameters: dict[str, Any] = {}

    def log_epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        """
        Log metrics for a specific epoch.

        Args:
            epoch: Epoch number
            metrics: Dictionary of metric names to values
        """
        epoch_data = {"epoch": epoch, **metrics}
        self.metrics_log.append(epoch_data)

        if self.external_tracker:
            self.external_tracker.log_metrics(metrics, step=epoch)

    def log_hyperparameters(self, params: dict[str, Any]) -> None:
        """
        Log hyperparameters for the experiment.

        Args:
            params: Dictionary of hyperparameter names to values
        """
        self.hyperparameters.update(params)

        if self.external_tracker:
            self.external_tracker.log_hyperparameters(params)

    def save_checkpoint(self, model: nn.Module, epoch: int, checkpoint_dir: str) -> str:
        """
        Save model checkpoint.

        Args:
            model: Model to save
            epoch: Current epoch
            checkpoint_dir: Directory to save checkpoint

        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        filename = f"epoch_{epoch:03d}.pt"
        filepath = checkpoint_path / filename

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "experiment_id": self.experiment_id,
                "run_id": self.run_id,
            },
            filepath,
        )

        return str(filepath)

    def get_metrics(self) -> TrainingMetrics:
        """
        Convert logged data to TrainingMetrics object.

        Returns:
            TrainingMetrics containing all logged data
        """
        if not self.metrics_log:
            raise ValueError("No metrics logged")

        return TrainingMetrics.from_log_data(
            self.experiment_id, self.run_id, self.metrics_log
        )
