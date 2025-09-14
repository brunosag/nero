"""Abstract base classes defining system interfaces."""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .models import OptimizerConfig, TrainingMetrics


class Optimizer(ABC):
    """Abstract base class for all optimization strategies."""

    def __init__(self, config: OptimizerConfig):
        self.config = config

    @abstractmethod
    def optimize(
        self,
        model: nn.Module,
        dataset: "Dataset",
        epochs: int,
        logger: "ExperimentLogger",
    ) -> TrainingMetrics:
        """
        Optimize the given model on the dataset.

        Args:
            model: Neural network model to optimize
            dataset: Dataset to train on
            epochs: Number of training epochs
            logger: Logger for collecting metrics

        Returns:
            TrainingMetrics containing all collected metrics
        """
        pass

    @abstractmethod
    def get_optimizer_type(self) -> str:
        """Return the type of optimizer (e.g., 'gradient', 'neuroevolution')."""
        pass


class Dataset(ABC):
    """Abstract base class for datasets."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the dataset."""
        pass

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Return the number of classes in the dataset."""
        pass

    @property
    @abstractmethod
    def input_shape(self) -> tuple[int, ...]:
        """Return the shape of input samples (excluding batch dimension)."""
        pass

    @abstractmethod
    def get_train_loader(
        self, batch_size: int | None = None, shuffle: bool = True
    ) -> DataLoader:
        """
        Get training data loader.

        Args:
            batch_size: Batch size (uses default if None)
            shuffle: Whether to shuffle the data

        Returns:
            DataLoader for training data
        """
        pass

    @abstractmethod
    def get_test_loader(self, batch_size: int | None = None) -> DataLoader:
        """
        Get test data loader.

        Args:
            batch_size: Batch size (uses default if None)

        Returns:
            DataLoader for test data
        """
        pass

    @abstractmethod
    def get_sample(self, index: int) -> tuple[torch.Tensor, int]:
        """
        Get a single sample by index.

        Args:
            index: Sample index

        Returns:
            Tuple of (sample, target)
        """
        pass

    @abstractmethod
    def get_sample_batch(
        self, batch_size: int, indices: list[int] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a batch of samples.

        Args:
            batch_size: Number of samples to return
            indices: Specific indices to sample (random if None)

        Returns:
            Tuple of (samples, targets)
        """
        pass

    @abstractmethod
    def validate_integrity(self) -> bool:
        """
        Validate dataset integrity (checksums, completeness, etc.).

        Returns:
            True if dataset is valid, False otherwise
        """
        pass


class ExperimentTracker(ABC):
    """Abstract base class for external experiment tracking."""

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """
        Log metrics for a specific step.

        Args:
            metrics: Dictionary of metric names to values
            step: Step number (e.g., epoch)
        """
        pass

    @abstractmethod
    def log_hyperparameters(self, params: dict[str, Any]) -> None:
        """
        Log hyperparameters for the experiment.

        Args:
            params: Dictionary of hyperparameter names to values
        """
        pass

    @abstractmethod
    def log_artifact(self, file_path: str, artifact_type: str = "model") -> None:
        """
        Log an artifact (model checkpoint, figure, etc.).

        Args:
            file_path: Path to the artifact file
            artifact_type: Type of artifact (e.g., "model", "figure")
        """
        pass

    @abstractmethod
    def finish(self) -> None:
        """Finish the experiment tracking session."""
        pass


class ExperimentLogger(ABC):
    """Abstract base class for experiment logging."""

    def __init__(
        self,
        experiment_id: str,
        run_id: int,
        external_tracker: ExperimentTracker | None = None,
    ):
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.external_tracker = external_tracker
        self.metrics_log: list[dict[str, Any]] = []

    @abstractmethod
    def log_epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        """
        Log metrics for a specific epoch.

        Args:
            epoch: Epoch number
            metrics: Dictionary of metric names to values
        """
        pass

    @abstractmethod
    def log_hyperparameters(self, params: dict[str, Any]) -> None:
        """
        Log hyperparameters for the experiment.

        Args:
            params: Dictionary of hyperparameter names to values
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_metrics(self) -> TrainingMetrics:
        """
        Convert logged data to TrainingMetrics object.

        Returns:
            TrainingMetrics containing all logged data
        """
        pass
