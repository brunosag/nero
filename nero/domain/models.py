"""Core domain models and value objects for NERO."""

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable experiment configuration."""

    experiment_id: str
    optimizer_name: str
    dataset_name: str
    model_architecture: str
    epochs: int
    num_runs: int
    random_seed_base: int
    hpo_budget: int
    hpo_budget_type: str  # 'trials', 'time', 'epochs'

    def __post_init__(self):
        """Validate configuration after initialization."""
        errors = self.validate()
        if errors:
            raise ValueError(f"Invalid ExperimentConfig: {'; '.join(errors)}")

    def validate(self) -> list[str]:
        """Validate configuration and return any errors."""
        errors = []

        if not self.experiment_id or not self.experiment_id.strip():
            errors.append("experiment_id cannot be empty")

        if not self.optimizer_name or not self.optimizer_name.strip():
            errors.append("optimizer_name cannot be empty")

        if self.optimizer_name not in ["adam", "sgd", "leea", "shade-ils"]:
            errors.append("optimizer_name must be one of: adam, sgd, leea, shade-ils")

        if not self.dataset_name or not self.dataset_name.strip():
            errors.append("dataset_name cannot be empty")

        if self.dataset_name not in ["mnist", "cifar-10"]:
            errors.append("dataset_name must be one of: mnist, cifar-10")

        if self.epochs <= 0:
            errors.append("epochs must be positive")

        if self.num_runs < 30:
            errors.append("minimum 30 runs required for statistical validity")

        if self.random_seed_base < 0:
            errors.append("random_seed_base must be non-negative")

        if self.hpo_budget <= 0:
            errors.append("hpo_budget must be positive")

        if self.hpo_budget_type not in ["trials", "time", "epochs"]:
            errors.append("hpo_budget_type must be one of: trials, time, epochs")

        return errors


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimized hyperparameter configuration for an optimizer."""

    name: str
    params: dict[str, Any]
    search_space: dict[str, Any]
    optimization_history: pd.DataFrame
    hpo_method: str = "optuna"
    budget_used: int = 0

    def __post_init__(self):
        """Validate configuration after initialization."""
        errors = self.validate()
        if errors:
            raise ValueError(f"Invalid OptimizerConfig: {'; '.join(errors)}")

    def validate(self) -> list[str]:
        """Validate configuration and return any errors."""
        errors = []

        if not self.name or not self.name.strip():
            errors.append("name cannot be empty")

        if not isinstance(self.params, dict):
            errors.append("params must be a dictionary")

        if not isinstance(self.search_space, dict):
            errors.append("search_space must be a dictionary")

        if not isinstance(self.optimization_history, pd.DataFrame):
            errors.append("optimization_history must be a pandas DataFrame")

        if self.budget_used < 0:
            errors.append("budget_used must be non-negative")

        return errors


@dataclass(frozen=True)
class TrainingMetrics:
    """Complete training run results."""

    experiment_id: str
    run_id: int
    optimizer_name: str
    epochs: list[int]
    train_losses: list[float]
    train_accuracies: list[float]
    test_losses: list[float]
    test_accuracies: list[float]
    training_times: list[float]

    # Optimizer-specific metrics
    gradient_norms: list[float] | None = None
    population_diversity: list[float] | None = None

    # Resource usage
    peak_memory_usage: float = 0.0
    total_training_time: float = 0.0

    def __post_init__(self):
        """Validate metrics after initialization."""
        errors = self.validate()
        if errors:
            raise ValueError(f"Invalid TrainingMetrics: {'; '.join(errors)}")

    def validate(self) -> list[str]:
        """Validate metrics and return any errors."""
        errors = []

        if not self.experiment_id or not self.experiment_id.strip():
            errors.append("experiment_id cannot be empty")

        if self.run_id < 0:
            errors.append("run_id must be non-negative")

        if not self.optimizer_name or not self.optimizer_name.strip():
            errors.append("optimizer_name cannot be empty")

        # Check that all metric lists have the same length
        metric_lists = [
            self.epochs,
            self.train_losses,
            self.train_accuracies,
            self.test_losses,
            self.test_accuracies,
            self.training_times,
        ]

        if not all(len(lst) == len(metric_lists[0]) for lst in metric_lists):
            errors.append("all metric lists must have the same length")

        if len(self.epochs) == 0:
            errors.append("metrics cannot be empty")

        # Validate that epochs are sequential
        if self.epochs != list(range(len(self.epochs))):
            errors.append("epochs must be sequential starting from 0")

        # Validate that losses and accuracies are non-negative
        if any(loss < 0 for loss in self.train_losses + self.test_losses):
            errors.append("losses must be non-negative")

        if any(
            acc < 0 or acc > 1 for acc in self.train_accuracies + self.test_accuracies
        ):
            errors.append("accuracies must be between 0 and 1")

        if any(time < 0 for time in self.training_times):
            errors.append("training times must be non-negative")

        if self.peak_memory_usage < 0:
            errors.append("peak_memory_usage must be non-negative")

        if self.total_training_time < 0:
            errors.append("total_training_time must be non-negative")

        return errors

    @classmethod
    def from_log_data(
        cls, experiment_id: str, run_id: int, metrics_log: list[dict[str, float]]
    ) -> "TrainingMetrics":
        """Create TrainingMetrics from logged data."""
        if not metrics_log:
            raise ValueError("metrics_log cannot be empty")

        # Extract optimizer name from first log entry
        optimizer_name = str(metrics_log[0].get("optimizer_name", "unknown"))

        # Extract metrics
        epochs = [int(entry["epoch"]) for entry in metrics_log]
        train_losses = [entry["train_loss"] for entry in metrics_log]
        train_accuracies = [entry["train_accuracy"] for entry in metrics_log]
        test_losses = [entry["test_loss"] for entry in metrics_log]
        test_accuracies = [entry["test_accuracy"] for entry in metrics_log]
        training_times = [entry.get("training_time", 0.0) for entry in metrics_log]

        # Extract optimizer-specific metrics
        gradient_norms = None
        population_diversity = None

        if "gradient_norm" in metrics_log[0]:
            gradient_norms = [entry["gradient_norm"] for entry in metrics_log]

        if "population_diversity" in metrics_log[0]:
            population_diversity = [
                entry["population_diversity"] for entry in metrics_log
            ]

        # Calculate total training time
        total_training_time = sum(training_times)

        return cls(
            experiment_id=experiment_id,
            run_id=run_id,
            optimizer_name=optimizer_name,
            epochs=epochs,
            train_losses=train_losses,
            train_accuracies=train_accuracies,
            test_losses=test_losses,
            test_accuracies=test_accuracies,
            training_times=training_times,
            gradient_norms=gradient_norms,
            population_diversity=population_diversity,
            total_training_time=total_training_time,
        )
