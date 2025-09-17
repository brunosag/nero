"""Basic experiment manager for single experiment execution."""

from pathlib import Path

import pandas as pd
import torch

from nero.domain.interfaces import Dataset
from nero.domain.models import ExperimentConfig, OptimizerConfig, TrainingMetrics
from nero.domain.neural_models import ModelFactory
from nero.logging.experiment_logger import BasicExperimentLogger
from nero.optimizers.gradient_optimizer import GradientOptimizer
from nero.orchestration.seed_manager import SeedManager


class BasicExperimentManager:
    """Basic experiment manager for single optimizer execution."""

    def __init__(self, seed_manager: SeedManager | None = None):
        self.seed_manager = seed_manager or SeedManager(42)

    def run_experiment(
        self,
        config: ExperimentConfig,
        dataset: Dataset,
        run_id: int = 0,
        output_dir: str | None = None,
        model=None,
    ) -> TrainingMetrics:
        """
        Run a single experiment.

        Args:
            config: Experiment configuration
            dataset: Dataset to train on
            run_id: Run identifier for this experiment
            output_dir: Optional output directory for results
            model: Optional pre-created model to use (creates new if None)

        Returns:
            TrainingMetrics containing experiment results
        """
        # Get deterministic seeds
        seeds = self.seed_manager.get_experiment_seeds(config.experiment_id, 1)
        model_seed = seeds["model_init"]

        # Set all seeds for complete reproducibility
        self.seed_manager.set_model_init_seed(config.experiment_id, run_id)
        self.seed_manager.set_data_shuffle_seed(config.experiment_id, run_id)

        # Enable deterministic operations for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Create model if not provided
        if model is None:
            model = ModelFactory.create_cnn_model(
                dataset_name=config.dataset_name, seed=model_seed
            )

        # Create optimizer config (minimal for vertical slice)
        optimizer_config = self._create_basic_optimizer_config(config.optimizer_name)

        # Create optimizer
        optimizer = GradientOptimizer(optimizer_config)

        # Create logger
        logger = BasicExperimentLogger(config.experiment_id, run_id)

        # Run training
        metrics = optimizer.optimize(model, dataset, config.epochs, logger)

        # Save results if output directory specified
        if output_dir:
            self._save_results(metrics, output_dir)

        return metrics

    def _create_basic_optimizer_config(self, optimizer_name: str) -> OptimizerConfig:
        """Create basic optimizer configuration for vertical slice."""
        # Default parameters for minimal implementation
        default_params = {
            "adam": {
                "learning_rate": 0.001,
                "beta1": 0.9,
                "beta2": 0.999,
                "weight_decay": 0.0,
            },
            "sgd": {
                "learning_rate": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0,
            },
        }

        params = default_params.get(optimizer_name.lower(), default_params["adam"])

        return OptimizerConfig(
            name=optimizer_name,
            params=params,
            search_space={},
            optimization_history=pd.DataFrame(),
            hpo_method="none",
            budget_used=0,
        )

    def _save_results(self, metrics: TrainingMetrics, output_dir: str) -> None:
        """Save experiment results to output directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save metrics as CSV
        metrics_data = []
        for i, epoch in enumerate(metrics.epochs):
            metrics_data.append(
                {
                    "epoch": epoch,
                    "train_loss": metrics.train_losses[i],
                    "train_accuracy": metrics.train_accuracies[i],
                    "test_loss": metrics.test_losses[i],
                    "test_accuracy": metrics.test_accuracies[i],
                    "training_time": metrics.training_times[i],
                    "gradient_norm": metrics.gradient_norms[i]
                    if metrics.gradient_norms
                    else None,
                }
            )

        df = pd.DataFrame(metrics_data)
        df.to_csv(output_path / "metrics.csv", index=False)

        # Save summary
        summary = {
            "experiment_id": metrics.experiment_id,
            "run_id": metrics.run_id,
            "optimizer_name": metrics.optimizer_name,
            "total_epochs": len(metrics.epochs),
            "final_train_accuracy": metrics.train_accuracies[-1],
            "final_test_accuracy": metrics.test_accuracies[-1],
            "total_training_time": metrics.total_training_time,
            "peak_memory_usage": metrics.peak_memory_usage,
        }

        import json

        with open(output_path / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
