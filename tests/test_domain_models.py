"""Unit tests for domain models and value objects."""

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from nero.domain.models import ExperimentConfig, OptimizerConfig, TrainingMetrics


class TestExperimentConfig:
    """Test cases for ExperimentConfig value object."""

    def test_valid_config_creation(self):
        """Test creating a valid ExperimentConfig."""
        config = ExperimentConfig(
            experiment_id="test_exp_001",
            optimizer_name="adam",
            dataset_name="mnist",
            model_architecture="cnn",
            epochs=100,
            num_runs=30,
            random_seed_base=42,
            hpo_budget=50,
            hpo_budget_type="trials",
        )

        assert config.experiment_id == "test_exp_001"
        assert config.optimizer_name == "adam"
        assert config.dataset_name == "mnist"
        assert config.epochs == 100
        assert config.num_runs == 30

    def test_immutability(self):
        """Test that ExperimentConfig is immutable."""
        config = ExperimentConfig(
            experiment_id="test_exp_001",
            optimizer_name="adam",
            dataset_name="mnist",
            model_architecture="cnn",
            epochs=100,
            num_runs=30,
            random_seed_base=42,
            hpo_budget=50,
            hpo_budget_type="trials",
        )

        with pytest.raises(FrozenInstanceError):
            config.epochs = 200

    def test_empty_experiment_id_validation(self):
        """Test validation fails for empty experiment_id."""
        with pytest.raises(ValueError, match="experiment_id cannot be empty"):
            ExperimentConfig(
                experiment_id="",
                optimizer_name="adam",
                dataset_name="mnist",
                model_architecture="cnn",
                epochs=100,
                num_runs=30,
                random_seed_base=42,
                hpo_budget=50,
                hpo_budget_type="trials",
            )

    def test_invalid_optimizer_name_validation(self):
        """Test validation fails for invalid optimizer_name."""
        with pytest.raises(ValueError, match="optimizer_name must be one of"):
            ExperimentConfig(
                experiment_id="test_exp_001",
                optimizer_name="invalid_optimizer",
                dataset_name="mnist",
                model_architecture="cnn",
                epochs=100,
                num_runs=30,
                random_seed_base=42,
                hpo_budget=50,
                hpo_budget_type="trials",
            )

    def test_invalid_dataset_name_validation(self):
        """Test validation fails for invalid dataset_name."""
        with pytest.raises(ValueError, match="dataset_name must be one of"):
            ExperimentConfig(
                experiment_id="test_exp_001",
                optimizer_name="adam",
                dataset_name="invalid_dataset",
                model_architecture="cnn",
                epochs=100,
                num_runs=30,
                random_seed_base=42,
                hpo_budget=50,
                hpo_budget_type="trials",
            )

    def test_negative_epochs_validation(self):
        """Test validation fails for negative epochs."""
        with pytest.raises(ValueError, match="epochs must be positive"):
            ExperimentConfig(
                experiment_id="test_exp_001",
                optimizer_name="adam",
                dataset_name="mnist",
                model_architecture="cnn",
                epochs=-10,
                num_runs=30,
                random_seed_base=42,
                hpo_budget=50,
                hpo_budget_type="trials",
            )

    def test_insufficient_runs_validation(self):
        """Test validation fails for insufficient number of runs."""
        with pytest.raises(ValueError, match="minimum 30 runs required"):
            ExperimentConfig(
                experiment_id="test_exp_001",
                optimizer_name="adam",
                dataset_name="mnist",
                model_architecture="cnn",
                epochs=100,
                num_runs=10,
                random_seed_base=42,
                hpo_budget=50,
                hpo_budget_type="trials",
            )

    def test_negative_seed_validation(self):
        """Test validation fails for negative random seed."""
        with pytest.raises(ValueError, match="random_seed_base must be non-negative"):
            ExperimentConfig(
                experiment_id="test_exp_001",
                optimizer_name="adam",
                dataset_name="mnist",
                model_architecture="cnn",
                epochs=100,
                num_runs=30,
                random_seed_base=-1,
                hpo_budget=50,
                hpo_budget_type="trials",
            )

    def test_invalid_hpo_budget_type_validation(self):
        """Test validation fails for invalid hpo_budget_type."""
        with pytest.raises(ValueError, match="hpo_budget_type must be one of"):
            ExperimentConfig(
                experiment_id="test_exp_001",
                optimizer_name="adam",
                dataset_name="mnist",
                model_architecture="cnn",
                epochs=100,
                num_runs=30,
                random_seed_base=42,
                hpo_budget=50,
                hpo_budget_type="invalid_type",
            )


class TestOptimizerConfig:
    """Test cases for OptimizerConfig value object."""

    def test_valid_config_creation(self):
        """Test creating a valid OptimizerConfig."""
        history_df = pd.DataFrame({"trial": [0, 1, 2], "value": [0.8, 0.85, 0.82]})

        config = OptimizerConfig(
            name="adam",
            params={"learning_rate": 0.001, "beta1": 0.9},
            search_space={"learning_rate": (1e-5, 1e-1), "beta1": (0.8, 0.99)},
            optimization_history=history_df,
            hpo_method="optuna",
            budget_used=25,
        )

        assert config.name == "adam"
        assert config.params["learning_rate"] == 0.001
        assert config.hpo_method == "optuna"
        assert config.budget_used == 25

    def test_immutability(self):
        """Test that OptimizerConfig is immutable."""
        history_df = pd.DataFrame({"trial": [0], "value": [0.8]})

        config = OptimizerConfig(
            name="adam",
            params={"learning_rate": 0.001},
            search_space={"learning_rate": (1e-5, 1e-1)},
            optimization_history=history_df,
        )

        with pytest.raises(FrozenInstanceError):
            config.name = "sgd"

    def test_empty_name_validation(self):
        """Test validation fails for empty name."""
        history_df = pd.DataFrame({"trial": [0], "value": [0.8]})

        with pytest.raises(ValueError, match="name cannot be empty"):
            OptimizerConfig(
                name="",
                params={"learning_rate": 0.001},
                search_space={"learning_rate": (1e-5, 1e-1)},
                optimization_history=history_df,
            )

    def test_invalid_params_type_validation(self):
        """Test validation fails for non-dict params."""
        history_df = pd.DataFrame({"trial": [0], "value": [0.8]})

        with pytest.raises(ValueError, match="params must be a dictionary"):
            OptimizerConfig(
                name="adam",
                params="invalid",
                search_space={"learning_rate": (1e-5, 1e-1)},
                optimization_history=history_df,
            )

    def test_invalid_search_space_type_validation(self):
        """Test validation fails for non-dict search_space."""
        history_df = pd.DataFrame({"trial": [0], "value": [0.8]})

        with pytest.raises(ValueError, match="search_space must be a dictionary"):
            OptimizerConfig(
                name="adam",
                params={"learning_rate": 0.001},
                search_space="invalid",
                optimization_history=history_df,
            )

    def test_invalid_optimization_history_type_validation(self):
        """Test validation fails for non-DataFrame optimization_history."""
        with pytest.raises(
            ValueError, match="optimization_history must be a pandas DataFrame"
        ):
            OptimizerConfig(
                name="adam",
                params={"learning_rate": 0.001},
                search_space={"learning_rate": (1e-5, 1e-1)},
                optimization_history="invalid",
            )

    def test_negative_budget_used_validation(self):
        """Test validation fails for negative budget_used."""
        history_df = pd.DataFrame({"trial": [0], "value": [0.8]})

        with pytest.raises(ValueError, match="budget_used must be non-negative"):
            OptimizerConfig(
                name="adam",
                params={"learning_rate": 0.001},
                search_space={"learning_rate": (1e-5, 1e-1)},
                optimization_history=history_df,
                budget_used=-5,
            )


class TestTrainingMetrics:
    """Test cases for TrainingMetrics value object."""

    def test_valid_metrics_creation(self):
        """Test creating valid TrainingMetrics."""
        metrics = TrainingMetrics(
            experiment_id="test_exp_001",
            run_id=0,
            optimizer_name="adam",
            epochs=[0, 1, 2],
            train_losses=[1.0, 0.8, 0.6],
            train_accuracies=[0.3, 0.5, 0.7],
            test_losses=[1.1, 0.9, 0.7],
            test_accuracies=[0.25, 0.45, 0.65],
            training_times=[10.0, 9.5, 9.0],
            gradient_norms=[2.5, 2.0, 1.8],
            peak_memory_usage=1024.0,
            total_training_time=28.5,
        )

        assert metrics.experiment_id == "test_exp_001"
        assert metrics.run_id == 0
        assert len(metrics.epochs) == 3
        assert metrics.gradient_norms == [2.5, 2.0, 1.8]

    def test_immutability(self):
        """Test that TrainingMetrics is immutable."""
        metrics = TrainingMetrics(
            experiment_id="test_exp_001",
            run_id=0,
            optimizer_name="adam",
            epochs=[0, 1, 2],
            train_losses=[1.0, 0.8, 0.6],
            train_accuracies=[0.3, 0.5, 0.7],
            test_losses=[1.1, 0.9, 0.7],
            test_accuracies=[0.25, 0.45, 0.65],
            training_times=[10.0, 9.5, 9.0],
        )

        with pytest.raises(FrozenInstanceError):
            metrics.run_id = 1

    def test_empty_experiment_id_validation(self):
        """Test validation fails for empty experiment_id."""
        with pytest.raises(ValueError, match="experiment_id cannot be empty"):
            TrainingMetrics(
                experiment_id="",
                run_id=0,
                optimizer_name="adam",
                epochs=[0, 1, 2],
                train_losses=[1.0, 0.8, 0.6],
                train_accuracies=[0.3, 0.5, 0.7],
                test_losses=[1.1, 0.9, 0.7],
                test_accuracies=[0.25, 0.45, 0.65],
                training_times=[10.0, 9.5, 9.0],
            )

    def test_negative_run_id_validation(self):
        """Test validation fails for negative run_id."""
        with pytest.raises(ValueError, match="run_id must be non-negative"):
            TrainingMetrics(
                experiment_id="test_exp_001",
                run_id=-1,
                optimizer_name="adam",
                epochs=[0, 1, 2],
                train_losses=[1.0, 0.8, 0.6],
                train_accuracies=[0.3, 0.5, 0.7],
                test_losses=[1.1, 0.9, 0.7],
                test_accuracies=[0.25, 0.45, 0.65],
                training_times=[10.0, 9.5, 9.0],
            )

    def test_mismatched_metric_lengths_validation(self):
        """Test validation fails for mismatched metric list lengths."""
        with pytest.raises(
            ValueError, match="all metric lists must have the same length"
        ):
            TrainingMetrics(
                experiment_id="test_exp_001",
                run_id=0,
                optimizer_name="adam",
                epochs=[0, 1, 2],
                train_losses=[1.0, 0.8],  # Different length
                train_accuracies=[0.3, 0.5, 0.7],
                test_losses=[1.1, 0.9, 0.7],
                test_accuracies=[0.25, 0.45, 0.65],
                training_times=[10.0, 9.5, 9.0],
            )

    def test_empty_metrics_validation(self):
        """Test validation fails for empty metrics."""
        with pytest.raises(ValueError, match="metrics cannot be empty"):
            TrainingMetrics(
                experiment_id="test_exp_001",
                run_id=0,
                optimizer_name="adam",
                epochs=[],
                train_losses=[],
                train_accuracies=[],
                test_losses=[],
                test_accuracies=[],
                training_times=[],
            )

    def test_non_sequential_epochs_validation(self):
        """Test validation fails for non-sequential epochs."""
        with pytest.raises(
            ValueError, match="epochs must be sequential starting from 0"
        ):
            TrainingMetrics(
                experiment_id="test_exp_001",
                run_id=0,
                optimizer_name="adam",
                epochs=[0, 2, 3],  # Missing epoch 1
                train_losses=[1.0, 0.8, 0.6],
                train_accuracies=[0.3, 0.5, 0.7],
                test_losses=[1.1, 0.9, 0.7],
                test_accuracies=[0.25, 0.45, 0.65],
                training_times=[10.0, 9.5, 9.0],
            )

    def test_negative_losses_validation(self):
        """Test validation fails for negative losses."""
        with pytest.raises(ValueError, match="losses must be non-negative"):
            TrainingMetrics(
                experiment_id="test_exp_001",
                run_id=0,
                optimizer_name="adam",
                epochs=[0, 1, 2],
                train_losses=[1.0, -0.8, 0.6],  # Negative loss
                train_accuracies=[0.3, 0.5, 0.7],
                test_losses=[1.1, 0.9, 0.7],
                test_accuracies=[0.25, 0.45, 0.65],
                training_times=[10.0, 9.5, 9.0],
            )

    def test_invalid_accuracies_validation(self):
        """Test validation fails for accuracies outside [0, 1] range."""
        with pytest.raises(ValueError, match="accuracies must be between 0 and 1"):
            TrainingMetrics(
                experiment_id="test_exp_001",
                run_id=0,
                optimizer_name="adam",
                epochs=[0, 1, 2],
                train_losses=[1.0, 0.8, 0.6],
                train_accuracies=[0.3, 1.5, 0.7],  # Accuracy > 1
                test_losses=[1.1, 0.9, 0.7],
                test_accuracies=[0.25, 0.45, 0.65],
                training_times=[10.0, 9.5, 9.0],
            )

    def test_from_log_data_creation(self):
        """Test creating TrainingMetrics from log data."""
        log_data = [
            {
                "epoch": 0,
                "optimizer_name": "adam",
                "train_loss": 1.0,
                "train_accuracy": 0.3,
                "test_loss": 1.1,
                "test_accuracy": 0.25,
                "training_time": 10.0,
                "gradient_norm": 2.5,
            },
            {
                "epoch": 1,
                "optimizer_name": "adam",
                "train_loss": 0.8,
                "train_accuracy": 0.5,
                "test_loss": 0.9,
                "test_accuracy": 0.45,
                "training_time": 9.5,
                "gradient_norm": 2.0,
            },
        ]

        metrics = TrainingMetrics.from_log_data("test_exp_001", 0, log_data)

        assert metrics.experiment_id == "test_exp_001"
        assert metrics.run_id == 0
        assert metrics.optimizer_name == "adam"
        assert metrics.epochs == [0, 1]
        assert metrics.gradient_norms == [2.5, 2.0]
        assert metrics.total_training_time == 19.5

    def test_from_log_data_empty_validation(self):
        """Test from_log_data fails for empty log data."""
        with pytest.raises(ValueError, match="metrics_log cannot be empty"):
            TrainingMetrics.from_log_data("test_exp_001", 0, [])
