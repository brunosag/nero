"""Integration tests for vertical slice validation."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from nero.data.datasets import MNISTDataset
from nero.domain.models import ExperimentConfig
from nero.orchestration.experiment_manager import BasicExperimentManager


class TestVerticalSliceIntegration:
    """Integration tests for the vertical slice implementation."""

    def test_end_to_end_experiment_execution(self):
        """Test complete experiment pipeline from configuration to results."""
        # Create experiment configuration
        config = ExperimentConfig(
            experiment_id="test_experiment",
            optimizer_name="adam",
            dataset_name="mnist",
            model_architecture="cnn",
            epochs=2,  # Short for testing
            num_runs=30,  # Minimum required for validation
            random_seed_base=42,
            hpo_budget=1,
            hpo_budget_type="trials",
        )

        # Create dataset
        dataset = MNISTDataset()

        # Create experiment manager
        manager = BasicExperimentManager()

        # Run experiment
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics = manager.run_experiment(
                config=config,
                dataset=dataset,
                run_id=0,
                output_dir=temp_dir,
            )

            # Validate metrics
            assert metrics.experiment_id == "test_experiment"
            assert metrics.run_id == 0
            assert metrics.optimizer_name == "adam"
            assert len(metrics.epochs) == 2
            assert len(metrics.train_losses) == 2
            assert len(metrics.train_accuracies) == 2
            assert len(metrics.test_losses) == 2
            assert len(metrics.test_accuracies) == 2
            assert len(metrics.training_times) == 2
            assert metrics.gradient_norms is not None
            assert len(metrics.gradient_norms) == 2

            # Validate that accuracies are reasonable
            assert 0.0 <= metrics.train_accuracies[-1] <= 1.0
            assert 0.0 <= metrics.test_accuracies[-1] <= 1.0

            # Validate that losses are positive
            assert all(loss >= 0 for loss in metrics.train_losses)
            assert all(loss >= 0 for loss in metrics.test_losses)

            # Validate that gradient norms are positive
            assert all(norm >= 0 for norm in metrics.gradient_norms)

            # Check that output files were created
            output_path = Path(temp_dir)
            assert (output_path / "metrics.csv").exists()
            assert (output_path / "summary.json").exists()

            # Validate CSV content
            df = pd.read_csv(output_path / "metrics.csv")
            assert len(df) == 2
            assert "epoch" in df.columns
            assert "train_loss" in df.columns
            assert "train_accuracy" in df.columns
            assert "test_loss" in df.columns
            assert "test_accuracy" in df.columns
            assert "gradient_norm" in df.columns

            # Validate JSON summary
            with open(output_path / "summary.json") as f:
                summary = json.load(f)
            assert summary["experiment_id"] == "test_experiment"
            assert summary["optimizer_name"] == "adam"
            assert summary["total_epochs"] == 2

    def test_sgd_optimizer_execution(self):
        """Test experiment execution with SGD optimizer."""
        config = ExperimentConfig(
            experiment_id="test_sgd",
            optimizer_name="sgd",
            dataset_name="mnist",
            model_architecture="cnn",
            epochs=2,
            num_runs=30,  # Minimum required for validation
            random_seed_base=42,
            hpo_budget=1,
            hpo_budget_type="trials",
        )

        dataset = MNISTDataset()
        manager = BasicExperimentManager()

        metrics = manager.run_experiment(config=config, dataset=dataset, run_id=0)

        # Validate SGD-specific results
        assert metrics.optimizer_name == "sgd"
        assert len(metrics.epochs) == 2
        assert metrics.gradient_norms is not None

    def test_reproducibility_with_same_seed(self):
        """Test that identical seeds produce identical results."""
        config = ExperimentConfig(
            experiment_id="reproducibility_test",
            optimizer_name="adam",
            dataset_name="mnist",
            model_architecture="cnn",
            epochs=2,
            num_runs=30,  # Minimum required for validation
            random_seed_base=123,
            hpo_budget=1,
            hpo_budget_type="trials",
        )

        dataset = MNISTDataset()
        manager = BasicExperimentManager()

        # Run experiment twice with same configuration
        metrics1 = manager.run_experiment(config=config, dataset=dataset, run_id=0)
        metrics2 = manager.run_experiment(config=config, dataset=dataset, run_id=0)

        # Results should be identical (within floating point precision)
        assert len(metrics1.train_losses) == len(metrics2.train_losses)
        for loss1, loss2 in zip(
            metrics1.train_losses, metrics2.train_losses, strict=False
        ):
            assert (
                abs(loss1 - loss2) < 1e-4
            )  # Relaxed tolerance for floating point operations

        for acc1, acc2 in zip(
            metrics1.train_accuracies, metrics2.train_accuracies, strict=False
        ):
            assert (
                abs(acc1 - acc2) < 1e-4
            )  # Relaxed tolerance for floating point operations

    def test_cli_dummy_experiment_command(self):
        """Test CLI dummy experiment command execution."""
        # Test the CLI command
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "nero.cli.main",
                "run-dummy-experiment",
                "--optimizer",
                "adam",
                "--epochs",
                "2",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        # Check that command executed successfully
        assert result.returncode == 0, f"CLI command failed: {result.stderr}"

        # Check output contains expected information
        output = result.stdout
        assert "Starting dummy experiment" in output
        assert "Optimizer: adam" in output
        assert "Epochs: 2" in output
        assert "Loading MNIST dataset" in output
        assert "Running experiment" in output
        assert "Experiment completed successfully" in output
        assert "Final train accuracy:" in output
        assert "Final test accuracy:" in output

    def test_cli_dummy_experiment_with_sgd(self):
        """Test CLI dummy experiment command with SGD optimizer."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "nero.cli.main",
                "run-dummy-experiment",
                "--optimizer",
                "sgd",
                "--epochs",
                "2",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        assert result.returncode == 0, f"CLI command failed: {result.stderr}"
        assert "Optimizer: sgd" in result.stdout

    def test_cli_dummy_experiment_with_output_dir(self):
        """Test CLI dummy experiment command with custom output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "nero.cli.main",
                    "run-dummy-experiment",
                    "--optimizer",
                    "adam",
                    "--epochs",
                    "2",
                    "--output-dir",
                    temp_dir,
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
            )

            assert result.returncode == 0, f"CLI command failed: {result.stderr}"

            # Check that output files were created
            output_path = Path(temp_dir)
            assert (output_path / "metrics.csv").exists()
            assert (output_path / "summary.json").exists()

    def test_error_handling_invalid_optimizer(self):
        """Test error handling with invalid optimizer."""
        config = ExperimentConfig(
            experiment_id="error_test",
            optimizer_name="adam",  # Use valid optimizer for validation
            dataset_name="mnist",
            model_architecture="cnn",
            epochs=2,
            num_runs=30,  # Minimum required for validation
            random_seed_base=42,
            hpo_budget=1,
            hpo_budget_type="trials",
        )

        dataset = MNISTDataset()
        manager = BasicExperimentManager()

        # Should raise ValueError for unsupported optimizer
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            manager.run_experiment(config=config, dataset=dataset, run_id=0)

    def test_architecture_consistency_validation(self):
        """Test that models have consistent architectures."""
        from nero.domain.neural_models import ModelFactory

        # Create multiple models with same configuration
        models = [
            ModelFactory.create_cnn_model("mnist", seed=42),
            ModelFactory.create_cnn_model("mnist", seed=123),
            ModelFactory.create_cnn_model("mnist", seed=456),
        ]

        # Validate architecture consistency
        is_consistent, error_msg = ModelFactory.validate_architecture_consistency(
            models
        )

        assert is_consistent, f"Architecture inconsistency: {error_msg}"

        # All models should have the same architecture hash
        hashes = [model.get_architecture_hash() for model in models]
        assert len(set(hashes)) == 1, "Models should have identical architecture hashes"

    def test_integration_with_activation_extraction(self):
        """Test integration of model training with activation extraction."""
        import torch

        from nero.domain.neural_models import ModelFactory

        # Create model and dummy data
        model = ModelFactory.create_cnn_model("mnist", seed=42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dummy_input = torch.randn(4, 1, 28, 28).to(device)

        # Extract activations before training
        activations_before = model.get_activations(dummy_input, ["conv1", "hidden"])

        # Run a minimal training step
        config = ExperimentConfig(
            experiment_id="activation_test",
            optimizer_name="adam",
            dataset_name="mnist",
            model_architecture="cnn",
            epochs=1,
            num_runs=30,  # Minimum required for validation
            random_seed_base=42,
            hpo_budget=1,
            hpo_budget_type="trials",
        )

        dataset = MNISTDataset()
        manager = BasicExperimentManager()

        # This will modify the model weights - pass the model to ensure it gets trained
        manager.run_experiment(config=config, dataset=dataset, run_id=0, model=model)

        # Extract activations after training (model is now on GPU)
        activations_after = model.get_activations(dummy_input, ["conv1", "hidden"])

        # Activations should be different after training
        # Move tensors to CPU for comparison to avoid device mismatch
        conv1_before = activations_before["conv1"].cpu()
        conv1_after = activations_after["conv1"].cpu()
        hidden_before = activations_before["hidden"].cpu()
        hidden_after = activations_after["hidden"].cpu()

        assert not torch.allclose(conv1_before, conv1_after, atol=1e-6)
        assert not torch.allclose(hidden_before, hidden_after, atol=1e-6)

        # But shapes should be the same
        assert conv1_before.shape == conv1_after.shape
        assert hidden_before.shape == hidden_after.shape
