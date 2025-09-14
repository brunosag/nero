"""Unit tests for domain interfaces."""

from abc import ABC

import pytest

from nero.domain.interfaces import (
    Dataset,
    ExperimentLogger,
    ExperimentTracker,
    Optimizer,
)


class TestAbstractInterfaces:
    """Test cases for abstract interface definitions."""

    def test_optimizer_is_abstract(self):
        """Test that Optimizer is an abstract base class."""
        assert issubclass(Optimizer, ABC)

        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            Optimizer.__new__(Optimizer)

    def test_dataset_is_abstract(self):
        """Test that Dataset is an abstract base class."""
        assert issubclass(Dataset, ABC)

        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            Dataset()

    def test_experiment_tracker_is_abstract(self):
        """Test that ExperimentTracker is an abstract base class."""
        assert issubclass(ExperimentTracker, ABC)

        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            ExperimentTracker()

    def test_experiment_logger_is_abstract(self):
        """Test that ExperimentLogger is an abstract base class."""
        assert issubclass(ExperimentLogger, ABC)

        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            ExperimentLogger("test", 0)

    def test_optimizer_has_required_methods(self):
        """Test that Optimizer has all required abstract methods."""
        required_methods = ["optimize", "get_optimizer_type"]

        for method_name in required_methods:
            assert hasattr(Optimizer, method_name)
            method = getattr(Optimizer, method_name)
            assert getattr(method, "__isabstractmethod__", False)

    def test_dataset_has_required_methods(self):
        """Test that Dataset has all required abstract methods."""
        required_methods = [
            "get_train_loader",
            "get_test_loader",
            "get_sample",
            "get_sample_batch",
            "validate_integrity",
        ]
        required_properties = ["name", "num_classes", "input_shape"]

        for method_name in required_methods:
            assert hasattr(Dataset, method_name)
            method = getattr(Dataset, method_name)
            assert getattr(method, "__isabstractmethod__", False)

        for prop_name in required_properties:
            assert hasattr(Dataset, prop_name)
            prop = getattr(Dataset, prop_name)
            assert getattr(prop.fget, "__isabstractmethod__", False)

    def test_experiment_tracker_has_required_methods(self):
        """Test that ExperimentTracker has all required abstract methods."""
        required_methods = [
            "log_metrics",
            "log_hyperparameters",
            "log_artifact",
            "finish",
        ]

        for method_name in required_methods:
            assert hasattr(ExperimentTracker, method_name)
            method = getattr(ExperimentTracker, method_name)
            assert getattr(method, "__isabstractmethod__", False)

    def test_experiment_logger_has_required_methods(self):
        """Test that ExperimentLogger has all required abstract methods."""
        required_methods = [
            "log_epoch",
            "log_hyperparameters",
            "save_checkpoint",
            "get_metrics",
        ]

        for method_name in required_methods:
            assert hasattr(ExperimentLogger, method_name)
            method = getattr(ExperimentLogger, method_name)
            assert getattr(method, "__isabstractmethod__", False)
