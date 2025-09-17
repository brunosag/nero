"""Tests for neural network models with activation extraction."""

import pytest
import torch
import torch.nn as nn

from nero.domain.neural_models import CNNModel, ModelFactory


class TestCNNModel:
    """Test cases for CNNModel class."""

    def test_mnist_model_creation(self):
        """Test creating a CNN model for MNIST."""
        model = CNNModel("mnist")

        assert model.dataset_name == "mnist"
        assert model.input_channels == 1
        assert model.input_size == 28
        assert model.num_classes == 10
        assert isinstance(model.conv_layers, nn.Sequential)
        assert isinstance(model.classifier, nn.Sequential)

    def test_cifar10_model_creation(self):
        """Test creating a CNN model for CIFAR-10."""
        model = CNNModel("cifar-10")

        assert model.dataset_name == "cifar-10"
        assert model.input_channels == 3
        assert model.input_size == 32
        assert model.num_classes == 10
        assert isinstance(model.conv_layers, nn.Sequential)
        assert isinstance(model.classifier, nn.Sequential)

    def test_invalid_dataset_raises_error(self):
        """Test that invalid dataset name raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported dataset"):
            CNNModel("invalid_dataset")

    def test_custom_architecture_config(self):
        """Test creating model with custom architecture configuration."""
        custom_config = {
            "conv_layers": [
                {"out_channels": 16, "kernel_size": 5, "padding": 2},
            ],
            "pool_size": 2,
            "hidden_size": 64,
            "dropout_rate": 0.3,
        }

        model = CNNModel("mnist", custom_config)
        assert model.architecture_config == custom_config

    def test_forward_pass_mnist(self):
        """Test forward pass with MNIST-sized input."""
        model = CNNModel("mnist")
        batch_size = 4
        input_tensor = torch.randn(batch_size, 1, 28, 28)

        output = model(input_tensor)

        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()

    def test_forward_pass_cifar10(self):
        """Test forward pass with CIFAR-10-sized input."""
        model = CNNModel("cifar-10")
        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 32, 32)

        output = model(input_tensor)

        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()

    def test_get_activations_single_layer(self):
        """Test activation extraction for a single layer."""
        model = CNNModel("mnist")
        input_tensor = torch.randn(2, 1, 28, 28)

        activations = model.get_activations(input_tensor, ["conv1"])

        assert "conv1" in activations
        assert isinstance(activations["conv1"], torch.Tensor)
        assert activations["conv1"].shape[0] == 2  # Batch size
        assert activations["conv1"].shape[1] == 32  # Output channels

    def test_get_activations_multiple_layers(self):
        """Test activation extraction for multiple layers."""
        model = CNNModel("mnist")
        input_tensor = torch.randn(2, 1, 28, 28)
        layer_names = ["conv1", "conv2", "hidden"]

        activations = model.get_activations(input_tensor, layer_names)

        for layer_name in layer_names:
            assert layer_name in activations
            assert isinstance(activations[layer_name], torch.Tensor)
            assert activations[layer_name].shape[0] == 2  # Batch size

    def test_get_activations_invalid_layer(self):
        """Test activation extraction with invalid layer name."""
        model = CNNModel("mnist")
        input_tensor = torch.randn(2, 1, 28, 28)

        # Should not raise error, just skip invalid layers
        activations = model.get_activations(input_tensor, ["invalid_layer"])

        assert "invalid_layer" not in activations
        assert len(activations) == 0

    def test_get_activations_mixed_valid_invalid(self):
        """Test activation extraction with mix of valid and invalid layer names."""
        model = CNNModel("mnist")
        input_tensor = torch.randn(2, 1, 28, 28)
        layer_names = ["conv1", "invalid_layer", "hidden"]

        activations = model.get_activations(input_tensor, layer_names)

        assert "conv1" in activations
        assert "hidden" in activations
        assert "invalid_layer" not in activations
        assert len(activations) == 2

    def test_architecture_hash_consistency(self):
        """Test that identical models produce identical hashes."""
        model1 = CNNModel("mnist")
        model2 = CNNModel("mnist")

        hash1 = model1.get_architecture_hash()
        hash2 = model2.get_architecture_hash()

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex string length

    def test_architecture_hash_different_datasets(self):
        """Test that different datasets produce different hashes."""
        model_mnist = CNNModel("mnist")
        model_cifar = CNNModel("cifar-10")

        hash_mnist = model_mnist.get_architecture_hash()
        hash_cifar = model_cifar.get_architecture_hash()

        assert hash_mnist != hash_cifar

    def test_architecture_hash_different_configs(self):
        """Test that different configurations produce different hashes."""
        config1 = {
            "conv_layers": [{"out_channels": 32, "kernel_size": 3}],
            "pool_size": 2,
            "hidden_size": 128,
            "dropout_rate": 0.5,
        }
        config2 = {
            "conv_layers": [{"out_channels": 64, "kernel_size": 3}],
            "pool_size": 2,
            "hidden_size": 128,
            "dropout_rate": 0.5,
        }

        model1 = CNNModel("mnist", config1)
        model2 = CNNModel("mnist", config2)

        hash1 = model1.get_architecture_hash()
        hash2 = model2.get_architecture_hash()

        assert hash1 != hash2

    def test_get_model_id(self):
        """Test model ID generation."""
        model = CNNModel("mnist")
        model_id = model.get_model_id()

        assert model_id.startswith("mnist_cnn_")
        assert len(model_id) == len("mnist_cnn_") + 8  # 8-char hash suffix

    def test_get_layer_names(self):
        """Test getting available layer names."""
        model = CNNModel("mnist")
        layer_names = model.get_layer_names()

        expected_layers = ["conv1", "conv2", "hidden", "output"]
        assert layer_names == expected_layers

    def test_activation_hooks_cleanup(self):
        """Test that activation hooks are properly cleaned up."""
        model = CNNModel("mnist")
        input_tensor = torch.randn(1, 1, 28, 28)

        # Extract activations
        model.get_activations(input_tensor, ["conv1"])

        # Hooks should be cleaned up
        assert len(model._hook_handles) == 0
        assert len(model._activation_hooks) == 0


class TestModelFactory:
    """Test cases for ModelFactory class."""

    def test_create_cnn_model_mnist(self):
        """Test creating CNN model through factory for MNIST."""
        model = ModelFactory.create_cnn_model("mnist")

        assert isinstance(model, CNNModel)
        assert model.dataset_name == "mnist"

    def test_create_cnn_model_cifar10(self):
        """Test creating CNN model through factory for CIFAR-10."""
        model = ModelFactory.create_cnn_model("cifar-10")

        assert isinstance(model, CNNModel)
        assert model.dataset_name == "cifar-10"

    def test_create_cnn_model_with_custom_config(self):
        """Test creating CNN model with custom configuration."""
        custom_config = {
            "conv_layers": [{"out_channels": 16, "kernel_size": 3, "padding": 1}],
            "pool_size": 2,
            "hidden_size": 64,
            "dropout_rate": 0.3,
        }

        model = ModelFactory.create_cnn_model("mnist", custom_config)

        assert model.architecture_config == custom_config

    def test_create_cnn_model_with_seed(self):
        """Test creating CNN model with deterministic seed."""
        seed = 42
        model1 = ModelFactory.create_cnn_model("mnist", seed=seed)
        model2 = ModelFactory.create_cnn_model("mnist", seed=seed)

        # Models should have identical weights
        for p1, p2 in zip(model1.parameters(), model2.parameters(), strict=False):
            assert torch.allclose(p1, p2)

    def test_create_cnn_model_different_seeds(self):
        """Test creating CNN models with different seeds."""
        model1 = ModelFactory.create_cnn_model("mnist", seed=42)
        model2 = ModelFactory.create_cnn_model("mnist", seed=123)

        # Models should have different weights
        weights_different = False
        for p1, p2 in zip(model1.parameters(), model2.parameters(), strict=False):
            if not torch.allclose(p1, p2):
                weights_different = True
                break

        assert weights_different

    def test_validate_architecture_consistency_identical(self):
        """Test architecture consistency validation with identical models."""
        models = [
            ModelFactory.create_cnn_model("mnist"),
            ModelFactory.create_cnn_model("mnist"),
            ModelFactory.create_cnn_model("mnist"),
        ]

        is_consistent, error_msg = ModelFactory.validate_architecture_consistency(
            models
        )

        assert is_consistent
        assert error_msg is None

    def test_validate_architecture_consistency_different(self):
        """Test architecture consistency validation with different models."""
        config1 = {
            "conv_layers": [{"out_channels": 32, "kernel_size": 3}],
            "pool_size": 2,
            "hidden_size": 128,
            "dropout_rate": 0.5,
        }
        config2 = {
            "conv_layers": [{"out_channels": 64, "kernel_size": 3}],
            "pool_size": 2,
            "hidden_size": 128,
            "dropout_rate": 0.5,
        }

        models = [
            ModelFactory.create_cnn_model("mnist", config1),
            ModelFactory.create_cnn_model("mnist", config2),
        ]

        is_consistent, error_msg = ModelFactory.validate_architecture_consistency(
            models
        )

        assert not is_consistent
        assert error_msg is not None
        assert "Architecture mismatch" in error_msg

    def test_validate_architecture_consistency_empty_list(self):
        """Test architecture consistency validation with empty list."""
        is_consistent, error_msg = ModelFactory.validate_architecture_consistency([])

        assert is_consistent
        assert error_msg is None

    def test_validate_architecture_consistency_single_model(self):
        """Test architecture consistency validation with single model."""
        models = [ModelFactory.create_cnn_model("mnist")]

        is_consistent, error_msg = ModelFactory.validate_architecture_consistency(
            models
        )

        assert is_consistent
        assert error_msg is None

    def test_get_default_architectures(self):
        """Test getting default architecture configurations."""
        default_archs = ModelFactory.get_default_architectures()

        assert "mnist" in default_archs
        assert "cifar-10" in default_archs

        # Check MNIST config
        mnist_config = default_archs["mnist"]
        assert "conv_layers" in mnist_config
        assert "pool_size" in mnist_config
        assert "hidden_size" in mnist_config
        assert "dropout_rate" in mnist_config

        # Check CIFAR-10 config
        cifar_config = default_archs["cifar-10"]
        assert "conv_layers" in cifar_config
        assert len(cifar_config["conv_layers"]) == 3  # More layers for CIFAR-10

    def test_weight_initialization(self):
        """Test that weight initialization is applied correctly."""
        model = ModelFactory.create_cnn_model("mnist", seed=42)

        # Check that weights are not all zeros (indicating initialization occurred)
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                assert not torch.allclose(
                    module.weight, torch.zeros_like(module.weight)
                )
                if module.bias is not None:
                    # Biases should be initialized to zero
                    assert torch.allclose(module.bias, torch.zeros_like(module.bias))


class TestIntegration:
    """Integration tests for neural models."""

    def test_model_training_compatibility(self):
        """Test that models are compatible with PyTorch training."""
        model = ModelFactory.create_cnn_model("mnist")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Create dummy batch
        batch_size = 4
        inputs = torch.randn(batch_size, 1, 28, 28)
        targets = torch.randint(0, 10, (batch_size,))

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check that gradients were computed
        for param in model.parameters():
            assert param.grad is not None

    def test_activation_extraction_during_training(self):
        """Test activation extraction doesn't interfere with training."""
        model = ModelFactory.create_cnn_model("mnist")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Create dummy batch
        batch_size = 4
        inputs = torch.randn(batch_size, 1, 28, 28)
        targets = torch.randint(0, 10, (batch_size,))

        # Extract activations
        activations = model.get_activations(inputs, ["conv1", "hidden"])

        # Training should still work
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Activations should be extracted correctly
        assert "conv1" in activations
        assert "hidden" in activations
        assert activations["conv1"].shape[0] == batch_size

    def test_model_serialization(self):
        """Test that models can be saved and loaded."""
        import os
        import tempfile

        model = ModelFactory.create_cnn_model("mnist", seed=42)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
            temp_path = f.name

        try:
            # Save model
            torch.save(model.state_dict(), temp_path)

            # Load model
            loaded_model = ModelFactory.create_cnn_model("mnist")
            loaded_model.load_state_dict(torch.load(temp_path))

            # Models should have identical weights
            for p1, p2 in zip(
                model.parameters(), loaded_model.parameters(), strict=False
            ):
                assert torch.allclose(p1, p2)

        finally:
            os.unlink(temp_path)
