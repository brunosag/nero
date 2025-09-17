"""Neural network models with activation extraction capabilities."""

import hashlib
from typing import Any

import torch
import torch.nn as nn


class CNNModel(nn.Module):
    """Configurable CNN model for MNIST and CIFAR-10 with activation extraction."""

    def __init__(
        self,
        dataset_name: str,
        architecture_config: dict[str, Any] | None = None,
    ):
        """
        Initialize CNN model.

        Args:
            dataset_name: Name of dataset ('mnist' or 'cifar-10')
            architecture_config: Optional custom architecture configuration
        """
        super().__init__()
        self.dataset_name = dataset_name.lower()
        self.architecture_config = architecture_config or self._get_default_config()

        # Validate dataset
        if self.dataset_name not in ["mnist", "cifar-10"]:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        # Set input parameters based on dataset
        if self.dataset_name == "mnist":
            self.input_channels = 1
            self.input_size = 28
            self.num_classes = 10
        else:  # cifar-10
            self.input_channels = 3
            self.input_size = 32
            self.num_classes = 10

        # Build the network
        self._build_network()

        # Store activation hooks
        self._activation_hooks: dict[str, torch.Tensor] = {}
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []

    def _get_default_config(self) -> dict[str, Any]:
        """Get default architecture configuration."""
        return {
            "conv_layers": [
                {"out_channels": 32, "kernel_size": 3, "padding": 1},
                {"out_channels": 64, "kernel_size": 3, "padding": 1},
            ],
            "pool_size": 2,
            "hidden_size": 128,
            "dropout_rate": 0.5,
        }

    def _build_network(self) -> None:
        """Build the CNN architecture."""
        config = self.architecture_config

        # Convolutional layers
        conv_layers = []
        in_channels = self.input_channels

        for _i, layer_config in enumerate(config["conv_layers"]):
            conv_layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        layer_config["out_channels"],
                        kernel_size=layer_config["kernel_size"],
                        padding=layer_config.get("padding", 0),
                    ),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(config["pool_size"]),
                ]
            )
            in_channels = layer_config["out_channels"]

        self.conv_layers = nn.Sequential(*conv_layers)

        # Calculate the size after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(
                1, self.input_channels, self.input_size, self.input_size
            )
            conv_output = self.conv_layers(dummy_input)
            flattened_size = conv_output.numel()

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(config["dropout_rate"]),
            nn.Linear(flattened_size, config["hidden_size"]),
            nn.ReLU(inplace=True),
            nn.Dropout(config["dropout_rate"]),
            nn.Linear(config["hidden_size"], self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_activations(
        self, x: torch.Tensor, layer_names: list[str]
    ) -> dict[str, torch.Tensor]:
        """
        Extract activations from specified layers.

        Args:
            x: Input tensor
            layer_names: List of layer names to extract activations from

        Returns:
            Dictionary mapping layer names to activation tensors
        """
        # Ensure input tensor is on the same device as the model
        device = next(self.parameters()).device
        x = x.to(device)

        # Clear previous activations
        self._activation_hooks.clear()
        self._remove_hooks()

        # Register hooks for requested layers
        for layer_name in layer_names:
            layer = self._get_layer_by_name(layer_name)
            if layer is not None:
                handle = layer.register_forward_hook(self._create_hook_fn(layer_name))
                self._hook_handles.append(handle)

        # Forward pass to collect activations
        with torch.no_grad():
            _ = self.forward(x)

        # Clean up hooks and get result
        result = self._activation_hooks.copy()
        self._remove_hooks()
        self._activation_hooks.clear()

        return result

    def _get_layer_by_name(self, layer_name: str) -> nn.Module | None:
        """Get layer by name."""
        # Define available layers
        layer_map = {
            "conv1": self.conv_layers[0],  # First conv layer
            "conv2": self.conv_layers[3]
            if len(self.conv_layers) > 3
            else None,  # Second conv layer
            "hidden": self.classifier[1],  # Hidden linear layer
            "output": self.classifier[-1],  # Output layer
        }

        return layer_map.get(layer_name)

    def _create_hook_fn(self, layer_name: str):
        """Create hook function for a specific layer."""

        def hook_fn(module, input, output):
            self._activation_hooks[layer_name] = output.detach().clone()

        return hook_fn

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def get_architecture_hash(self) -> str:
        """
        Generate a hash of the architecture for fair comparison validation.

        Returns:
            SHA-256 hash of the architecture configuration
        """
        # Create a deterministic string representation of the architecture
        arch_str = (
            f"dataset:{self.dataset_name}|"
            f"input_channels:{self.input_channels}|"
            f"input_size:{self.input_size}|"
            f"num_classes:{self.num_classes}|"
            f"config:{str(sorted(self.architecture_config.items()))}"
        )

        return hashlib.sha256(arch_str.encode()).hexdigest()

    def get_model_id(self) -> str:
        """Get unique model identifier."""
        return f"{self.dataset_name}_cnn_{self.get_architecture_hash()[:8]}"

    def get_layer_names(self) -> list[str]:
        """Get list of available layer names for activation extraction."""
        return ["conv1", "conv2", "hidden", "output"]

    def __del__(self):
        """Cleanup hooks when object is destroyed."""
        if hasattr(self, "_hook_handles"):
            self._remove_hooks()


class ModelFactory:
    """Factory for creating consistent model instances across experiments."""

    @staticmethod
    def create_cnn_model(
        dataset_name: str,
        architecture_config: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> CNNModel:
        """
        Create a CNN model with optional deterministic initialization.

        Args:
            dataset_name: Name of dataset ('mnist' or 'cifar-10')
            architecture_config: Optional custom architecture configuration
            seed: Optional seed for deterministic initialization

        Returns:
            Initialized CNN model
        """
        if seed is not None:
            torch.manual_seed(seed)

        model = CNNModel(dataset_name, architecture_config)

        # Initialize weights deterministically if seed is provided
        if seed is not None:
            model.apply(ModelFactory._init_weights)

        return model

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Initialize weights using Xavier/Glorot initialization."""
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def validate_architecture_consistency(
        models: list[CNNModel],
    ) -> tuple[bool, str | None]:
        """
        Validate that all models have identical architectures.

        Args:
            models: List of models to validate

        Returns:
            Tuple of (is_consistent, error_message)
        """
        if not models:
            return True, None

        reference_hash = models[0].get_architecture_hash()
        reference_config = models[0].architecture_config

        for i, model in enumerate(models[1:], 1):
            if model.get_architecture_hash() != reference_hash:
                return False, (
                    f"Architecture mismatch at model {i}: "
                    f"expected {reference_config}, got {model.architecture_config}"
                )

        return True, None

    @staticmethod
    def get_default_architectures() -> dict[str, dict[str, Any]]:
        """Get default architecture configurations for different datasets."""
        return {
            "mnist": {
                "conv_layers": [
                    {"out_channels": 32, "kernel_size": 3, "padding": 1},
                    {"out_channels": 64, "kernel_size": 3, "padding": 1},
                ],
                "pool_size": 2,
                "hidden_size": 128,
                "dropout_rate": 0.5,
            },
            "cifar-10": {
                "conv_layers": [
                    {"out_channels": 32, "kernel_size": 3, "padding": 1},
                    {"out_channels": 64, "kernel_size": 3, "padding": 1},
                    {"out_channels": 128, "kernel_size": 3, "padding": 1},
                ],
                "pool_size": 2,
                "hidden_size": 256,
                "dropout_rate": 0.5,
            },
        }
