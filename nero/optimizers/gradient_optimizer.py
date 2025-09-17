"""Gradient-based optimizer implementation."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nero.domain.interfaces import Dataset, ExperimentLogger, Optimizer
from nero.domain.models import OptimizerConfig, TrainingMetrics


class GradientOptimizer(Optimizer):
    """Minimal gradient-based optimizer implementation."""

    def __init__(self, config: OptimizerConfig):
        super().__init__(config)
        self.optimizer_name = config.name.lower()

        if self.optimizer_name not in ["adam", "sgd"]:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

    def optimize(
        self,
        model: nn.Module,
        dataset: Dataset,
        epochs: int,
        logger: ExperimentLogger,
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
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"Using device: {device}")

        # Optimize batch size using VRAMManager if on GPU
        if device.type == "cuda":
            from nero.orchestration.vram_manager import VRAMError, VRAMManager

            try:
                vram_manager = VRAMManager(safety_margin=0.1, device=device)

                # Get sample input for batch size optimization
                sample_batch, _ = dataset.get_sample_batch(1)
                sample_input = sample_batch[0]  # Single sample

                # Optimize batch size
                batch_size = vram_manager.optimize_batch_size(
                    model=model,
                    sample_input=sample_input,
                    initial_batch_size=64,
                    min_batch_size=16,
                    max_batch_size=512,
                )

                # Log memory stats
                memory_stats = vram_manager.get_memory_stats()
                print(f"VRAM optimized batch size: {batch_size}")
                print(f"GPU memory utilization: {memory_stats['utilization']:.1%}")

            except VRAMError as e:
                print(f"VRAM optimization failed: {e}")
                batch_size = 32  # Fallback
        else:
            batch_size = 32  # CPU default

        # Create PyTorch optimizer
        optimizer = self._create_torch_optimizer(model.parameters())
        criterion = nn.CrossEntropyLoss()

        # Set data shuffle seed for reproducibility
        data_generator = torch.Generator()
        data_generator.manual_seed(42)  # Fixed seed for reproducible shuffling

        # Get data loaders with optimized batch size
        train_loader = dataset.get_train_loader(batch_size=batch_size, shuffle=True)
        test_loader = dataset.get_test_loader(batch_size=batch_size)

        # Log hyperparameters
        logger.log_hyperparameters(
            {
                "optimizer": self.optimizer_name,
                "epochs": epochs,
                "device": str(device),
                "batch_size": batch_size,
                **self.config.params,
            }
        )

        # Training loop
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss, train_acc = self._train_epoch(
                model, train_loader, optimizer, criterion, device
            )

            # Evaluation phase
            model.eval()
            test_loss, test_acc = self._evaluate(model, test_loader, criterion, device)

            # Compute gradient norm
            grad_norm = self._compute_gradient_norm(model)

            # Log epoch metrics
            logger.log_epoch(
                epoch,
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "test_loss": test_loss,
                    "test_accuracy": test_acc,
                    "gradient_norm": grad_norm,
                },
            )

        return logger.get_metrics()

    def _create_torch_optimizer(self, parameters) -> torch.optim.Optimizer:
        """Create PyTorch optimizer from config."""
        params = self.config.params

        if self.optimizer_name == "adam":
            return torch.optim.Adam(
                parameters,
                lr=params.get("learning_rate", 0.001),
                betas=(params.get("beta1", 0.9), params.get("beta2", 0.999)),
                weight_decay=params.get("weight_decay", 0.0),
            )
        elif self.optimizer_name == "sgd":
            return torch.optim.SGD(
                parameters,
                lr=params.get("learning_rate", 0.01),
                momentum=params.get("momentum", 0.0),
                weight_decay=params.get("weight_decay", 0.0),
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
    ) -> tuple[float, float]:
        """Train for one epoch."""
        total_loss = 0.0
        correct = 0
        total = 0

        for _batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def _evaluate(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> tuple[float, float]:
        """Evaluate model on test set."""
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                # Move data to device
                data, target = data.to(device), target.to(device)

                output = model(data)
                loss = criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def _compute_gradient_norm(self, model: nn.Module) -> float:
        """Compute L2 norm of gradients."""
        total_norm = 0.0
        param_count = 0

        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1

        if param_count == 0:
            return 0.0

        return total_norm**0.5

    def get_optimizer_type(self) -> str:
        """Return the type of optimizer."""
        return "gradient"
