"""VRAM management for dynamic batch size optimization."""

import logging

import torch
import torch.nn as nn


class VRAMError(Exception):
    """Exception raised when VRAM operations fail."""

    pass


class VRAMManager:
    """Manages GPU memory and optimizes batch sizes dynamically."""

    def __init__(self, safety_margin: float = 0.1, device: torch.device | None = None):
        """
        Initialize VRAM manager.

        Args:
            safety_margin: Fraction of memory to keep as safety buffer (0.0-1.0)
            device: CUDA device to manage (auto-detected if None)

        Raises:
            VRAMError: If CUDA is not available or device is invalid
        """
        if not torch.cuda.is_available():
            raise VRAMError("CUDA is not available")

        if device is None:
            device = torch.device("cuda:0")
        elif device.type != "cuda":
            raise VRAMError(f"Device must be CUDA device, got {device}")

        self.device = device
        self.safety_margin = max(0.0, min(1.0, safety_margin))

        try:
            self.total_memory = torch.cuda.get_device_properties(device).total_memory
        except Exception as e:
            raise VRAMError(f"Failed to get device properties: {e}") from e

        self.available_memory = self.total_memory * (1 - self.safety_margin)
        self.logger = logging.getLogger(__name__)

        self.logger.info(
            f"VRAMManager initialized for {device} with "
            f"{self.total_memory / 1024**3:.2f}GB total, "
            f"{self.available_memory / 1024**3:.2f}GB available "
            f"(safety margin: {self.safety_margin:.1%})"
        )

    def optimize_batch_size(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        initial_batch_size: int = 32,
        min_batch_size: int = 1,
        max_batch_size: int | None = None,
    ) -> int:
        """
        Find optimal batch size using binary search.

        Args:
            model: Neural network model to test
            sample_input: Sample input tensor (single sample, no batch dimension)
            initial_batch_size: Starting batch size for search
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum batch size to test (2x initial if None)

        Returns:
            Optimal batch size that fits in available VRAM

        Raises:
            VRAMError: If even minimum batch size doesn't fit
        """
        if max_batch_size is None:
            max_batch_size = initial_batch_size * 2

        # Ensure model is on the correct device
        model = model.to(self.device)
        sample_input = sample_input.to(self.device)

        # Test minimum batch size first
        if not self._test_batch_size(model, sample_input, min_batch_size):
            current_memory = self._get_current_memory_usage()
            raise VRAMError(
                f"Insufficient VRAM: minimum batch size {min_batch_size} requires "
                f"more than available {self.available_memory / 1024**3:.2f}GB "
                f"(current usage: {current_memory / 1024**3:.2f}GB)"
            )

        # Binary search for optimal batch size
        low, high = min_batch_size, max_batch_size
        optimal_batch_size = min_batch_size

        self.logger.debug(f"Starting binary search: low={low}, high={high}")

        while low <= high:
            mid = (low + high) // 2
            self.logger.debug(f"Testing batch size: {mid}")

            if self._test_batch_size(model, sample_input, mid):
                optimal_batch_size = mid
                low = mid + 1
                self.logger.debug(f"Batch size {mid} fits, trying larger")
            else:
                high = mid - 1
                self.logger.debug(f"Batch size {mid} too large, trying smaller")

        current_memory = self._get_current_memory_usage()
        self.logger.info(
            f"Optimal batch size: {optimal_batch_size} "
            f"(memory usage: {current_memory / 1024**3:.2f}GB / "
            f"{self.available_memory / 1024**3:.2f}GB available)"
        )

        return optimal_batch_size

    def _test_batch_size(
        self, model: nn.Module, sample_input: torch.Tensor, batch_size: int
    ) -> bool:
        """
        Test if a batch size fits in available VRAM.

        Args:
            model: Model to test
            sample_input: Sample input tensor
            batch_size: Batch size to test

        Returns:
            True if batch size fits, False otherwise
        """
        try:
            # Clear cache before testing
            torch.cuda.empty_cache()

            # Create batch of inputs
            batch_input = sample_input.unsqueeze(0).repeat(
                batch_size, *([1] * len(sample_input.shape))
            )

            # Test forward pass
            model.eval()
            with torch.no_grad():
                _ = model(batch_input)

            # Test backward pass (more memory intensive)
            model.train()
            batch_input.requires_grad_(True)
            output = model(batch_input)

            # Create dummy loss for backward pass
            if output.dim() > 1:
                dummy_target = torch.randint(
                    0, output.size(-1), (batch_size,), device=self.device
                )
                loss = torch.nn.functional.cross_entropy(output, dummy_target)
            else:
                dummy_target = torch.randn_like(output)
                loss = torch.nn.functional.mse_loss(output, dummy_target)

            loss.backward()

            # Check if we're within memory limits
            current_memory = self._get_current_memory_usage()
            fits = current_memory <= self.available_memory

            # Clean up
            del batch_input, output, loss, dummy_target
            torch.cuda.empty_cache()

            return fits

        except torch.cuda.OutOfMemoryError:
            # Clean up after OOM
            torch.cuda.empty_cache()
            return False
        except Exception as e:
            self.logger.warning(f"Error testing batch size {batch_size}: {e}")
            torch.cuda.empty_cache()
            return False

    def _get_current_memory_usage(self) -> int:
        """Get current GPU memory usage in bytes."""
        return torch.cuda.memory_allocated(self.device)

    def get_memory_stats(self) -> dict[str, float]:
        """
        Get current memory statistics.

        Returns:
            Dictionary with memory statistics in GB
        """
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)

        return {
            "total_gb": self.total_memory / 1024**3,
            "available_gb": self.available_memory / 1024**3,
            "allocated_gb": allocated / 1024**3,
            "reserved_gb": reserved / 1024**3,
            "free_gb": (self.total_memory - reserved) / 1024**3,
            "utilization": allocated / self.total_memory,
            "safety_margin": self.safety_margin,
        }

    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        torch.cuda.empty_cache()
        self.logger.debug("GPU memory cache cleared")

    def check_memory_available(self, required_bytes: int) -> bool:
        """
        Check if required memory is available.

        Args:
            required_bytes: Required memory in bytes

        Returns:
            True if memory is available, False otherwise
        """
        current_usage = self._get_current_memory_usage()
        return (current_usage + required_bytes) <= self.available_memory

    def get_recommended_batch_size(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        target_memory_usage: float = 0.8,
    ) -> int:
        """
        Get recommended batch size for target memory usage.

        Args:
            model: Model to analyze
            sample_input: Sample input tensor
            target_memory_usage: Target memory utilization (0.0-1.0)

        Returns:
            Recommended batch size
        """
        target_memory = self.available_memory * target_memory_usage

        try:
            # Start with batch size 1 and estimate memory per sample
            model = model.to(self.device)
            sample_input = sample_input.to(self.device)
            torch.cuda.empty_cache()

            # Measure memory for single sample
            initial_memory = self._get_current_memory_usage()

            model.eval()
            with torch.no_grad():
                single_input = sample_input.unsqueeze(0)
                _ = model(single_input)

            memory_per_sample = self._get_current_memory_usage() - initial_memory

            if memory_per_sample <= 0:
                return 32  # Default fallback

            estimated_batch_size = int(target_memory / memory_per_sample)

            # Clean up
            del single_input
            torch.cuda.empty_cache()

            return max(1, estimated_batch_size)

        except Exception as e:
            self.logger.warning(f"Error estimating batch size: {e}")
            torch.cuda.empty_cache()
            return 32  # Default fallback
