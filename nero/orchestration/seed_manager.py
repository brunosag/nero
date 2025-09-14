"""Seed management for deterministic experiment execution."""

import hashlib
import random

import numpy as np
import torch


class SeedManager:
    """Manages deterministic seed generation for fair experiment comparison."""

    def __init__(self, base_seed: int):
        """
        Initialize SeedManager with a base seed.

        Args:
            base_seed: Base seed for deterministic generation
        """
        if base_seed < 0:
            raise ValueError("base_seed must be non-negative")

        self.base_seed = base_seed

    def get_experiment_seeds(self, experiment_id: str, run_id: int) -> dict[str, int]:
        """
        Generate deterministic seeds for a specific experiment run.

        Args:
            experiment_id: Unique identifier for the experiment
            run_id: Run number within the experiment

        Returns:
            Dictionary containing seeds for different purposes:
            - model_init: For model weight initialization
            - data_shuffle: For data shuffling/augmentation
            - optimizer: For optimizer-specific randomness
        """
        if not experiment_id or not experiment_id.strip():
            raise ValueError("experiment_id cannot be empty")
        if run_id < 0:
            raise ValueError("run_id must be non-negative")

        # Create deterministic hash-based seeds
        model_init_seed = self._generate_seed(f"{experiment_id}_model_init_{run_id}")
        data_shuffle_seed = self._generate_seed(
            f"{experiment_id}_data_shuffle_{run_id}"
        )
        optimizer_seed = self._generate_seed(f"{experiment_id}_optimizer_{run_id}")

        return {
            "model_init": model_init_seed,
            "data_shuffle": data_shuffle_seed,
            "optimizer": optimizer_seed,
        }

    def get_batch_seeds(
        self, experiment_id: str, num_runs: int
    ) -> list[dict[str, int]]:
        """
        Generate seeds for multiple runs of the same experiment.

        Args:
            experiment_id: Unique identifier for the experiment
            num_runs: Number of runs to generate seeds for

        Returns:
            List of seed dictionaries, one for each run
        """
        if num_runs <= 0:
            raise ValueError("num_runs must be positive")

        return [
            self.get_experiment_seeds(experiment_id, run_id)
            for run_id in range(num_runs)
        ]

    def set_model_init_seed(self, experiment_id: str, run_id: int) -> None:
        """
        Set seeds for model initialization.

        Args:
            experiment_id: Unique identifier for the experiment
            run_id: Run number within the experiment
        """
        seeds = self.get_experiment_seeds(experiment_id, run_id)
        model_seed = seeds["model_init"]

        # Set all relevant random number generators
        torch.manual_seed(model_seed)
        np.random.seed(model_seed)
        random.seed(model_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(model_seed)
            torch.cuda.manual_seed_all(model_seed)

    def set_data_shuffle_seed(self, experiment_id: str, run_id: int) -> None:
        """
        Set seeds for data shuffling and augmentation.

        Args:
            experiment_id: Unique identifier for the experiment
            run_id: Run number within the experiment
        """
        seeds = self.get_experiment_seeds(experiment_id, run_id)
        data_seed = seeds["data_shuffle"]

        # Set numpy seed for data operations
        np.random.seed(data_seed)
        random.seed(data_seed)

    def set_optimizer_seed(self, experiment_id: str, run_id: int) -> None:
        """
        Set seeds for optimizer-specific randomness.

        Args:
            experiment_id: Unique identifier for the experiment
            run_id: Run number within the experiment
        """
        seeds = self.get_experiment_seeds(experiment_id, run_id)
        optimizer_seed = seeds["optimizer"]

        # Set seeds for optimizer randomness
        torch.manual_seed(optimizer_seed)
        np.random.seed(optimizer_seed)
        random.seed(optimizer_seed)

    def verify_seed_consistency(
        self, experiment_id: str, run_id: int, expected_seeds: dict[str, int]
    ) -> bool:
        """
        Verify that generated seeds match expected values for consistency.

        Args:
            experiment_id: Unique identifier for the experiment
            run_id: Run number within the experiment
            expected_seeds: Expected seed values to verify against

        Returns:
            True if seeds match, False otherwise
        """
        generated_seeds = self.get_experiment_seeds(experiment_id, run_id)
        return generated_seeds == expected_seeds

    def ensure_fair_comparison_seeds(
        self, experiment_ids: list[str], run_id: int
    ) -> dict[str, dict[str, int]]:
        """
        Generate seeds for fair comparison between different experiments.

        For fair comparison, model initialization seeds should be identical
        across experiments within the same run, while data and optimizer
        seeds should be experiment-specific.

        Args:
            experiment_ids: List of experiment IDs to compare
            run_id: Run number for the comparison

        Returns:
            Dictionary mapping experiment IDs to their seed dictionaries
        """
        if not experiment_ids:
            raise ValueError("experiment_ids cannot be empty")
        if run_id < 0:
            raise ValueError("run_id must be non-negative")

        # Generate a shared model initialization seed for fair comparison
        shared_model_seed = self._generate_seed(f"fair_comparison_model_{run_id}")

        result = {}
        for exp_id in experiment_ids:
            seeds = self.get_experiment_seeds(exp_id, run_id)
            # Override model_init seed to ensure fair comparison
            seeds["model_init"] = shared_model_seed
            result[exp_id] = seeds

        return result

    def _generate_seed(self, seed_string: str) -> int:
        """
        Generate a deterministic seed from a string using hash function.

        Args:
            seed_string: String to hash for seed generation

        Returns:
            Integer seed value
        """
        # Combine base seed with the string for deterministic generation
        combined_string = f"{self.base_seed}_{seed_string}"

        # Use SHA-256 hash for deterministic seed generation
        hash_object = hashlib.sha256(combined_string.encode())
        hash_hex = hash_object.hexdigest()

        # Convert first 8 characters of hex to integer
        # This gives us a 32-bit integer which is suitable for most RNG seeds
        seed_value = int(hash_hex[:8], 16)

        # Ensure the seed is within valid range for most RNG implementations
        return seed_value % (2**31 - 1)
