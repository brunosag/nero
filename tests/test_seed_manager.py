"""Tests for SeedManager functionality."""

import numpy as np
import pytest
import torch

from nero.orchestration.seed_manager import SeedManager


class TestSeedManager:
    """Test cases for SeedManager."""

    def test_init_valid_base_seed(self):
        """Test SeedManager initialization with valid base seed."""
        seed_manager = SeedManager(42)
        assert seed_manager.base_seed == 42

    def test_init_invalid_base_seed(self):
        """Test SeedManager initialization with invalid base seed."""
        with pytest.raises(ValueError, match="base_seed must be non-negative"):
            SeedManager(-1)

    def test_get_experiment_seeds_valid_input(self):
        """Test getting experiment seeds with valid input."""
        seed_manager = SeedManager(42)
        seeds = seed_manager.get_experiment_seeds("test_exp", 0)

        assert isinstance(seeds, dict)
        assert "model_init" in seeds
        assert "data_shuffle" in seeds
        assert "optimizer" in seeds

        # All seeds should be non-negative integers
        for _seed_type, seed_value in seeds.items():
            assert isinstance(seed_value, int)
            assert seed_value >= 0

    def test_get_experiment_seeds_invalid_experiment_id(self):
        """Test getting experiment seeds with invalid experiment ID."""
        seed_manager = SeedManager(42)

        with pytest.raises(ValueError, match="experiment_id cannot be empty"):
            seed_manager.get_experiment_seeds("", 0)

        with pytest.raises(ValueError, match="experiment_id cannot be empty"):
            seed_manager.get_experiment_seeds("   ", 0)

    def test_get_experiment_seeds_invalid_run_id(self):
        """Test getting experiment seeds with invalid run ID."""
        seed_manager = SeedManager(42)

        with pytest.raises(ValueError, match="run_id must be non-negative"):
            seed_manager.get_experiment_seeds("test_exp", -1)

    def test_seed_determinism(self):
        """Test that seeds are deterministic for same inputs."""
        seed_manager = SeedManager(42)

        seeds1 = seed_manager.get_experiment_seeds("test_exp", 0)
        seeds2 = seed_manager.get_experiment_seeds("test_exp", 0)

        assert seeds1 == seeds2

    def test_seed_uniqueness_across_runs(self):
        """Test that seeds are different across different runs."""
        seed_manager = SeedManager(42)

        seeds_run0 = seed_manager.get_experiment_seeds("test_exp", 0)
        seeds_run1 = seed_manager.get_experiment_seeds("test_exp", 1)

        assert seeds_run0 != seeds_run1

        # Each seed type should be different
        assert seeds_run0["model_init"] != seeds_run1["model_init"]
        assert seeds_run0["data_shuffle"] != seeds_run1["data_shuffle"]
        assert seeds_run0["optimizer"] != seeds_run1["optimizer"]

    def test_seed_uniqueness_across_experiments(self):
        """Test that seeds are different across different experiments."""
        seed_manager = SeedManager(42)

        seeds_exp1 = seed_manager.get_experiment_seeds("exp1", 0)
        seeds_exp2 = seed_manager.get_experiment_seeds("exp2", 0)

        assert seeds_exp1 != seeds_exp2

    def test_seed_uniqueness_within_experiment(self):
        """Test that different seed types within same experiment are unique."""
        seed_manager = SeedManager(42)

        seeds = seed_manager.get_experiment_seeds("test_exp", 0)

        # All seed values should be different
        seed_values = list(seeds.values())
        assert len(seed_values) == len(set(seed_values))

    def test_get_batch_seeds_valid_input(self):
        """Test getting batch seeds with valid input."""
        seed_manager = SeedManager(42)
        batch_seeds = seed_manager.get_batch_seeds("test_exp", 3)

        assert len(batch_seeds) == 3
        assert all(isinstance(seeds, dict) for seeds in batch_seeds)

        # Each run should have different seeds
        for i in range(len(batch_seeds)):
            for j in range(i + 1, len(batch_seeds)):
                assert batch_seeds[i] != batch_seeds[j]

    def test_get_batch_seeds_invalid_num_runs(self):
        """Test getting batch seeds with invalid number of runs."""
        seed_manager = SeedManager(42)

        with pytest.raises(ValueError, match="num_runs must be positive"):
            seed_manager.get_batch_seeds("test_exp", 0)

        with pytest.raises(ValueError, match="num_runs must be positive"):
            seed_manager.get_batch_seeds("test_exp", -1)

    def test_set_model_init_seed(self):
        """Test setting model initialization seed."""
        seed_manager = SeedManager(42)

        # Get expected seed
        expected_seeds = seed_manager.get_experiment_seeds("test_exp", 0)
        expected_model_seed = expected_seeds["model_init"]

        # Set the seed
        seed_manager.set_model_init_seed("test_exp", 0)

        # Verify PyTorch seed was set correctly
        # We can't directly check the seed, but we can verify reproducibility
        torch.manual_seed(expected_model_seed)
        expected_tensor = torch.randn(10)

        seed_manager.set_model_init_seed("test_exp", 0)
        actual_tensor = torch.randn(10)

        assert torch.allclose(expected_tensor, actual_tensor)

    def test_set_data_shuffle_seed(self):
        """Test setting data shuffle seed."""
        seed_manager = SeedManager(42)

        # Get expected seed
        expected_seeds = seed_manager.get_experiment_seeds("test_exp", 0)
        expected_data_seed = expected_seeds["data_shuffle"]

        # Set the seed and verify numpy reproducibility
        np.random.seed(expected_data_seed)
        expected_array = np.random.randn(10)

        seed_manager.set_data_shuffle_seed("test_exp", 0)
        actual_array = np.random.randn(10)

        assert np.allclose(expected_array, actual_array)

    def test_set_optimizer_seed(self):
        """Test setting optimizer seed."""
        seed_manager = SeedManager(42)

        # Get expected seed
        expected_seeds = seed_manager.get_experiment_seeds("test_exp", 0)
        expected_optimizer_seed = expected_seeds["optimizer"]

        # Set the seed and verify reproducibility
        torch.manual_seed(expected_optimizer_seed)
        expected_tensor = torch.randn(10)

        seed_manager.set_optimizer_seed("test_exp", 0)
        actual_tensor = torch.randn(10)

        assert torch.allclose(expected_tensor, actual_tensor)

    def test_verify_seed_consistency_matching(self):
        """Test seed consistency verification with matching seeds."""
        seed_manager = SeedManager(42)

        seeds = seed_manager.get_experiment_seeds("test_exp", 0)
        is_consistent = seed_manager.verify_seed_consistency("test_exp", 0, seeds)

        assert is_consistent is True

    def test_verify_seed_consistency_non_matching(self):
        """Test seed consistency verification with non-matching seeds."""
        seed_manager = SeedManager(42)

        wrong_seeds = {"model_init": 123, "data_shuffle": 456, "optimizer": 789}
        is_consistent = seed_manager.verify_seed_consistency("test_exp", 0, wrong_seeds)

        assert is_consistent is False

    def test_ensure_fair_comparison_seeds_valid_input(self):
        """Test fair comparison seed generation with valid input."""
        seed_manager = SeedManager(42)

        experiment_ids = ["exp1", "exp2", "exp3"]
        fair_seeds = seed_manager.ensure_fair_comparison_seeds(experiment_ids, 0)

        assert len(fair_seeds) == 3
        assert all(exp_id in fair_seeds for exp_id in experiment_ids)

        # Model init seeds should be identical for fair comparison
        model_seeds = [seeds["model_init"] for seeds in fair_seeds.values()]
        assert len(set(model_seeds)) == 1

        # Data and optimizer seeds should be different across experiments
        data_seeds = [seeds["data_shuffle"] for seeds in fair_seeds.values()]
        optimizer_seeds = [seeds["optimizer"] for seeds in fair_seeds.values()]

        assert len(set(data_seeds)) == len(experiment_ids)
        assert len(set(optimizer_seeds)) == len(experiment_ids)

    def test_ensure_fair_comparison_seeds_invalid_input(self):
        """Test fair comparison seed generation with invalid input."""
        seed_manager = SeedManager(42)

        with pytest.raises(ValueError, match="experiment_ids cannot be empty"):
            seed_manager.ensure_fair_comparison_seeds([], 0)

        with pytest.raises(ValueError, match="run_id must be non-negative"):
            seed_manager.ensure_fair_comparison_seeds(["exp1"], -1)

    def test_fair_comparison_consistency_across_runs(self):
        """Test that fair comparison maintains consistency across different runs."""
        seed_manager = SeedManager(42)

        experiment_ids = ["exp1", "exp2"]

        fair_seeds_run0 = seed_manager.ensure_fair_comparison_seeds(experiment_ids, 0)
        fair_seeds_run1 = seed_manager.ensure_fair_comparison_seeds(experiment_ids, 1)

        # Model seeds should be different across runs but same within each run
        run0_model_seeds = [seeds["model_init"] for seeds in fair_seeds_run0.values()]
        run1_model_seeds = [seeds["model_init"] for seeds in fair_seeds_run1.values()]

        # Within each run, model seeds should be identical
        assert len(set(run0_model_seeds)) == 1
        assert len(set(run1_model_seeds)) == 1

        # Across runs, model seeds should be different
        assert run0_model_seeds[0] != run1_model_seeds[0]

    def test_different_base_seeds_produce_different_results(self):
        """Test that different base seeds produce different seed values."""
        seed_manager1 = SeedManager(42)
        seed_manager2 = SeedManager(123)

        seeds1 = seed_manager1.get_experiment_seeds("test_exp", 0)
        seeds2 = seed_manager2.get_experiment_seeds("test_exp", 0)

        assert seeds1 != seeds2

    def test_seed_range_validity(self):
        """Test that generated seeds are within valid range."""
        seed_manager = SeedManager(42)

        # Test with various inputs to ensure seeds are always valid
        test_cases = [
            ("short", 0),
            ("very_long_experiment_name_with_many_characters", 999),
            ("exp_with_numbers_123", 42),
            ("exp-with-dashes", 0),
            ("exp_with_underscores", 100),
        ]

        for exp_id, run_id in test_cases:
            seeds = seed_manager.get_experiment_seeds(exp_id, run_id)

            for _seed_type, seed_value in seeds.items():
                # Seeds should be within 32-bit signed integer range
                assert 0 <= seed_value < 2**31 - 1

    def test_reproducibility_across_instances(self):
        """
        Test that different SeedManager instances with same base seed produce
        same results.
        """
        seed_manager1 = SeedManager(42)
        seed_manager2 = SeedManager(42)

        seeds1 = seed_manager1.get_experiment_seeds("test_exp", 0)
        seeds2 = seed_manager2.get_experiment_seeds("test_exp", 0)

        assert seeds1 == seeds2

    def test_integration_with_pytorch_cuda(self):
        """Test integration with PyTorch CUDA if available."""
        seed_manager = SeedManager(42)

        # This should not raise an error regardless of CUDA availability
        seed_manager.set_model_init_seed("test_exp", 0)

        if torch.cuda.is_available():
            # If CUDA is available, verify that CUDA seeds are set
            # We can't directly test this, but we ensure no errors occur
            seed_manager.set_model_init_seed("test_exp", 1)
            assert True  # If we reach here, no errors occurred
