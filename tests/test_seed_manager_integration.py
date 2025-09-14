"""Integration tests for SeedManager demonstrating fair comparison scenarios."""

import numpy as np
import torch
import torch.nn as nn

from nero.orchestration.seed_manager import SeedManager


class SimpleModel(nn.Module):
    """Simple model for testing seed consistency."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


class TestSeedManagerIntegration:
    """
    Integration tests demonstrating SeedManager usage in fair comparison scenarios.
    """

    def test_fair_comparison_model_initialization(self):
        """
        Test that models initialized with fair comparison seeds have identical weights.
        """
        seed_manager = SeedManager(42)

        # Get fair comparison seeds for two different optimizers
        experiment_ids = ["adam_experiment", "sgd_experiment"]
        fair_seeds = seed_manager.ensure_fair_comparison_seeds(experiment_ids, run_id=0)

        # Initialize models for each optimizer using their respective seeds
        models = {}
        for exp_id in experiment_ids:
            # Set model initialization seed
            model_seed = fair_seeds[exp_id]["model_init"]
            torch.manual_seed(model_seed)
            np.random.seed(model_seed)

            # Create and initialize model
            model = SimpleModel()
            models[exp_id] = model

        # Verify that models have identical initial weights (fair comparison)
        adam_weights = models["adam_experiment"].linear.weight.data
        sgd_weights = models["sgd_experiment"].linear.weight.data

        assert torch.allclose(adam_weights, sgd_weights), (
            "Models should have identical initial weights for fair comparison"
        )

        # Verify that biases are also identical
        adam_bias = models["adam_experiment"].linear.bias.data
        sgd_bias = models["sgd_experiment"].linear.bias.data

        assert torch.allclose(adam_bias, sgd_bias), (
            "Models should have identical initial biases for fair comparison"
        )

    def test_different_runs_have_different_model_seeds(self):
        """Test that different runs have different model initialization seeds."""
        seed_manager = SeedManager(42)

        experiment_ids = ["adam_experiment", "sgd_experiment"]

        # Get seeds for run 0 and run 1
        fair_seeds_run0 = seed_manager.ensure_fair_comparison_seeds(
            experiment_ids, run_id=0
        )
        fair_seeds_run1 = seed_manager.ensure_fair_comparison_seeds(
            experiment_ids, run_id=1
        )

        # Model seeds should be different across runs
        run0_model_seed = fair_seeds_run0["adam_experiment"]["model_init"]
        run1_model_seed = fair_seeds_run1["adam_experiment"]["model_init"]

        assert run0_model_seed != run1_model_seed, (
            "Different runs should have different model initialization seeds"
        )

        # But within each run, model seeds should be identical across experiments
        assert (
            fair_seeds_run0["adam_experiment"]["model_init"]
            == fair_seeds_run0["sgd_experiment"]["model_init"]
        )
        assert (
            fair_seeds_run1["adam_experiment"]["model_init"]
            == fair_seeds_run1["sgd_experiment"]["model_init"]
        )

    def test_data_and_optimizer_seeds_are_experiment_specific(self):
        """
        Test that data and optimizer seeds are different across experiments
        for diversity.
        """
        seed_manager = SeedManager(42)

        experiment_ids = ["adam_experiment", "sgd_experiment"]
        fair_seeds = seed_manager.ensure_fair_comparison_seeds(experiment_ids, run_id=0)

        adam_seeds = fair_seeds["adam_experiment"]
        sgd_seeds = fair_seeds["sgd_experiment"]

        # Model seeds should be identical (fair comparison)
        assert adam_seeds["model_init"] == sgd_seeds["model_init"]

        # Data and optimizer seeds should be different (experiment-specific)
        assert adam_seeds["data_shuffle"] != sgd_seeds["data_shuffle"]
        assert adam_seeds["optimizer"] != sgd_seeds["optimizer"]

    def test_reproducible_model_initialization_across_sessions(self):
        """Test that model initialization is reproducible across different sessions."""
        # First session
        seed_manager1 = SeedManager(42)
        seed_manager1.set_model_init_seed("test_exp", 0)
        model1 = SimpleModel()
        weights1 = model1.linear.weight.data.clone()

        # Second session (new SeedManager instance)
        seed_manager2 = SeedManager(42)
        seed_manager2.set_model_init_seed("test_exp", 0)
        model2 = SimpleModel()
        weights2 = model2.linear.weight.data.clone()

        # Weights should be identical
        assert torch.allclose(weights1, weights2), (
            "Model initialization should be reproducible across sessions"
        )

    def test_seed_manager_supports_batch_experiment_workflow(self):
        """Test SeedManager in a typical batch experiment workflow."""
        seed_manager = SeedManager(42)

        # Simulate a batch experiment with 3 runs comparing Adam vs SGD
        experiment_ids = ["adam_exp", "sgd_exp"]
        num_runs = 3

        # Collect all initialized models for verification
        all_models = {}

        for run_id in range(num_runs):
            fair_seeds = seed_manager.ensure_fair_comparison_seeds(
                experiment_ids, run_id
            )
            run_models = {}

            for exp_id in experiment_ids:
                # Set model initialization seed
                model_seed = fair_seeds[exp_id]["model_init"]
                torch.manual_seed(model_seed)

                # Initialize model
                model = SimpleModel()
                run_models[exp_id] = model

            all_models[run_id] = run_models

        # Verify fair comparison within each run
        for run_id in range(num_runs):
            adam_weights = all_models[run_id]["adam_exp"].linear.weight.data
            sgd_weights = all_models[run_id]["sgd_exp"].linear.weight.data

            assert torch.allclose(adam_weights, sgd_weights), (
                f"Run {run_id}: Models should have identical weights "
                f"for fair comparison"
            )

        # Verify diversity across runs
        for run_id1 in range(num_runs):
            for run_id2 in range(run_id1 + 1, num_runs):
                weights1 = all_models[run_id1]["adam_exp"].linear.weight.data
                weights2 = all_models[run_id2]["adam_exp"].linear.weight.data

                assert not torch.allclose(weights1, weights2), (
                    f"Runs {run_id1} and {run_id2} should have different "
                    f"model initializations"
                )

    def test_seed_manager_handles_large_batch_sizes(self):
        """Test SeedManager performance with large batch experiments."""
        seed_manager = SeedManager(42)

        # Test with a larger number of runs (typical for statistical validity)
        num_runs = 30
        experiment_ids = ["adam", "sgd", "leea", "shade-ils"]

        # Generate all seeds
        all_seeds = []
        for run_id in range(num_runs):
            fair_seeds = seed_manager.ensure_fair_comparison_seeds(
                experiment_ids, run_id
            )
            all_seeds.append(fair_seeds)

        # Verify that we have the expected number of seed sets
        assert len(all_seeds) == num_runs

        # Verify that each run has seeds for all experiments
        for run_seeds in all_seeds:
            assert len(run_seeds) == len(experiment_ids)
            for exp_id in experiment_ids:
                assert exp_id in run_seeds
                assert "model_init" in run_seeds[exp_id]
                assert "data_shuffle" in run_seeds[exp_id]
                assert "optimizer" in run_seeds[exp_id]

        # Verify fair comparison property (model seeds identical within runs)
        for run_seeds in all_seeds:
            model_seeds = [seeds["model_init"] for seeds in run_seeds.values()]
            assert len(set(model_seeds)) == 1, (
                "All experiments in a run should have same model seed"
            )

        # Verify diversity across runs
        model_seeds_per_run = [
            list(run_seeds.values())[0]["model_init"] for run_seeds in all_seeds
        ]
        assert len(set(model_seeds_per_run)) == num_runs, (
            "Each run should have a unique model initialization seed"
        )
