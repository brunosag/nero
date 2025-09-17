"""CLI command implementations."""

import tempfile

from nero.data.datasets import MNISTDataset
from nero.orchestration.experiment_manager import BasicExperimentManager


def run_dummy_experiment(args) -> int:
    """
    Run a dummy experiment for integration testing.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        print("Starting dummy experiment...")
        print(f"Optimizer: {args.optimizer}")
        print(f"Epochs: {args.epochs}")

        # Create experiment configuration
        from nero.domain.models import ExperimentConfig

        # Create a minimal valid config for dummy experiment
        config = ExperimentConfig(
            experiment_id="dummy_experiment",
            optimizer_name=args.optimizer,
            dataset_name="mnist",
            model_architecture="cnn",
            epochs=args.epochs,
            num_runs=30,  # Minimum required for validation
            random_seed_base=42,
            hpo_budget=1,
            hpo_budget_type="trials",
        )

        # Create dataset
        print("Loading MNIST dataset...")
        dataset = MNISTDataset()

        # Create experiment manager
        manager = BasicExperimentManager()

        # Determine output directory
        output_dir = args.output_dir or tempfile.mkdtemp(prefix="nero_dummy_")

        print(f"Output directory: {output_dir}")

        # Run experiment
        print("Running experiment...")
        metrics = manager.run_experiment(
            config=config,
            dataset=dataset,
            run_id=0,
            output_dir=output_dir,
        )

        # Print results
        print("\nExperiment completed successfully!")
        print(f"Final train accuracy: {metrics.train_accuracies[-1]:.4f}")
        print(f"Final test accuracy: {metrics.test_accuracies[-1]:.4f}")
        print(f"Total training time: {metrics.total_training_time:.2f}s")
        print(f"Results saved to: {output_dir}")

        return 0

    except Exception as e:
        print(f"Error running dummy experiment: {e}")
        import traceback

        traceback.print_exc()
        return 1
