"""Integration test for MNIST dataset with actual download."""

import tempfile

import pytest

from nero.data.datasets import MNISTDataset


@pytest.mark.slow
class TestMNISTIntegration:
    """Integration tests that actually download MNIST dataset."""

    def test_mnist_download_and_basic_functionality(self, capsys):
        """Test that MNIST dataset can be downloaded and basic functionality works."""
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"\nTesting MNIST download in: {temp_dir}")

            # This will actually download MNIST
            dataset = MNISTDataset(root=temp_dir, download=True)

            # Capture and display the output
            captured = capsys.readouterr()
            if captured.out:
                print(captured.out)

            # Test basic properties
            assert dataset.name == "mnist"
            assert dataset.num_classes == 10
            assert dataset.input_shape == (1, 28, 28)

            # Test dataset sizes
            assert len(dataset.train_dataset) == 60000
            assert len(dataset.test_dataset) == 10000

            # Test single sample
            sample, target = dataset.get_sample(0)
            assert sample.shape == (1, 28, 28)
            assert 0 <= target <= 9

            # Test sample batch
            samples, targets = dataset.get_sample_batch(5)
            assert samples.shape == (5, 1, 28, 28)
            assert targets.shape == (5,)

            # Test data loaders
            train_loader = dataset.get_train_loader(batch_size=10)
            test_loader = dataset.get_test_loader(batch_size=10)

            train_batch = next(iter(train_loader))
            test_batch = next(iter(test_loader))

            assert len(train_batch) == 2  # samples, targets
            assert len(test_batch) == 2

            # Test integrity validation
            assert dataset.validate_integrity() is True
