"""Basic tests for MNIST dataset implementation without full download."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from nero.data.datasets import MNISTDataset


class TestMNISTDatasetBasic:
    """Basic test suite for MNISTDataset without full dataset download."""

    def test_dataset_properties(self):
        """Test basic dataset properties without downloading."""
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("torchvision.datasets.MNIST") as mock_mnist,
        ):
            # Mock the MNIST dataset
            mock_train = Mock()
            mock_train.__len__ = Mock(return_value=60000)
            mock_test = Mock()
            mock_test.__len__ = Mock(return_value=10000)

            mock_mnist.side_effect = [mock_train, mock_test]

            dataset = MNISTDataset(root=temp_dir, download=False)

            # Test properties
            assert dataset.name == "mnist"
            assert dataset.num_classes == 10
            assert dataset.input_shape == (1, 28, 28)

    def test_checksum_computation(self):
        """Test checksum computation functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test file
            test_file = Path(temp_dir) / "test.txt"
            with open(test_file, "w") as f:
                f.write("test content")

            with patch("torchvision.datasets.MNIST"):
                dataset = MNISTDataset(root=temp_dir, download=False)
                checksum = dataset._compute_file_checksum(test_file)

                # Should return a valid SHA256 hash (64 hex characters)
                assert len(checksum) == 64
                assert all(c in "0123456789abcdef" for c in checksum)

    def test_expected_checksums_defined(self):
        """Test that expected checksums are properly defined."""
        expected_files = {
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz",
        }

        assert set(MNISTDataset.EXPECTED_CHECKSUMS.keys()) == expected_files

        # All checksums should be valid SHA256 hashes
        for checksum in MNISTDataset.EXPECTED_CHECKSUMS.values():
            assert len(checksum) == 64
            assert all(c in "0123456789abcdef" for c in checksum)

    def test_interface_compliance(self):
        """Test that MNISTDataset implements the Dataset interface."""
        from nero.domain.interfaces import Dataset

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("torchvision.datasets.MNIST"),
        ):
            dataset = MNISTDataset(root=temp_dir, download=False)

            # Should be instance of Dataset interface
            assert isinstance(dataset, Dataset)

            # Should have all required methods
            required_methods = [
                "name",
                "num_classes",
                "input_shape",
                "get_train_loader",
                "get_test_loader",
                "get_sample",
                "get_sample_batch",
                "validate_integrity",
            ]

            for method in required_methods:
                assert hasattr(dataset, method)
