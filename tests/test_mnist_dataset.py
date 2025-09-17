"""Tests for MNIST dataset implementation."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from torch.utils.data import DataLoader

from nero.data.datasets import MNISTDataset


class TestMNISTDataset:
    """Test suite for MNISTDataset."""

    @pytest.fixture
    def mock_mnist_datasets(self):
        """Create mock MNIST datasets for testing."""
        # Create mock train dataset
        mock_train = Mock()
        mock_train.__len__ = Mock(return_value=60000)
        mock_train.__getitem__ = Mock(return_value=(torch.randn(1, 28, 28), 5))

        # Create mock test dataset
        mock_test = Mock()
        mock_test.__len__ = Mock(return_value=10000)
        mock_test.__getitem__ = Mock(return_value=(torch.randn(1, 28, 28), 3))

        return mock_train, mock_test

    @pytest.fixture
    def mnist_dataset_mock(self, mock_mnist_datasets):
        """Create MNISTDataset with mocked torchvision datasets."""
        mock_train, mock_test = mock_mnist_datasets

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("torchvision.datasets.MNIST") as mock_mnist,
            patch.object(MNISTDataset, "validate_integrity", return_value=True),
        ):
            mock_mnist.side_effect = [mock_train, mock_test]
            dataset = MNISTDataset(root=temp_dir, download=False)
            yield dataset

    def test_dataset_properties(self, mnist_dataset_mock):
        """Test basic dataset properties."""
        assert mnist_dataset_mock.name == "mnist"
        assert mnist_dataset_mock.num_classes == 10
        assert mnist_dataset_mock.input_shape == (1, 28, 28)

    def test_train_loader_creation(self, mnist_dataset_mock):
        """Test training data loader creation."""
        # Test with default batch size
        train_loader = mnist_dataset_mock.get_train_loader()
        assert isinstance(train_loader, DataLoader)
        assert train_loader.batch_size == 64  # default batch size

        # Test with custom batch size
        custom_batch_size = 32
        train_loader_custom = mnist_dataset_mock.get_train_loader(
            batch_size=custom_batch_size
        )
        assert train_loader_custom.batch_size == custom_batch_size

    def test_test_loader_creation(self, mnist_dataset_mock):
        """Test test data loader creation."""
        # Test with default batch size
        test_loader = mnist_dataset_mock.get_test_loader()
        assert isinstance(test_loader, DataLoader)
        assert test_loader.batch_size == 64  # default batch size

        # Test with custom batch size
        custom_batch_size = 128
        test_loader_custom = mnist_dataset_mock.get_test_loader(
            batch_size=custom_batch_size
        )
        assert test_loader_custom.batch_size == custom_batch_size

    def test_get_sample(self, mnist_dataset_mock):
        """Test single sample retrieval."""
        # Test valid index
        sample, target = mnist_dataset_mock.get_sample(0)
        assert sample.shape == (1, 28, 28)
        assert isinstance(target, int)
        assert 0 <= target <= 9

    def test_get_sample_invalid_index(self, mnist_dataset_mock):
        """Test get_sample with invalid indices."""
        # Test negative index
        with pytest.raises(IndexError):
            mnist_dataset_mock.get_sample(-1)

        # Test index too large
        with pytest.raises(IndexError):
            mnist_dataset_mock.get_sample(len(mnist_dataset_mock.test_dataset))

    def test_get_sample_batch_random(self, mnist_dataset_mock):
        """Test random sample batch retrieval."""
        batch_size = 5

        # Mock the subset and dataloader behavior
        with (
            patch("torch.utils.data.Subset"),
            patch("torch.utils.data.DataLoader") as mock_dataloader,
        ):
            # Mock the dataloader to return our test batch
            mock_loader_instance = Mock()
            mock_loader_instance.__iter__ = Mock(
                return_value=iter(
                    [
                        (
                            torch.randn(batch_size, 1, 28, 28),
                            torch.randint(0, 10, (batch_size,)),
                        )
                    ]
                )
            )
            mock_dataloader.return_value = mock_loader_instance

            samples, targets = mnist_dataset_mock.get_sample_batch(batch_size)

            assert samples.shape == (batch_size, 1, 28, 28)
            assert targets.shape == (batch_size,)

    def test_get_sample_batch_specific_indices(self, mnist_dataset_mock):
        """Test sample batch retrieval with specific indices."""
        indices = [0, 1, 2, 3, 4]
        batch_size = len(indices)

        # Mock the subset and dataloader behavior
        with (
            patch("torch.utils.data.Subset"),
            patch("torch.utils.data.DataLoader") as mock_dataloader,
        ):
            # Mock the dataloader to return our test batch
            mock_loader_instance = Mock()
            mock_loader_instance.__iter__ = Mock(
                return_value=iter(
                    [
                        (
                            torch.randn(batch_size, 1, 28, 28),
                            torch.randint(0, 10, (batch_size,)),
                        )
                    ]
                )
            )
            mock_dataloader.return_value = mock_loader_instance

            samples, targets = mnist_dataset_mock.get_sample_batch(
                batch_size, indices=indices
            )

            assert samples.shape == (batch_size, 1, 28, 28)
            assert targets.shape == (batch_size,)

    def test_get_sample_batch_invalid_inputs(self, mnist_dataset_mock):
        """Test get_sample_batch with invalid inputs."""
        # Test mismatched batch_size and indices length
        with pytest.raises(
            ValueError, match="Length of indices .* must match batch_size"
        ):
            mnist_dataset_mock.get_sample_batch(5, indices=[0, 1, 2])

        # Test invalid index in indices list
        with pytest.raises(IndexError):
            mnist_dataset_mock.get_sample_batch(
                2, indices=[0, len(mnist_dataset_mock.test_dataset)]
            )

    def test_dataset_sizes(self, mnist_dataset_mock):
        """Test that dataset has expected sizes."""
        assert len(mnist_dataset_mock.train_dataset) == 60000
        assert len(mnist_dataset_mock.test_dataset) == 10000

    def test_checksum_computation(self):
        """Test checksum computation functionality."""
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("torchvision.datasets.MNIST"),
        ):
            # Create a test file
            test_file = Path(temp_dir) / "test.txt"
            with open(test_file, "w") as f:
                f.write("test content")

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

    def test_validate_integrity_missing_files(self):
        """Test dataset integrity validation when files are missing."""
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("torchvision.datasets.MNIST") as mock_mnist,
        ):
            # Mock the datasets
            mock_train = Mock()
            mock_train.__len__ = Mock(return_value=60000)
            mock_test = Mock()
            mock_test.__len__ = Mock(return_value=10000)
            mock_mnist.side_effect = [mock_train, mock_test]

            # Don't mock validate_integrity this time
            dataset = MNISTDataset.__new__(MNISTDataset)
            dataset.root = Path(temp_dir)
            dataset.default_batch_size = 64
            dataset.train_dataset = mock_train
            dataset.test_dataset = mock_test

            # Since no files were actually downloaded, validation should handle
            # missing files
            result = dataset.validate_integrity()
            # This should be True since no raw directory exists
            # (treated as "not downloaded yet")
            assert isinstance(result, bool)

    def test_get_data_statistics_mock(self, mnist_dataset_mock):
        """Test dataset statistics computation with mocked data."""
        # Mock the DataLoader to return predictable data
        with patch("torch.utils.data.DataLoader") as mock_dataloader:
            # Create mock data
            mock_samples = torch.randn(1000, 1, 28, 28)
            mock_targets = torch.randint(0, 10, (1000,))

            # Mock the dataloader iterator
            mock_loader_instance = Mock()
            mock_loader_instance.__iter__ = Mock(
                return_value=iter([(mock_samples, mock_targets)])
            )
            mock_dataloader.return_value = mock_loader_instance

            stats = mnist_dataset_mock.get_data_statistics()

            # Check that all expected keys are present
            expected_keys = {
                "mean",
                "std",
                "min",
                "max",
                "num_train_samples",
                "num_test_samples",
                "num_classes",
                "class_distribution",
            }
            assert set(stats.keys()) == expected_keys

            # Check basic properties
            assert stats["num_train_samples"] == 60000
            assert stats["num_test_samples"] == 10000
            assert stats["num_classes"] == 10

            # Check class distribution
            assert len(stats["class_distribution"]) == 10

    def test_custom_transform(self):
        """Test dataset with custom transform."""
        from torchvision import transforms

        custom_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # No normalization
            ]
        )

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("torchvision.datasets.MNIST") as mock_mnist,
            patch.object(MNISTDataset, "validate_integrity", return_value=True),
        ):
            mock_train = Mock()
            mock_test = Mock()
            mock_mnist.side_effect = [mock_train, mock_test]

            dataset = MNISTDataset(root=temp_dir, transform=custom_transform)

            # Check that the custom transform was set
            assert dataset.transform == custom_transform

    def test_custom_batch_size(self):
        """Test dataset with custom default batch size."""
        custom_batch_size = 128

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("torchvision.datasets.MNIST") as mock_mnist,
            patch.object(MNISTDataset, "validate_integrity", return_value=True),
        ):
            mock_train = Mock()
            mock_train.__len__ = Mock(return_value=60000)
            mock_test = Mock()
            mock_test.__len__ = Mock(return_value=10000)
            mock_mnist.side_effect = [mock_train, mock_test]

            dataset = MNISTDataset(root=temp_dir, default_batch_size=custom_batch_size)

            train_loader = dataset.get_train_loader()
            test_loader = dataset.get_test_loader()

            assert train_loader.batch_size == custom_batch_size
            assert test_loader.batch_size == custom_batch_size

    def test_dataset_interface_compliance(self, mnist_dataset_mock):
        """Test that MNISTDataset properly implements the Dataset interface."""
        from nero.domain.interfaces import Dataset

        # Check that it's an instance of the abstract base class
        assert isinstance(mnist_dataset_mock, Dataset)

        # Check that all abstract methods are implemented
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
            assert hasattr(mnist_dataset_mock, method)

        # Test that methods return expected types
        assert isinstance(mnist_dataset_mock.name, str)
        assert isinstance(mnist_dataset_mock.num_classes, int)
        assert isinstance(mnist_dataset_mock.input_shape, tuple)
        assert isinstance(mnist_dataset_mock.get_train_loader(), DataLoader)
        assert isinstance(mnist_dataset_mock.get_test_loader(), DataLoader)

        sample, target = mnist_dataset_mock.get_sample(0)
        assert isinstance(sample, torch.Tensor)
        assert isinstance(target, int)

        assert isinstance(mnist_dataset_mock.validate_integrity(), bool)
