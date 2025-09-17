"""Tests for CIFAR-10 dataset implementation."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from nero.data.datasets import (
    AdversariallyFilteredDataset,
    CIFAR10Dataset,
    TextureShapeBiasDataset,
)


class TestCIFAR10Dataset:
    """Test cases for CIFAR10Dataset."""

    def test_cifar10_dataset_initialization(self):
        """Test CIFAR-10 dataset initialization."""
        # Mock torchvision datasets to avoid actual download
        with patch("nero.data.datasets.datasets.CIFAR10") as mock_cifar10:
            # Mock dataset with expected properties
            mock_train = MagicMock()
            mock_train.__len__ = MagicMock(return_value=50000)
            mock_test = MagicMock()
            mock_test.__len__ = MagicMock(return_value=10000)
            mock_test.__getitem__ = MagicMock(return_value=(torch.randn(3, 32, 32), 5))

            mock_cifar10.side_effect = [mock_train, mock_test]

            dataset = CIFAR10Dataset(download=False)

            assert dataset.name == "cifar10"
            assert dataset.num_classes == 10
            assert dataset.input_shape == (3, 32, 32)

    def test_cifar10_data_loaders(self):
        """Test CIFAR-10 data loader creation."""
        with patch("nero.data.datasets.datasets.CIFAR10") as mock_cifar10:
            mock_train = MagicMock()
            mock_train.__len__ = MagicMock(return_value=50000)
            mock_test = MagicMock()
            mock_test.__len__ = MagicMock(return_value=10000)
            mock_test.__getitem__ = MagicMock(return_value=(torch.randn(3, 32, 32), 5))
            mock_cifar10.side_effect = [mock_train, mock_test]

            dataset = CIFAR10Dataset(download=False, default_batch_size=32)

            # Test train loader
            train_loader = dataset.get_train_loader()
            assert train_loader.batch_size == 32
            assert train_loader.dataset == mock_train

            # Test test loader
            test_loader = dataset.get_test_loader(batch_size=64)
            assert test_loader.batch_size == 64
            assert test_loader.dataset == mock_test

    def test_cifar10_sample_access(self):
        """Test individual sample access."""
        with patch("nero.data.datasets.datasets.CIFAR10") as mock_cifar10:
            mock_train = MagicMock()
            mock_train.__len__ = MagicMock(return_value=50000)
            mock_test = MagicMock()
            mock_test.__len__ = MagicMock(return_value=10000)
            mock_test.__getitem__ = MagicMock(return_value=(torch.randn(3, 32, 32), 7))
            mock_cifar10.side_effect = [mock_train, mock_test]

            dataset = CIFAR10Dataset(download=False)

            # Test get_sample
            sample, target = dataset.get_sample(100)
            assert sample.shape == (3, 32, 32)
            assert target == 7

            # Test index out of range
            with pytest.raises(IndexError):
                dataset.get_sample(10000)

    def test_cifar10_sample_batch(self):
        """Test batch sample access."""
        with patch("nero.data.datasets.datasets.CIFAR10") as mock_cifar10:
            mock_train = MagicMock()
            mock_train.__len__ = MagicMock(return_value=50000)
            mock_test = MagicMock()
            mock_test.__len__ = MagicMock(return_value=10000)

            mock_test.__getitem__ = MagicMock(
                side_effect=lambda idx: (torch.randn(3, 32, 32), idx % 10)
            )

            mock_cifar10.side_effect = [mock_train, mock_test]

            dataset = CIFAR10Dataset(download=False)

            # Test random batch
            samples, targets = dataset.get_sample_batch(5)
            assert samples.shape == (5, 3, 32, 32)
            assert targets.shape == (5,)

            # Test specific indices
            indices = [0, 1, 2, 3, 4]
            samples, targets = dataset.get_sample_batch(5, indices)
            assert samples.shape == (5, 3, 32, 32)
            assert targets.shape == (5,)

    def test_cifar10_validation(self):
        """Test dataset validation."""
        with patch("nero.data.datasets.datasets.CIFAR10") as mock_cifar10:
            mock_train = MagicMock()
            mock_train.__len__ = MagicMock(return_value=50000)
            mock_test = MagicMock()
            mock_test.__len__ = MagicMock(return_value=10000)
            mock_test.__getitem__ = MagicMock(return_value=(torch.randn(3, 32, 32), 5))

            mock_cifar10.side_effect = [mock_train, mock_test]

            dataset = CIFAR10Dataset(download=False)
            assert dataset.validate_integrity() is True

    def test_cifar10_data_statistics(self):
        """Test data statistics computation."""
        with patch("nero.data.datasets.datasets.CIFAR10") as mock_cifar10:
            # Create mock data with known statistics
            mock_train = MagicMock()
            mock_train.__len__ = MagicMock(return_value=50000)
            mock_test = MagicMock()
            mock_test.__len__ = MagicMock(return_value=10000)
            mock_test.__getitem__ = MagicMock(return_value=(torch.randn(3, 32, 32), 5))

            # Mock DataLoader to return predictable data
            with patch("nero.data.datasets.DataLoader") as mock_dataloader:
                mock_loader = MagicMock()
                # Create mock batches with known statistics
                mock_batch1 = (torch.ones(1000, 3, 32, 32) * 0.5, torch.zeros(1000))
                mock_batch2 = (torch.ones(1000, 3, 32, 32) * -0.5, torch.ones(1000))
                mock_loader.__iter__ = MagicMock(
                    return_value=iter([mock_batch1, mock_batch2])
                )
                mock_dataloader.return_value = mock_loader

                mock_cifar10.side_effect = [mock_train, mock_test]

                dataset = CIFAR10Dataset(download=False)
                stats = dataset.get_data_statistics()

                assert "mean" in stats
                assert "std" in stats
                assert "num_train_samples" in stats
                assert "num_test_samples" in stats
                assert "class_distribution" in stats
                assert isinstance(stats["mean"], list)  # RGB channels
                assert len(stats["mean"]) == 3
                assert isinstance(stats["std"], list)  # RGB channels
                assert len(stats["std"]) == 3


class TestTextureShapeBiasDataset:
    """Test cases for TextureShapeBiasDataset."""

    def test_texture_shape_bias_initialization(self):
        """Test texture-shape bias dataset initialization."""
        # Create mock base dataset
        with patch("nero.data.datasets.datasets.CIFAR10") as mock_cifar10:
            mock_train = MagicMock()
            mock_train.__len__ = MagicMock(return_value=50000)
            mock_test = MagicMock()
            mock_test.__len__ = MagicMock(return_value=100)
            mock_test.__getitem__ = MagicMock(
                side_effect=lambda idx: (torch.randn(3, 32, 32), idx % 10)
            )

            mock_cifar10.side_effect = [mock_train, mock_test]

            base_dataset = CIFAR10Dataset(download=False)
            bias_dataset = TextureShapeBiasDataset(
                base_dataset, conflict_ratio=0.3, seed=42
            )

            assert bias_dataset.name == "cifar10_texture_shape_bias"
            assert bias_dataset.num_classes == 10
            assert bias_dataset.input_shape == (3, 32, 32)
            assert bias_dataset.conflict_ratio == 0.3

    def test_texture_shape_conflict_creation(self):
        """Test creation of texture-shape conflicts."""
        with patch("nero.data.datasets.datasets.CIFAR10") as mock_cifar10:
            mock_train = MagicMock()
            mock_train.__len__ = MagicMock(return_value=50000)
            mock_test = MagicMock()
            mock_test.__len__ = MagicMock(return_value=100)
            mock_test.__getitem__ = MagicMock(
                side_effect=lambda idx: (torch.randn(3, 32, 32), idx % 10)
            )

            mock_cifar10.side_effect = [mock_train, mock_test]

            base_dataset = CIFAR10Dataset(download=False)
            bias_dataset = TextureShapeBiasDataset(
                base_dataset, conflict_ratio=0.3, seed=42
            )

            # Check that conflicts were created
            assert len(bias_dataset.conflicted_data) == 100
            assert len(bias_dataset.conflicted_targets) == 100
            assert len(bias_dataset.conflict_labels) == 100

            # Check conflict ratio
            num_conflicted = sum(bias_dataset.conflict_labels)
            expected_conflicted = int(100 * 0.3)
            assert abs(num_conflicted - expected_conflicted) <= 1

    def test_texture_shape_bias_score(self):
        """Test texture vs shape bias score computation."""
        with patch("nero.data.datasets.datasets.CIFAR10") as mock_cifar10:
            mock_train = MagicMock()
            mock_train.__len__ = MagicMock(return_value=50000)
            mock_test = MagicMock()
            mock_test.__len__ = MagicMock(return_value=10)
            mock_test.__getitem__ = MagicMock(
                side_effect=lambda idx: (torch.randn(3, 32, 32), idx % 10)
            )

            mock_cifar10.side_effect = [mock_train, mock_test]

            base_dataset = CIFAR10Dataset(download=False)
            bias_dataset = TextureShapeBiasDataset(
                base_dataset, conflict_ratio=0.5, seed=42
            )

            # Create mock predictions
            predictions = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

            bias_score = bias_dataset.get_texture_shape_bias_score(predictions)

            assert "texture_bias" in bias_score
            assert "shape_bias" in bias_score
            assert "total_conflicted" in bias_score
            assert "texture_preference" in bias_score
            assert 0 <= bias_score["texture_bias"] <= 1
            assert 0 <= bias_score["shape_bias"] <= 1

    def test_texture_shape_validation(self):
        """Test texture-shape bias dataset validation."""
        with patch("nero.data.datasets.datasets.CIFAR10") as mock_cifar10:
            mock_train = MagicMock()
            mock_train.__len__ = MagicMock(return_value=50000)
            mock_test = MagicMock()
            mock_test.__len__ = MagicMock(return_value=100)
            mock_test.__getitem__ = MagicMock(
                side_effect=lambda idx: (torch.randn(3, 32, 32), idx % 10)
            )

            mock_cifar10.side_effect = [mock_train, mock_test]

            base_dataset = CIFAR10Dataset(download=False)
            bias_dataset = TextureShapeBiasDataset(
                base_dataset, conflict_ratio=0.3, seed=42
            )

            assert bias_dataset.validate_integrity() is True


class TestAdversariallyFilteredDataset:
    """Test cases for AdversariallyFilteredDataset."""

    def test_adversarially_filtered_initialization(self):
        """Test adversarially filtered dataset initialization."""
        with patch("nero.data.datasets.datasets.CIFAR10") as mock_cifar10:
            mock_train = MagicMock()
            mock_train.__len__ = MagicMock(return_value=50000)
            mock_test = MagicMock()
            mock_test.__len__ = MagicMock(return_value=100)
            mock_test.__getitem__ = MagicMock(
                side_effect=lambda idx: (torch.randn(3, 32, 32), idx % 10)
            )

            mock_cifar10.side_effect = [mock_train, mock_test]

            base_dataset = CIFAR10Dataset(download=False)
            filtered_dataset = AdversariallyFilteredDataset(
                base_dataset, filter_ratio=0.2, seed=42
            )

            assert filtered_dataset.name == "cifar10_adversarially_filtered"
            assert filtered_dataset.num_classes == 10
            assert filtered_dataset.input_shape == (3, 32, 32)
            assert filtered_dataset.filter_ratio == 0.2

    def test_spurious_sample_filtering(self):
        """Test filtering of spurious correlation samples."""
        with patch("nero.data.datasets.datasets.CIFAR10") as mock_cifar10:
            mock_train = MagicMock()
            mock_train.__len__ = MagicMock(return_value=50000)
            mock_test = MagicMock()
            mock_test.__len__ = MagicMock(return_value=100)
            mock_test.__getitem__ = MagicMock(
                side_effect=lambda idx: (torch.randn(3, 32, 32), idx % 10)
            )

            mock_cifar10.side_effect = [mock_train, mock_test]

            base_dataset = CIFAR10Dataset(download=False)
            filtered_dataset = AdversariallyFilteredDataset(
                base_dataset, filter_ratio=0.2, seed=42
            )

            # Check that samples were filtered
            original_size = 100
            expected_removed = int(original_size * 0.2)
            expected_remaining = original_size - expected_removed

            assert len(filtered_dataset.filtered_data) == expected_remaining
            assert len(filtered_dataset.filtered_targets) == expected_remaining
            assert len(filtered_dataset.removed_indices) == expected_removed

    def test_robustness_score_computation(self):
        """Test robustness score computation."""
        with patch("nero.data.datasets.datasets.CIFAR10") as mock_cifar10:
            mock_train = MagicMock()
            mock_train.__len__ = MagicMock(return_value=50000)
            mock_test = MagicMock()
            mock_test.__len__ = MagicMock(return_value=100)
            mock_test.__getitem__ = MagicMock(
                side_effect=lambda idx: (torch.randn(3, 32, 32), idx % 10)
            )

            mock_cifar10.side_effect = [mock_train, mock_test]

            base_dataset = CIFAR10Dataset(download=False)
            filtered_dataset = AdversariallyFilteredDataset(
                base_dataset, filter_ratio=0.2, seed=42
            )

            # Create mock predictions (all correct)
            predictions = torch.tensor(filtered_dataset.filtered_targets)

            robustness_score = filtered_dataset.get_robustness_score(predictions)

            assert "filtered_accuracy" in robustness_score
            assert "total_samples" in robustness_score
            assert "removed_samples" in robustness_score
            assert "removal_ratio" in robustness_score
            assert "spurious_features" in robustness_score

            assert (
                robustness_score["filtered_accuracy"] == 1.0
            )  # All predictions correct
            assert robustness_score["total_samples"] == 80  # 100 - 20 removed
            assert robustness_score["removed_samples"] == 20

    def test_adversarial_filtering_validation(self):
        """Test adversarially filtered dataset validation."""
        with patch("nero.data.datasets.datasets.CIFAR10") as mock_cifar10:
            mock_train = MagicMock()
            mock_train.__len__ = MagicMock(return_value=50000)
            mock_test = MagicMock()
            mock_test.__len__ = MagicMock(return_value=100)
            mock_test.__getitem__ = MagicMock(
                side_effect=lambda idx: (torch.randn(3, 32, 32), idx % 10)
            )

            mock_cifar10.side_effect = [mock_train, mock_test]

            base_dataset = CIFAR10Dataset(download=False)
            filtered_dataset = AdversariallyFilteredDataset(
                base_dataset, filter_ratio=0.2, seed=42
            )

            assert filtered_dataset.validate_integrity() is True

    def test_custom_spurious_features(self):
        """Test custom spurious features specification."""
        with patch("nero.data.datasets.datasets.CIFAR10") as mock_cifar10:
            mock_train = MagicMock()
            mock_train.__len__ = MagicMock(return_value=50000)
            mock_test = MagicMock()
            mock_test.__len__ = MagicMock(return_value=100)
            mock_test.__getitem__ = MagicMock(
                side_effect=lambda idx: (torch.randn(3, 32, 32), idx % 10)
            )

            mock_cifar10.side_effect = [mock_train, mock_test]

            base_dataset = CIFAR10Dataset(download=False)
            custom_features = ["background_color", "position_bias"]
            filtered_dataset = AdversariallyFilteredDataset(
                base_dataset,
                filter_ratio=0.2,
                spurious_features=custom_features,
                seed=42,
            )

            assert filtered_dataset.spurious_features == custom_features

    def test_edge_cases(self):
        """Test edge cases for distribution shift datasets."""
        with patch("nero.data.datasets.datasets.CIFAR10") as mock_cifar10:
            mock_train = MagicMock()
            mock_train.__len__ = MagicMock(return_value=50000)
            mock_test = MagicMock()
            mock_test.__len__ = MagicMock(return_value=10)  # Small dataset
            mock_test.__getitem__ = MagicMock(
                side_effect=lambda idx: (torch.randn(3, 32, 32), idx % 10)
            )

            mock_cifar10.side_effect = [mock_train, mock_test]

            base_dataset = CIFAR10Dataset(download=False)

            # Test with very high conflict ratio
            bias_dataset = TextureShapeBiasDataset(
                base_dataset, conflict_ratio=0.9, seed=42
            )
            assert bias_dataset.validate_integrity() is True

            # Test with very high filter ratio
            filtered_dataset = AdversariallyFilteredDataset(
                base_dataset, filter_ratio=0.8, seed=42
            )
            assert filtered_dataset.validate_integrity() is True

    def test_reproducibility_with_seeds(self):
        """Test that datasets are reproducible with same seeds."""
        with patch("nero.data.datasets.datasets.CIFAR10") as mock_cifar10:
            mock_train = MagicMock()
            mock_train.__len__ = MagicMock(return_value=50000)
            mock_test = MagicMock()
            mock_test.__len__ = MagicMock(return_value=100)
            mock_test.__getitem__ = MagicMock(
                side_effect=lambda idx: (torch.randn(3, 32, 32), idx % 10)
            )

            mock_cifar10.side_effect = [mock_train, mock_test, mock_train, mock_test]

            base_dataset1 = CIFAR10Dataset(download=False)
            base_dataset2 = CIFAR10Dataset(download=False)

            # Create two datasets with same seed
            bias_dataset1 = TextureShapeBiasDataset(
                base_dataset1, conflict_ratio=0.3, seed=42
            )
            bias_dataset2 = TextureShapeBiasDataset(
                base_dataset2, conflict_ratio=0.3, seed=42
            )

            # Should have identical conflict patterns
            assert bias_dataset1.conflict_labels == bias_dataset2.conflict_labels
