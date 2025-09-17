"""Dataset implementations for NERO."""

import hashlib
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from nero.domain.interfaces import Dataset


class MNISTDataset(Dataset):
    """MNIST dataset implementation with automatic downloading and validation."""

    # Expected checksums for MNIST dataset files
    EXPECTED_CHECKSUMS = {
        'train-images-idx3-ubyte.gz': '440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609',
        'train-labels-idx1-ubyte.gz': '3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c',
        't10k-images-idx3-ubyte.gz': '8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6',
        't10k-labels-idx1-ubyte.gz': 'f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6',
    }

    def __init__(
        self,
        root: str = './data',
        download: bool = True,
        transform: transforms.Compose | None = None,
        default_batch_size: int = 64,
    ):
        """
        Initialize MNIST dataset.

        Args:
            root: Root directory for dataset storage
            download: Whether to download dataset if not found
            transform: Optional transform to apply to samples
            default_batch_size: Default batch size for data loaders
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.default_batch_size = default_batch_size

        # Default transform: normalize to [0, 1] and standardize
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
                ]
            )
        else:
            self.transform = transform

        # Load datasets with progress feedback
        if download:
            print(
                'Downloading MNIST dataset (this may take a few minutes on first run)...'
            )
            print('Progress: ', end='', flush=True)

        self.train_dataset = datasets.MNIST(
            root=str(self.root), train=True, download=download, transform=self.transform
        )

        if download:
            print('✓ Training data downloaded')
            print('Progress: ', end='', flush=True)

        self.test_dataset = datasets.MNIST(
            root=str(self.root),
            train=False,
            download=download,
            transform=self.transform,
        )

        if download:
            print('✓ Test data downloaded')
            print('Validating dataset integrity...')
        elif not self._dataset_exists():
            print(
                'MNIST dataset not found. Set download=True to download automatically.'
            )

        # Validate dataset integrity after loading
        if not self.validate_integrity():
            raise RuntimeError('MNIST dataset integrity validation failed')
        elif download:
            print('✓ Dataset integrity validated successfully')

    @property
    def name(self) -> str:
        """Return the name of the dataset."""
        return 'mnist'

    @property
    def num_classes(self) -> int:
        """Return the number of classes in the dataset."""
        return 10

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Return the shape of input samples (excluding batch dimension)."""
        return (1, 28, 28)  # Single channel, 28x28 pixels

    def get_train_loader(
        self, batch_size: int | None = None, shuffle: bool = True
    ) -> DataLoader:
        """
        Get training data loader.

        Args:
            batch_size: Batch size (uses default if None)
            shuffle: Whether to shuffle the data

        Returns:
            DataLoader for training data
        """
        if batch_size is None:
            batch_size = self.default_batch_size

        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

    def get_test_loader(self, batch_size: int | None = None) -> DataLoader:
        """
        Get test data loader.

        Args:
            batch_size: Batch size (uses default if None)

        Returns:
            DataLoader for test data
        """
        if batch_size is None:
            batch_size = self.default_batch_size

        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

    def get_sample(self, index: int) -> tuple[torch.Tensor, int]:
        """
        Get a single sample by index from test set.

        Args:
            index: Sample index

        Returns:
            Tuple of (sample, target)
        """
        if index < 0 or index >= len(self.test_dataset):
            raise IndexError(
                f'Index {index} out of range for test dataset of size {len(self.test_dataset)}'
            )

        sample, target = self.test_dataset[index]
        return sample, target

    def get_sample_batch(
        self, batch_size: int, indices: list[int] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a batch of samples from test set.

        Args:
            batch_size: Number of samples to return
            indices: Specific indices to sample (random if None)

        Returns:
            Tuple of (samples, targets)
        """
        if indices is None:
            # Generate random indices from test set
            indices = torch.randperm(len(self.test_dataset))[:batch_size].tolist()
        else:
            if len(indices) != batch_size:
                raise ValueError(
                    f'Length of indices ({len(indices)}) must match batch_size ({batch_size})'
                )

            # Validate indices
            for idx in indices:
                if idx < 0 or idx >= len(self.test_dataset):
                    raise IndexError(
                        f'Index {idx} out of range for test dataset of size {len(self.test_dataset)}'
                    )

        # Create subset and data loader
        subset = Subset(self.test_dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

        # Get the batch
        samples, targets = next(iter(loader))
        return samples, targets

    def validate_integrity(self) -> bool:
        """
        Validate dataset integrity using checksums.

        Returns:
            True if dataset is valid, False otherwise
        """
        try:
            mnist_dir = self.root / 'MNIST' / 'raw'

            if not mnist_dir.exists():
                # Dataset not downloaded yet, assume it will be valid after download
                return True

            # Check if all expected files exist and have correct checksums
            for filename, expected_checksum in self.EXPECTED_CHECKSUMS.items():
                file_path = mnist_dir / filename

                if not file_path.exists():
                    print(f'Warning: MNIST file {filename} not found')
                    return False

                # Compute SHA256 checksum
                actual_checksum = self._compute_file_checksum(file_path)

                if actual_checksum != expected_checksum:
                    print(f'Warning: MNIST file {filename} has incorrect checksum')
                    print(f'Expected: {expected_checksum}')
                    print(f'Actual: {actual_checksum}')
                    return False

            # Additional validation: check dataset sizes
            if len(self.train_dataset) != 60000:
                print(
                    f'Warning: Expected 60000 training samples, got {len(self.train_dataset)}'
                )
                return False

            if len(self.test_dataset) != 10000:
                print(
                    f'Warning: Expected 10000 test samples, got {len(self.test_dataset)}'
                )
                return False

            # Validate sample shapes and ranges
            sample, target = self.test_dataset[0]
            if sample.shape != (1, 28, 28):
                print(f'Warning: Expected sample shape (1, 28, 28), got {sample.shape}')
                return False

            if not (0 <= target <= 9):
                print(f'Warning: Expected target in range [0, 9], got {target}')
                return False

            return True

        except Exception as e:
            print(f'Error during dataset validation: {e}')
            return False

    def _compute_file_checksum(self, file_path: Path) -> str:
        """
        Compute SHA256 checksum of a file.

        Args:
            file_path: Path to the file

        Returns:
            Hexadecimal SHA256 checksum
        """
        sha256_hash = hashlib.sha256()

        with open(file_path, 'rb') as f:
            # Read file in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(4096), b''):
                sha256_hash.update(chunk)

        return sha256_hash.hexdigest()

    def _dataset_exists(self) -> bool:
        """Check if MNIST dataset files already exist."""
        mnist_dir = self.root / 'MNIST' / 'raw'
        if not mnist_dir.exists():
            return False

        # Check if all expected files exist
        for filename in self.EXPECTED_CHECKSUMS.keys():
            if not (mnist_dir / filename).exists():
                return False

        return True

    def get_data_statistics(self) -> dict[str, float | int | dict[str, float]]:
        """
        Compute dataset statistics for validation and reporting.

        Returns:
            Dictionary containing dataset statistics
        """
        # Compute statistics on a sample of the training data
        sample_loader = DataLoader(self.train_dataset, batch_size=1000, shuffle=False)

        all_samples = []
        all_targets = []

        # Collect first few batches for statistics
        for i, (samples, targets) in enumerate(sample_loader):
            all_samples.append(samples)
            all_targets.append(targets)
            if i >= 10:  # Use first ~10k samples for statistics
                break

        samples_tensor = torch.cat(all_samples, dim=0)
        targets_tensor = torch.cat(all_targets, dim=0)

        return {
            'mean': float(samples_tensor.mean()),
            'std': float(samples_tensor.std()),
            'min': float(samples_tensor.min()),
            'max': float(samples_tensor.max()),
            'num_train_samples': len(self.train_dataset),
            'num_test_samples': len(self.test_dataset),
            'num_classes': self.num_classes,
            'class_distribution': {
                str(i): float((targets_tensor == i).sum()) / len(targets_tensor)
                for i in range(self.num_classes)
            },
        }
