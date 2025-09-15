"""Unit tests for VRAMManager that avoid CUDA initialization issues."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from nero.orchestration.vram_manager import VRAMError, VRAMManager


class TestVRAMManagerUnit:
    """Unit tests for VRAMManager core functionality."""

    @pytest.fixture
    def mock_cuda_available(self):
        """Mock CUDA availability."""
        with patch("torch.cuda.is_available", return_value=True):
            yield

    @pytest.fixture
    def mock_device_properties(self):
        """Mock device properties."""
        mock_props = Mock()
        mock_props.total_memory = 8 * 1024**3  # 8GB
        with patch("torch.cuda.get_device_properties", return_value=mock_props):
            yield mock_props

    @pytest.fixture
    def vram_manager(self, mock_cuda_available, mock_device_properties):
        """Create VRAMManager instance for testing."""
        return VRAMManager(safety_margin=0.1)

    def test_init_success(self, mock_cuda_available, mock_device_properties):
        """Test successful initialization."""
        manager = VRAMManager(safety_margin=0.2)

        assert manager.safety_margin == 0.2
        assert manager.total_memory == 8 * 1024**3
        assert manager.available_memory == 8 * 1024**3 * 0.8

    def test_init_no_cuda(self):
        """Test initialization when CUDA is not available."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            pytest.raises(VRAMError, match="CUDA is not available"),
        ):
            VRAMManager()

    def test_init_device_properties_error(self, mock_cuda_available):
        """Test initialization when device properties fail."""
        with (
            patch(
                "torch.cuda.get_device_properties",
                side_effect=RuntimeError("Device error"),
            ),
            pytest.raises(VRAMError, match="Failed to get device properties"),
        ):
            VRAMManager()

    def test_safety_margin_bounds(self, mock_cuda_available, mock_device_properties):
        """Test safety margin is bounded between 0 and 1."""
        # Test negative margin
        manager = VRAMManager(safety_margin=-0.1)
        assert manager.safety_margin == 0.0

        # Test margin > 1
        manager = VRAMManager(safety_margin=1.5)
        assert manager.safety_margin == 1.0

    def test_optimize_batch_size_binary_search_logic(self, vram_manager):
        """Test the binary search logic without CUDA operations."""
        # Mock all CUDA-related operations
        mock_model = MagicMock()
        mock_input = MagicMock()

        with (
            patch.object(mock_model, "to", return_value=mock_model),
            patch.object(mock_input, "to", return_value=mock_input),
            patch.object(vram_manager, "_test_batch_size") as mock_test,
            patch.object(
                vram_manager, "_get_current_memory_usage", return_value=3.2 * 1024**3
            ),
        ):
            # Simulate binary search: min=1 fits, then binary search
            # Just test that it returns a reasonable result
            mock_test.side_effect = [True, True, True, True, False, True]

            result = vram_manager.optimize_batch_size(
                mock_model, mock_input, initial_batch_size=16
            )

            # The binary search should find a batch size >= min_batch_size
            assert result >= 1
            assert (
                len(mock_test.call_args_list) >= 2
            )  # At least min test + some binary search

    def test_optimize_batch_size_min_fails(self, vram_manager):
        """Test when minimum batch size doesn't fit."""
        mock_model = MagicMock()
        mock_input = MagicMock()

        with (
            patch.object(mock_model, "to", return_value=mock_model),
            patch.object(mock_input, "to", return_value=mock_input),
            patch.object(vram_manager, "_test_batch_size", return_value=False),
            patch.object(
                vram_manager, "_get_current_memory_usage", return_value=1 * 1024**3
            ),
            pytest.raises(VRAMError, match="Insufficient VRAM"),
        ):
            vram_manager.optimize_batch_size(mock_model, mock_input, min_batch_size=1)

    def test_test_batch_size_memory_limit_logic(self, vram_manager):
        """Test the memory limit checking logic."""
        mock_model = MagicMock()
        mock_input = MagicMock()

        # Test case where memory usage exceeds limit
        with (
            patch("torch.cuda.empty_cache"),
            patch.object(
                vram_manager, "_get_current_memory_usage", return_value=10 * 1024**3
            ),
        ):
            # Mock all tensor operations to avoid CUDA
            mock_batch_input = MagicMock()
            mock_batch_input.requires_grad_ = MagicMock(return_value=mock_batch_input)

            with (
                patch.object(mock_input, "unsqueeze") as mock_unsqueeze,
                patch.object(
                    mock_unsqueeze.return_value, "repeat", return_value=mock_batch_input
                ),
                patch.object(mock_model, "to", return_value=mock_model),
                patch.object(mock_input, "to", return_value=mock_input),
                patch.object(mock_model, "forward", return_value=MagicMock()),
                patch("torch.randint", return_value=MagicMock()),
                patch("torch.nn.functional.cross_entropy", return_value=MagicMock()),
            ):
                result = vram_manager._test_batch_size(mock_model, mock_input, 4)
                assert result is False  # Should fail due to memory limit

    def test_test_batch_size_oom_handling(self, vram_manager):
        """Test OOM error handling."""
        mock_model = MagicMock()
        mock_input = MagicMock()

        with (
            patch("torch.cuda.empty_cache"),
            patch.object(mock_model, "to", return_value=mock_model),
            patch.object(mock_input, "to", return_value=mock_input),
            patch.object(
                mock_model, "forward", side_effect=Exception("CUDA out of memory")
            ),
        ):
            result = vram_manager._test_batch_size(mock_model, mock_input, 4)
            assert result is False

    @patch("torch.cuda.memory_allocated")
    def test_get_current_memory_usage(self, mock_memory_allocated, vram_manager):
        """Test getting current memory usage."""
        mock_memory_allocated.return_value = 2 * 1024**3

        result = vram_manager._get_current_memory_usage()
        assert result == 2 * 1024**3
        mock_memory_allocated.assert_called_once_with(vram_manager.device)

    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.memory_reserved")
    def test_get_memory_stats(
        self, mock_memory_reserved, mock_memory_allocated, vram_manager
    ):
        """Test getting memory statistics."""
        mock_memory_allocated.return_value = 2 * 1024**3
        mock_memory_reserved.return_value = 3 * 1024**3

        stats = vram_manager.get_memory_stats()

        expected_stats = {
            "total_gb": 8.0,
            "available_gb": 7.2,  # 8 * 0.9
            "allocated_gb": 2.0,
            "reserved_gb": 3.0,
            "free_gb": 5.0,  # 8 - 3
            "utilization": 0.25,  # 2/8
            "safety_margin": 0.1,
        }

        assert stats == expected_stats

    @patch("torch.cuda.empty_cache")
    def test_clear_cache(self, mock_empty_cache, vram_manager):
        """Test clearing GPU cache."""
        vram_manager.clear_cache()
        mock_empty_cache.assert_called_once()

    @patch("torch.cuda.memory_allocated")
    def test_check_memory_available(self, mock_memory_allocated, vram_manager):
        """Test memory availability check."""
        mock_memory_allocated.return_value = 2 * 1024**3

        # Test when memory is available
        result = vram_manager.check_memory_available(1 * 1024**3)
        assert result is True

        # Test when memory is not available
        result = vram_manager.check_memory_available(6 * 1024**3)
        assert result is False

    def test_get_recommended_batch_size_logic(self, vram_manager):
        """Test recommended batch size calculation logic."""
        mock_model = MagicMock()
        mock_input = MagicMock()

        # Mock memory usage: initial 1GB, after single sample 1.1GB (100MB per sample)
        with (
            patch("torch.cuda.empty_cache"),
            patch.object(
                vram_manager,
                "_get_current_memory_usage",
                side_effect=[1 * 1024**3, 1.1 * 1024**3],
            ),
        ):
            # Mock tensor operations
            mock_single_input = MagicMock()

            with (
                patch.object(mock_input, "unsqueeze", return_value=mock_single_input),
                patch.object(mock_model, "to", return_value=mock_model),
                patch.object(mock_input, "to", return_value=mock_input),
                patch.object(mock_model, "forward", return_value=MagicMock()),
            ):
                result = vram_manager.get_recommended_batch_size(
                    mock_model, mock_input, target_memory_usage=0.8
                )

                # Available memory: 7.2GB, target: 5.76GB, memory per sample: 0.1GB
                # Expected batch size: 57 (rounded down from 57.6)
                assert result == 57

    def test_get_recommended_batch_size_error_fallback(self, vram_manager):
        """Test fallback when recommended batch size calculation fails."""
        mock_model = MagicMock()
        mock_input = MagicMock()

        with (
            patch("torch.cuda.empty_cache"),
            patch.object(mock_model, "to", side_effect=RuntimeError("Error")),
        ):
            result = vram_manager.get_recommended_batch_size(mock_model, mock_input)
            assert result == 32  # Default fallback

    def test_get_recommended_batch_size_zero_memory_fallback(self, vram_manager):
        """Test fallback when memory per sample is zero or negative."""
        mock_model = MagicMock()
        mock_input = MagicMock()

        # Mock same memory usage before and after (no memory increase)
        with (
            patch("torch.cuda.empty_cache"),
            patch.object(
                vram_manager,
                "_get_current_memory_usage",
                side_effect=[1 * 1024**3, 1 * 1024**3],
            ),
        ):
            # Mock tensor operations
            mock_single_input = MagicMock()

            with (
                patch.object(mock_input, "unsqueeze", return_value=mock_single_input),
                patch.object(mock_model, "to", return_value=mock_model),
                patch.object(mock_input, "to", return_value=mock_input),
                patch.object(mock_model, "forward", return_value=MagicMock()),
            ):
                result = vram_manager.get_recommended_batch_size(mock_model, mock_input)
                assert result == 32  # Default fallback
