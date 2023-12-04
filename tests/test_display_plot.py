import sys
from unittest.mock import MagicMock, Mock, patch
import pytest
import numpy as np

from soundata import display_plot


class MockDataset:
    def __init__(self, clip_durations=None, tags=None, subdatasets=None, layers=None):
        self.clip_durations = clip_durations or []
        self.tags = tags or {}
        self.subdatasets = subdatasets or {}
        self.layers = layers or {}

        self._index = {"clips": {i: f"clip_{i}" for i in range(len(clip_durations))}}
        self._metadata = {
            clip_id: {
                "subdataset": self.subdatasets.get(clip_id, None),
                **{
                    f"subdataset_layer_{layer}": self.layers.get(layer, {}).get(
                        clip_id, None
                    )
                    for layer in range(len(self.layers))
                },
            }
            for clip_id in self._index["clips"]
        }

    def clip(self, clip_id):
        return MockClip(self.clip_durations[int(clip_id)], self.tags.get(clip_id, []))


class MockClip:
    def __init__(self, duration, tags):
        self.audio = (np.random.random(int(duration * 44100)), 44100)
        self.tags = MockTags(tags)


class MockTags:
    def __init__(self, labels):
        self.labels = labels


def test_compute_clip_statistics():
    # Test with mock data
    mock_clip_durations = [30, 60, 45, 90, 120]  # durations in seconds
    dataset = MockDataset(mock_clip_durations)  # Assume this is a mock class you define

    stats = display_plot.compute_clip_statistics(dataset)
    assert stats["total_duration"] == sum(mock_clip_durations)
    assert stats["mean_duration"] == np.mean(mock_clip_durations)
    assert stats["median_duration"] == np.median(mock_clip_durations)
    assert stats["std_deviation"] == np.std(mock_clip_durations)
    assert stats["min_duration"] == min(mock_clip_durations)
    assert stats["max_duration"] == max(mock_clip_durations)

    # Test with empty dataset
    empty_dataset = MockDataset([])
    with pytest.raises(ValueError):
        display_plot.compute_clip_statistics(empty_dataset)


def test_perform_dataset_exploration_initialization():
    """Test the initialization and default states of widgets."""
    # Create a mock instance with specific return values for the widget attributes
    exploration_instance = Mock()
    exploration_instance.event_dist_check = Mock(value=True)
    exploration_instance.dataset_analysis_check = Mock(value=False)
    exploration_instance.audio_plot_check = Mock(value=True)

    # Call the function with the mock instance
    display_plot.perform_dataset_exploration(exploration_instance)

    # Test initial values of widgets
    assert exploration_instance.event_dist_check.value is True
    assert exploration_instance.dataset_analysis_check.value is False
    assert exploration_instance.audio_plot_check.value is True


def test_time_unit_conversion(mocker):
    mock_stats = mocker.patch("soundata.display_plot.compute_clip_statistics")
    # Use values that would trigger conversion to minutes and hours
    mock_stats.return_value = {
        "durations": [120, 180, 240],  # durations in seconds
        "total_duration": 540,  # total duration in seconds
        "mean_duration": 180,  # mean in seconds
        "median_duration": 180,  # median in seconds
        "std_deviation": 60,
        "min_duration": 120,
        "max_duration": 240,
    }

    mock_show = mocker.patch("matplotlib.pyplot.show")

    dataset = MagicMock()
    dataset._index = {"clips": [1, 2, 3]}

    display_plot.plot_clip_durations(dataset)

    mock_stats.assert_called_once()
    mock_show.assert_called_once()
