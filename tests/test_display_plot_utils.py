import sys
import threading
from unittest.mock import MagicMock, Mock, patch
import pytest
import numpy as np

from soundata import display_plot_utils
import soundata
import simpleaudio as sa  # Alternative library for audio playback


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

    stats = display_plot_utils.compute_clip_statistics(dataset)
    assert stats["total_duration"] == sum(mock_clip_durations)
    assert stats["mean_duration"] == np.mean(mock_clip_durations)
    assert stats["median_duration"] == np.median(mock_clip_durations)
    assert stats["std_deviation"] == np.std(mock_clip_durations)
    assert stats["min_duration"] == min(mock_clip_durations)
    assert stats["max_duration"] == max(mock_clip_durations)

    # Test with empty dataset
    empty_dataset = MockDataset([])
    with pytest.raises(ValueError):
        display_plot_utils.compute_clip_statistics(empty_dataset)


def test_perform_dataset_exploration_initialization():
    """Test the initialization and default states of widgets."""
    # Create a mock instance with specific return values for the widget attributes
    exploration_instance = Mock()
    exploration_instance.event_dist_check = Mock(value=True)
    exploration_instance.dataset_analysis_check = Mock(value=False)
    exploration_instance.audio_plot_check = Mock(value=True)
    exploration_instance.output = Mock()
    exploration_instance.loader = Mock()

    # Call the function with the mock instance
    display_plot_utils.perform_dataset_exploration(exploration_instance)
    exploration_instance.on_button_clicked(Mock())
    # Test initial values of widgets
    assert exploration_instance.event_dist_check.value is True
    assert exploration_instance.dataset_analysis_check.value is False
    assert exploration_instance.audio_plot_check.value is True


def test_time_unit_conversion(mocker):
    mock_stats = mocker.patch("soundata.display_plot_utils.compute_clip_statistics")
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

    display_plot_utils.plot_clip_durations(dataset)

    mock_stats.assert_called_once()
    mock_show.assert_called_once()


def test_visualize_audio(mocker):
    # Mock audio data and sample rate
    mock_audio = np.random.rand(44100)  # 1 second of random audio data
    mock_sr = 44100  # Sample rate
    expected_duration = len(mock_audio) / mock_sr
    # Mock the clip method to return the mock audio and sample rate
    mock_clip = mocker.MagicMock()
    mock_clip.audio = (mock_audio, mock_sr)
    dataset = soundata.initialize("urbansound8k")
    mocker.patch.object(dataset, "clip", return_value=mock_clip)
    # Create an instance of your class
    instance = dataset
    instance._index = {"clips": {"dummy_clip_id": None}}  # Set up _index for the test

    display_plot_utils.visualize_audio(instance, "dummy_clip_id")

    instance.clip.assert_called_once_with("dummy_clip_id")


def test_visualize_audio_clip_id_none(mocker):
    # Mock the necessary attributes and methods
    mock_index = {"clips": {"clip1": "data1", "clip2": "data2"}}
    dataset = soundata.initialize("urbansound8k")
    dataset._index = mock_index

    # Mock the clip method to return a clip with an audio tuple (audio_data, sample_rate)
    mock_audio_data = np.random.rand(44100)  # Random audio data for testing
    mock_sample_rate = 44100  # Sample rate
    mock_clip = MagicMock()
    mock_clip.audio = (mock_audio_data, mock_sample_rate)
    mocker.patch.object(dataset, "clip", return_value=mock_clip)

    # Mock np.random.choice to control its output
    mocker.patch("numpy.random.choice", return_value="clip1")

    # Call the method with clip_id as None
    display_plot_utils.visualize_audio(dataset, None)

    # Assert that np.random.choice was called with the correct arguments
    np.random.choice.assert_called_once_with(list(mock_index["clips"].keys()))

    # Assert that the 'clip' method was called with the 'clip_id' chosen by np.random.choice
    dataset.clip.assert_called_once_with("clip1")


def test_play_segment(mocker):
    # Prepare the mock data
    mock_clip_id = "clip123"
    mock_audio_data = np.random.rand(44100)  # Random audio data for testing
    mock_sample_rate = 44100  # Sample rate

    # Mock the clip method to return a clip with an audio tuple (audio_data, sample_rate)
    mock_clip = MagicMock()
    mock_clip.audio = (mock_audio_data, mock_sample_rate)
    dataset = soundata.initialize("urbansound8k")
    mocker.patch.object(dataset, "clip", return_value=mock_clip)

    # Mock dependencies used in play_segment
    mocker.patch.object(sa, "play_buffer")
    mocker.patch.object(threading.Thread, "start")

    # Call the visualize_audio function with a specific clip_id
    display_plot_utils.visualize_audio(dataset, mock_clip_id)

    # Assert that the 'clip' method was called with the provided 'clip_id'
    dataset.clip.assert_called_once_with(mock_clip_id)
