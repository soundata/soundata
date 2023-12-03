import sys
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
