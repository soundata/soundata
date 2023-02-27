import os
import numpy as np
import pytest

from tests.test_utils import run_clip_tests, DEFAULT_DATA_HOME

from soundata import annotations
from soundata.datasets import starss2022

TEST_DATA_HOME = "tests/resources/sound_datasets/starss2022"


def test_clip():
    default_clipid = "foa_dev/dev-train-sony/fold3_room21_mix001"
    dataset = starss2022.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)

    expected_attributes = {
        "clip_id": "foa_dev/dev-train-sony/fold3_room21_mix001",
        "audio_path": "tests/resources/sound_datasets/starss2022/foa_dev/dev-train-sony/fold3_room21_mix001.wav",
        "csv_path": "tests/resources/sound_datasets/starss2022/metadata_dev/dev-train-sony/fold3_room21_mix001.csv",
        "format": "foa",
        "set": "dev",
        "split": "train",
    }

    expected_property_types = {
        "audio": tuple,
        "spatial_events": starss2022.SpatialEvents,
    }

    run_clip_tests(clip, expected_attributes, expected_property_types)


def test_load_audio():
    dataset = starss2022.Dataset(TEST_DATA_HOME)
    clip = dataset.clip("foa_dev/dev-train-sony/fold3_room21_mix001")
    audio_path = clip.audio_path
    audio, sr = starss2022.load_audio(audio_path)
    assert sr == 24000
    assert type(audio) is np.ndarray
    assert len(audio.shape) == 2  # check audio is loaded as 4 channels
    assert audio.shape[0] == 4  # check audio is loaded as 4 channels
    assert audio.shape[1] == 24000  # check audio duration in samples is as expected


def test_load_SpatialEvents():
    dataset = starss2022.Dataset(TEST_DATA_HOME)
    clip = dataset.clip("foa_dev/dev-train-sony/fold3_room21_mix001")
    annotations_path = clip.csv_path
    annotations = starss2022.load_spatialevents(annotations_path)

    confidence = [1.0] * 4
    intervals = [
        [[1.2, 2.7]],
        [[3.2, 3.7], [5.7, 6.3]],
        [[3.8, 4.8]],
        [[3.8, 4.8]],
    ]

    azimuths = [
        [[-98]],
        [[-52, -52, -52, -51, -51, -51], [-54]],
        [
            [-51, -51, -51, -51, -52, -52, -53, -53, -54, -54, -54],
        ],
        [[-98]],
    ]

    elevations = [
        [[-16]],
        [[-38, -38, -38, -38, -38, -39], [-37]],
        [[-39, -39, -39, -38, -38, -38, -37, -37, -37, -37, -37]],
        [[-16]],
    ]
    distances = [
        [np.array([None] * len(azimuth)) for azimuth in event_azimuths]
        for event_azimuths in azimuths
    ]

    labels = ["1", "1", "4", "4"]
    track_number_indices = ["1", "2", "2", "1"]
    assert np.allclose(annotations.time_step, 0.1)
    assert np.allclose(confidence, annotations.confidence)
    for pair in [
        zip(elevations, annotations.elevations),
        zip(azimuths, annotations.azimuths),
    ]:
        for event_test_data, event_data in pair:
            for test_data, data in zip(event_test_data, event_data):
                assert np.allclose(test_data, data)
    for pair in [zip(distances, annotations.distances)]:
        for event_test_data, event_data in pair:
            for test_data, data in zip(event_test_data, event_data):
                test_data == data
    for test_label, label in zip(labels, annotations.labels):
        assert test_label == label
    for test_track_index, track_index in zip(
        track_number_indices, annotations.track_number_index
    ):
        assert test_track_index == track_index
    with pytest.raises(ValueError):
        starss2022.validate_time_steps(0.1, np.array([[4, 5, 7]]), [1, 0])
    with pytest.raises(ValueError):
        starss2022.validate_time_steps(
            0.1, np.array([[4, 5, 7], [1, 2, 3]]), [0.0, 0.2]
        )
    with pytest.raises(ValueError):
        # locations are not 3D
        starss2022.validate_locations(np.array([[4, 5], [2, 3]]))
    with pytest.raises(ValueError):
        # distance is not None
        starss2022.validate_locations(np.array([[90, 5, None], [2, 3, 4]]))
    with pytest.raises(ValueError):
        # elevation is greater than 90
        starss2022.validate_locations(np.array([[91, 5, None], [2, 3, None]]))
    with pytest.raises(ValueError):
        # elevation is greater than 181
        starss2022.validate_locations(np.array([[90, 181, None], [2, 3, None]]))


def test_to_jams():
    default_clipid = "foa_dev/dev-train-sony/fold3_room21_mix001"
    dataset = starss2022.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)
    jam = clip.to_jams()

    # Validate starss2022 jam schema
    assert jam.validate()
