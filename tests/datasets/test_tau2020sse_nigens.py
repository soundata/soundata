import os
import numpy as np
import pytest

from tests.test_utils import run_clip_tests, DEFAULT_DATA_HOME

from soundata import annotations
from soundata.datasets import tau2020sse_nigens


TEST_DATA_HOME = "tests/resources/sound_datasets/tau2020sse_nigens"


def test_clip():
    default_clipid = "foa_dev/fold1_room1_mix001_ov1"
    dataset = tau2020sse_nigens.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)

    expected_attributes = {
        "clip_id": "foa_dev/fold1_room1_mix001_ov1",
        "audio_path": "tests/resources/sound_datasets/tau2020sse_nigens/foa_dev/fold1_room1_mix001_ov1.wav",
        "csv_path": "tests/resources/sound_datasets/tau2020sse_nigens/metadata_dev/fold1_room1_mix001_ov1.csv",
        "format": "foa",
        "set": "dev",
    }

    expected_property_types = {
        "audio": tuple,
        "spatial_events": tau2020sse_nigens.SpatialEvents,
    }

    run_clip_tests(clip, expected_attributes, expected_property_types)


def test_load_audio():
    dataset = tau2020sse_nigens.Dataset(TEST_DATA_HOME)
    clip = dataset.clip("foa_dev/fold1_room1_mix001_ov1")
    audio_path = clip.audio_path
    audio, sr = tau2020sse_nigens.load_audio(audio_path)
    assert sr == 24000
    assert type(audio) is np.ndarray
    assert len(audio.shape) == 2  # check audio is loaded as 4 channels
    assert audio.shape[0] == 4  # check audio is loaded as 4 channels
    assert audio.shape[1] == 24000  # check audio duration in samples is as expected


def test_load_SpatialEvents():
    dataset = tau2020sse_nigens.Dataset(TEST_DATA_HOME)
    clip = dataset.clip("foa_dev/fold1_room1_mix001_ov1")
    annotations_path = clip.csv_path
    annotations = tau2020sse_nigens.load_spatialevents(annotations_path)

    confidence = [1.0] * 6
    intervals = [
        [[0, 0], [0.8, 1.0]],
        [[0.3, 0.5]],
        [[0.8, 0.9]],
        [[0.9, 0.9]],
        [[0.9, 2.8], [3.1, 4.8], [6.5, 7.6]],
        [[11.0, 11.1]],
    ]

    azimuths = [
        [[10], [10, 11, 12]],
        [[5]],
        [[6, 5]],
        [[-5]],
        [
            [25, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0, 1, 3, 5, 7, 9],
            [15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49],
            [83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105],
        ],
        [[-47]],
    ]

    elevations = [
        [[2], [2, 2, 2]],
        [[4]],
        [[7, 6]],
        [[6]],
        [
            [1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 2, 4, 4, 5, 5, 5],
            [5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 4, 4, 4, 5, 5],
            [6, 5, 5, 5, 4, 5, 5, 5, 4, 4, 4, 4],
        ],
        [[9]],
    ]
    distances = [
        [np.array([None] * len(azimuth)) for azimuth in event_azimuths]
        for event_azimuths in azimuths
    ]

    labels = ["1", "2", "4", "4", "5", "6"]
    clip_number_indices = ["0", "0", "0", "1", "0", "0"]
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
    for test_clip_index, clip_index in zip(
        clip_number_indices, annotations.clip_number_index
    ):
        assert test_clip_index == clip_index
    with pytest.raises(ValueError):
        tau2020sse_nigens.validate_time_steps(0.1, [4, 5, 7], [2, 1])
    with pytest.raises(ValueError):
        tau2020sse_nigens.validate_time_steps(
            0.1, np.array([[4, 5, 7], [1, 2, 3]]), [0.0, 0.2]
        )
    with pytest.raises(ValueError):
        # locations are not 3D
        tau2020sse_nigens.validate_locations(np.array([[4, 5], [2, 3]]))
    with pytest.raises(ValueError):
        # distance is not None
        tau2020sse_nigens.validate_locations(np.array([[90, 5, None], [2, 3, 4]]))
    with pytest.raises(ValueError):
        # elevation is greater than 90
        tau2020sse_nigens.validate_locations(np.array([[91, 5, None], [2, 3, None]]))
    with pytest.raises(ValueError):
        # elevation is greater than 181
        tau2020sse_nigens.validate_locations(np.array([[90, 181, None], [2, 3, None]]))


def test_to_jams():
    default_clipid = "foa_dev/fold1_room1_mix001_ov1"
    dataset = tau2020sse_nigens.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)
    jam = clip.to_jams()

    # Validate tau2020sse_nigens jam schema
    assert jam.validate()
