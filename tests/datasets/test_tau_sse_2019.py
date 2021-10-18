import os
import numpy as np
import pytest

from tests.test_utils import run_clip_tests

from soundata import annotations
from soundata.datasets import tau_sse_2019


TEST_DATA_HOME = "tests/resources/sound_datasets/tau_sse_2019"


def test_clip():
    default_clipid = "foa_dev/split1_ir0_ov1_1"
    dataset = tau_sse_2019.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)

    expected_attributes = {
        "clip_id": "foa_dev/split1_ir0_ov1_1",
        "audio_path": "tests/resources/sound_datasets/tau_sse_2019/foa_dev/split1_ir0_ov1_1.wav",
        "csv_path": "tests/resources/sound_datasets/tau_sse_2019/metadata_dev/split1_ir0_ov1_1.csv",
        "format": "foa",
        "set": "dev",
    }

    expected_property_types = {
        "audio": tuple,
        "spatial_events": tau_sse_2019.SpatialEvents,
    }

    run_clip_tests(clip, expected_attributes, expected_property_types)


def test_load_audio():
    dataset = tau_sse_2019.Dataset(TEST_DATA_HOME)
    clip = dataset.clip("foa_dev/split1_ir0_ov1_1")
    audio_path = clip.audio_path
    audio, sr = tau_sse_2019.load_audio(audio_path)
    assert sr == 48000
    assert type(audio) is np.ndarray
    assert len(audio.shape) == 2  # check audio is multichannel
    assert audio.shape[0] == 4  # Check audio is 4 chanels
    assert audio.shape[1] == 48000  # Check audio duration in samples is as expected


def test_to_jams():

    # Note: original file  tsrimmed to 1 sec
    default_clipid = "foa_dev/split1_ir0_ov1_1"
    dataset = tau_sse_2019.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)
    jam = clip.to_jams()

    # Validate jam schema
    assert jam.validate()


def test_load_spatialevents():
    dataset = tau_sse_2019.Dataset(TEST_DATA_HOME)
    clip = dataset.clip("foa_dev/split1_ir0_ov1_1")
    csv_path = clip.csv_path
    events_data = tau_sse_2019.load_spatialevents(csv_path)
    assert events_data.labels[0] == "cough"
    assert events_data.labels[-1] == "phone"
    assert (events_data.intervals[0] == [0.36645229108,1.33445229108]).all()
    assert (events_data.intervals[-1] == [5.34011283858,7.12411283858]).all()
    assert (events_data.locations[0] == [-10,-10,2]).all()


def test_validate_locations():
    tau_sse_2019.validate_locations(None)

    with pytest.raises(ValueError):
        tau_sse_2019.validate_locations(np.array([0, 2, 9]))

    with pytest.raises(ValueError):
        tau_sse_2019.validate_locations(np.array([[91,0,0],[0,0,0]]))

    with pytest.raises(ValueError):
        tau_sse_2019.validate_locations(np.array([[0,-181,0],[0,0,0]]))

    with pytest.raises(ValueError):
        tau_sse_2019.validate_locations(np.array([[0,0,-1],[0,0,0]]))
