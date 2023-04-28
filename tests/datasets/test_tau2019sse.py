import os
import numpy as np
import pytest

from tests.test_utils import run_clip_tests

from soundata import annotations
from soundata.datasets import tau2019sse


TEST_DATA_HOME = os.path.normpath("tests/resources/sound_datasets/tau2019sse")


def test_clip():
    default_clipid = "foa_dev/split1_ir0_ov1_1"
    dataset = tau2019sse.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)

    expected_attributes = {
        "clip_id": "foa_dev/split1_ir0_ov1_1",
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/sound_datasets/tau2019sse/"),
            "foa_dev/split1_ir0_ov1_1.wav",
        ),
        "csv_path": os.path.join(
            os.path.normpath("tests/resources/sound_datasets/tau2019sse/"),
            "metadata_dev/split1_ir0_ov1_1.csv",
        ),
        "format": "foa",
        "set": "dev",
    }

    expected_property_types = {
        "audio": tuple,
        "events": annotations.Events,
    }

    run_clip_tests(clip, expected_attributes, expected_property_types)


def test_load_audio():
    dataset = tau2019sse.Dataset(TEST_DATA_HOME)
    clip = dataset.clip("foa_dev/split1_ir0_ov1_1")
    audio_path = clip.audio_path
    audio, sr = tau2019sse.load_audio(audio_path)
    assert sr == 48000
    assert type(audio) is np.ndarray
    assert len(audio.shape) == 2  # check audio is multichannel
    assert audio.shape[0] == 4  # Check audio is 4 chanels
    assert audio.shape[1] == 48000  # Check audio duration in samples is as expected


def test_to_jams():
    # Note: original file  tsrimmed to 1 sec
    default_clipid = "foa_dev/split1_ir0_ov1_1"
    dataset = tau2019sse.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)
    jam = clip.to_jams()

    # Validate jam schema
    assert jam.validate()


def test_load_events():
    dataset = tau2019sse.Dataset(TEST_DATA_HOME)
    clip = dataset.clip("foa_dev/split1_ir0_ov1_1")
    csv_path = clip.csv_path
    events_data = tau2019sse.load_events(csv_path)
    assert events_data.labels[0] == "cough"
    assert events_data.labels[-1] == "phone"
    assert (events_data.intervals[0] == [0.36645229108, 1.33445229108]).all()
    assert (events_data.intervals[-1] == [5.34011283858, 7.12411283858]).all()
    assert events_data.elevation[0] == -10
    assert events_data.azimuth[0] == -10
    assert events_data.distance[0] == 2
