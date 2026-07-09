import numpy as np

from tests.test_utils import run_clip_tests

from soundata import annotations
from soundata.datasets import dcase23_task6a
import os

TEST_DATA_HOME = os.path.normpath("tests/resources/sound_datasets/dcase23_task6a/")


def test_clip():
    default_clipid = "test_0001"
    dataset = dcase23_task6a.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip(default_clipid)

    expected_attributes = {
        "audio_path": (
            os.path.join(
                os.path.normpath("tests/resources/sound_datasets/dcase23_task6a/"),
                "test/test_0001.wav",
            )
        ),
        "clip_id": "test_0001",
    }

    expected_property_types = {
        "audio": tuple,
        "file_name": str,
        "start_end_samples": str,
        "manufacturer": str,
        "license": str,
    }

    run_clip_tests(clip, expected_attributes, expected_property_types)


def test_load_audio():
    default_clipid = "test_0001"
    dataset = dcase23_task6a.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip(default_clipid)
    audio_path = clip.audio_path
    audio, sr = dcase23_task6a.load_audio(audio_path)
    assert sr == 22050
    assert type(audio) is np.ndarray
    assert len(audio.shape) == 1  # check audio is loaded as stereo
    assert audio.shape[0] == 44100  # Check audio duration is as expected


def test_load_metadata():
    default_clipid = "test_0001"
    dataset = dcase23_task6a.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip(default_clipid)
    assert clip.file_name == "test_0001"
    assert clip.manufacturer == "ceejay"
    assert clip.start_end_samples == "[3584, 1325506]"
    assert clip.license == "http://creativecommons.org/publicdomain/zero/1.0/"
