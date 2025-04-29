import numpy as np

from tests.test_utils import run_clip_tests

from soundata import annotations
from soundata.datasets import dcase23_task6b
import os

TEST_DATA_HOME = os.path.normpath("tests/resources/sound_datasets/dcase23_task6b")


def test_clip():
    default_clipid = "development/1"
    dataset = dcase23_task6b.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip(default_clipid)

    expected_attributes = {
        "audio_path": (
            os.path.join(
                os.path.normpath("tests/resources/sound_datasets/dcase23_task6b/"),
                "development/1.wav",
            )
        ),
        "clip_id": "development/1",
    }

    expected_property_types = {
        "audio": tuple,
        "file_name": str,
        "keywords": str,
        "sound_id": str,
        "sound_link": str,
        "start_end_samples": str,
        "manufacturer": str,
        "license": str,
    }

    run_clip_tests(clip, expected_attributes, expected_property_types)


def test_load_audio():
    default_clipid = "development/1"
    dataset = dcase23_task6b.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip(default_clipid)
    audio_path = clip.audio_path
    audio, sr = dcase23_task6b.load_audio(audio_path)
    assert sr == 44100
    assert type(audio) is np.ndarray
    assert len(audio.shape) == 1  # check audio is loaded as stereo
    assert audio.shape[0] == 88200  # Check audio duration is as expected


def test_load_metadata():
    default_clipid = "development/1"
    dataset = dcase23_task6b.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip(default_clipid)
    assert clip.sound_id == "267105"
    assert clip.keywords == "thunder;weather;field-recording;rain;city"
    assert clip.sound_link == "https://freesound.org/people/Omega9/sounds/267105"
