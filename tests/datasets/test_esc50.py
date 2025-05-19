import os
import numpy as np

from tests.test_utils import run_clip_tests

from soundata import annotations
from soundata.datasets import esc50
from tests.test_utils import DEFAULT_DATA_HOME
import os

TEST_DATA_HOME = os.path.normpath("tests/resources/sound_datasets/esc50")


def test_clip():
    default_clipid = "1-104089-A-22"
    dataset = esc50.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip(default_clipid)

    expected_attributes = {
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/sound_datasets/esc50/"),
            "audio/1-104089-A-22.wav",
        ),
        "clip_id": "1-104089-A-22",
    }

    expected_property_types = {
        "filename": str,
        "fold": int,
        "target": int,
        "category": str,
        "esc10": bool,
        "src_file": str,
        "take": str,
        "audio": tuple,
        "tags": annotations.Tags,
    }

    run_clip_tests(clip, expected_attributes, expected_property_types)


def test_load_audio():
    dataset = esc50.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip("1-104089-A-22")
    audio_path = clip.audio_path
    audio, sr = esc50.load_audio(audio_path)
    assert sr == 44100
    assert type(audio) is np.ndarray
    assert len(audio.shape) == 1  # check audio is loaded as mono
    assert audio.shape[0] == 44100  # Check audio duration in sampels is as expected
