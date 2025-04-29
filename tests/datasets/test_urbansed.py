import os
import numpy as np

from tests.test_utils import run_clip_tests

from soundata import annotations
from soundata.datasets import urbansed


TEST_DATA_HOME = os.path.normpath("tests/resources/sound_datasets/urbansed")


def test_clip():
    default_clipid = "soundscape_train_uniform1736"
    dataset = urbansed.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip(default_clipid)

    expected_attributes = {
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/sound_datasets/urbansed/"),
            "audio/train/soundscape_train_uniform1736.wav",
        ),
        "jams_path": os.path.join(
            os.path.normpath("tests/resources/sound_datasets/urbansed/"),
            "annotations/train/soundscape_train_uniform1736.jams",
        ),
        "txt_path": os.path.join(
            os.path.normpath("tests/resources/sound_datasets/urbansed/"),
            "annotations/train/soundscape_train_uniform1736.txt",
        ),
        "clip_id": "soundscape_train_uniform1736",
    }

    expected_property_types = {
        "split": str,
        "audio": tuple,
        "events": annotations.Events,
    }

    run_clip_tests(clip, expected_attributes, expected_property_types)


def test_load_audio():
    dataset = urbansed.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip("soundscape_train_uniform1736")
    audio_path = clip.audio_path
    audio, sr = urbansed.load_audio(audio_path)
    assert sr == 44100
    assert type(audio) is np.ndarray
    assert len(audio.shape) == 1  # check audio is loaded as mono
    assert audio.shape[0] == 44100  # Check audio duration in samples is as expected
