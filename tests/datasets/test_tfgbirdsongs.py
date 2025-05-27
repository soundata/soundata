import os
import numpy as np
from soundata.datasets import tfgbirdsongs

from tests.test_utils import run_clip_tests

from soundata import annotations
from tests.test_utils import DEFAULT_DATA_HOME


TEST_DATA_HOME = os.path.normpath("tests/resources/sound_datasets/tfgbirdsongs")


def test_clip():
    default_clipid = "11713-2"
    dataset = tfgbirdsongs.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip(default_clipid)
    expected_attributes = {
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/sound_datasets/tfgbirdsongs/"),
            "wav/11713-2.wav",
        ),
        "clip_id": "11713-2",
    }

    expected_property_types = {
        "id": str,
        "genus": str,
        "species": str,
        "subspecies": str,
        "name": str,
        "recordist": str,
        "country": str,
        "location": str,
        "latitude": str,
        "longitude": str,
        "altitude": str,
        "sound_type": str,
        "source_url": str,
        "license": str,
        "time": str,
        "date": str,
        "remarks": str,
        "filename": str,
        "audio": tuple
    }
    run_clip_tests(clip, expected_attributes, expected_property_types)


def test_load_audio():
    dataset = tfgbirdsongs.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip("11713-2")
    audio_path = clip.audio_path
    audio, sr = tfgbirdsongs.load_audio(audio_path)

    assert sr == 44100
    assert type(audio) is np.ndarray
    assert len(audio.shape) == 1  # check audio is loaded as mono
    assert audio.shape[0] == 132300  # Check audio duration in samples is as expected

test_clip()