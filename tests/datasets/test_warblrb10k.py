import os
import numpy as np

from tests.test_utils import run_clip_tests

from soundata import annotations
from soundata.datasets import warblrb10k
from tests.test_utils import DEFAULT_DATA_HOME


TEST_DATA_HOME = os.path.normpath("tests/resources/sound_datasets/warblrb10k")


def test_clip():
    default_clipid = "759808e5-f824-401e-9058"
    dataset = warblrb10k.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip(default_clipid)

    expected_attributes = {
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/sound_datasets/warblrb10k/"),
            "wav/759808e5-f824-401e-9058.wav",
        ),
        "clip_id": "759808e5-f824-401e-9058",
    }

    expected_property_types = {
        "item_id": str,
        "has_bird": str,
        "audio": tuple,
    }
    run_clip_tests(clip, expected_attributes, expected_property_types)


def test_load_audio():
    dataset = warblrb10k.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip("759808e5-f824-401e-9058")
    audio_path = clip.audio_path
    audio, sr = warblrb10k.load_audio(audio_path)

    assert sr == 44100
    assert type(audio) is np.ndarray
    assert len(audio.shape) == 1  # check audio is loaded as mono
    assert audio.shape[0] == 444416  # Check audio duration in samples is as expected


def test_to_jams():
    default_clipid = "759808e5-f824-401e-9058"
    dataset = warblrb10k.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip(default_clipid)
    jam = clip.to_jams()
    # Validate warblrb10k jam schema
    assert jam.validate()

    # validate metadata
    assert round(jam.file_metadata.duration, 1) == 10.1
    assert jam.sandbox.itemid == "759808e5-f824-401e-9058"
    assert jam.sandbox.hasbird == "1"
