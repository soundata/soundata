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
    dataset = esc50.Dataset(TEST_DATA_HOME, version="default")
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
    dataset = esc50.Dataset(TEST_DATA_HOME, version="default")
    clip = dataset.clip("1-104089-A-22")
    audio_path = clip.audio_path
    audio, sr = esc50.load_audio(audio_path)
    assert sr == 44100
    assert type(audio) is np.ndarray
    assert len(audio.shape) == 1  # check audio is loaded as mono
    assert audio.shape[0] == 44100  # Check audio duration in sampels is as expected


def test_to_jams():
    # Note: original file is 5 sec, but for testing we've trimmed it to 1 sec
    default_clipid = "1-104089-A-22"
    dataset = esc50.Dataset(TEST_DATA_HOME, version="default")
    clip = dataset.clip(default_clipid)
    jam = clip.to_jams()

    # Validate esc50 jam schema
    assert jam.validate()

    # Validate Tags
    tags = jam.search(namespace="tag_open")[0]["data"]
    assert len(tags) == 1
    assert tags[0].time == 0
    assert tags[0].duration == 1.0
    assert tags[0].value == "clapping"
    assert tags[0].confidence == 1

    # validate metadata
    assert jam.file_metadata.duration == 1.0
    assert jam.sandbox.filename == "1-104089-A-22.wav"
    assert jam.sandbox.fold == 1
    assert jam.sandbox.target == 22
    assert jam.sandbox.category == "clapping"
    assert jam.sandbox.esc10 == False
    assert jam.sandbox.src_file == "104089"
    assert jam.annotations[0].annotation_metadata.data_source == "soundata"
