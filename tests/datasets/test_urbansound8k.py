import os
import numpy as np

from tests.test_utils import run_clip_tests

from soundata import annotations
from soundata.datasets import urbansound8k
from tests.test_utils import DEFAULT_DATA_HOME


TEST_DATA_HOME = "tests/resources/sound_datasets/urbansound8k"


def test_clip():
    default_clipid = "135776-2-0-49"
    dataset = urbansound8k.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)

    expected_attributes = {
        "audio_path": "tests/resources/sound_datasets/urbansound8k/audio/fold1/135776-2-0-49.wav",
        "clip_id": "135776-2-0-49",
    }

    expected_property_types = {
        "slice_file_name": str,
        "freesound_id": str,
        "freesound_start_time": float,
        "freesound_end_time": float,
        "salience": int,
        "fold": int,
        "class_id": int,
        "class_label": str,
        "audio": tuple,
        "tags": annotations.Tags,
    }

    run_clip_tests(clip, expected_attributes, expected_property_types)


def test_load_audio():
    dataset = urbansound8k.Dataset(TEST_DATA_HOME)
    clip = dataset.clip("135776-2-0-49")
    audio_path = clip.audio_path
    audio, sr = urbansound8k.load_audio(audio_path)
    assert sr == 44100
    assert type(audio) is np.ndarray
    assert len(audio.shape) == 1  # check audio is loaded as mono
    assert audio.shape[0] == 44100  # Check audio duration in sampels is as expected


def test_to_jams():

    # Note: original file is 4 sec, but for testing we've trimmed it to 1 sec
    default_clipid = "135776-2-0-49"
    dataset = urbansound8k.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)
    jam = clip.to_jams()

    # Validate urbansound8k jam schema
    assert jam.validate()

    # Validate Tags
    tags = jam.search(namespace="tag_open")[0]["data"]
    assert len(tags) == 1
    assert tags[0].time == 0
    assert tags[0].duration == 1.0
    assert tags[0].value == "children_playing"
    assert tags[0].confidence == 1

    # validate metadata
    assert jam.file_metadata.duration == 1.0
    assert jam.sandbox.fold == 1
    assert jam.sandbox.freesound_end_time == 28.5
    assert jam.sandbox.freesound_id == "135776"
    assert jam.sandbox.freesound_start_time == 24.5
    assert jam.sandbox.salience == 2
    assert jam.sandbox.slice_file_name == "135776-2-0-49.wav"
    assert jam.annotations[0].annotation_metadata.data_source == "soundata"
