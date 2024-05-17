import os
import numpy as np
import pytest

from tests.test_utils import run_clip_tests

from soundata import annotations
from soundata.datasets import dcase23_task2


TEST_DATA_HOME = os.path.normpath("tests/resources/sound_datasets/dcase23_task2")


def test_clip():
    default_clipid = "section_00_source_train_normal_0705_m-n_X"
    dataset = dcase23_task2.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip(default_clipid)
    expected_attributes = {
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/sound_datasets/dcase23_task2/"),
            "7882613/fan/train/section_00_source_train_normal_0705_m-n_X.wav",
        ),
        "clip_id": "section_00_source_train_normal_0705_m-n_X",
    }

    expected_property_types = {
        "file_name": str,
        "d1p": str,
        "d1v": str,
        "audio": tuple,
    }

    run_clip_tests(clip, expected_attributes, expected_property_types)


def test_load_audio():
    dataset = dcase23_task2.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip("section_00_source_train_normal_0705_m-n_X")
    audio_path = clip.audio_path
    audio, sr = dcase23_task2.load_audio(audio_path)
    assert sr == 44100
    assert type(audio) is np.ndarray
    assert len(audio.shape) == 1  # check audio is loaded as mono
    assert len(audio) == 441000


def test_to_jams():
    default_clipid = "section_00_source_train_normal_0705_m-n_X"
    dataset = dcase23_task2.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip(default_clipid)
    jam = clip.to_jams()
    # Validate dcase23_task2 jam schema
    assert jam.validate()
    # validate metadata
    assert jam.file_metadata.duration == 10.0
    assert (
        jam.sandbox.file_name
        == "fan/train/section_00_source_train_normal_0705_m-n_X.wav"
    )
    assert jam.sandbox.d1p == "m-n"
    assert jam.sandbox.d1v == "X"


def test_metadata_file_not_found():
    # Create a temporary dataset instance with an altered path to simulate missing files
    altered_test_data_home = os.path.join(TEST_DATA_HOME, "non_existent_directory")
    dataset = dcase23_task2.Dataset(altered_test_data_home, version="test")

    # Expect a FileNotFoundError
    with pytest.raises(FileNotFoundError):
        _ = dataset._metadata
