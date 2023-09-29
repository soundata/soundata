import os
import numpy as np

from tests.test_utils import run_clip_tests

from soundata import annotations
from soundata.datasets import dcase_birdVox20k
from tests.test_utils import DEFAULT_DATA_HOME


TEST_DATA_HOME = os.path.normpath("tests/resources/sound_datasets/dcase_birdVox20k")


def test_clip():
    default_clipid = "00053d90-e4b9-4045-a2f1-f39efc90cfa9"
    dataset = dcase_birdVox20k.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)

    expected_attributes = {
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/sound_datasets/dcase_birdVox20k/"),
            "00053d90-e4b9-4045-a2f1-f39efc90cfa9.wav",
        ),
        "clip_id": "00053d90-e4b9-4045-a2f1-f39efc90cfa9",
    }

    expected_property_types = {
        "item_id": str,
        "dataset_id": str,
        "has_bird": str,
        "audio": tuple,
    }
    run_clip_tests(clip, expected_attributes, expected_property_types)


def test_load_audio():
    dataset = dcase_birdVox20k.Dataset(TEST_DATA_HOME)
    clip = dataset.clip("00053d90-e4b9-4045-a2f1-f39efc90cfa9")
    audio_path = clip.audio_path
    audio, sr = dcase_birdVox20k.load_audio(audio_path)

    assert sr == 44100
    assert type(audio) is np.ndarray
    assert len(audio.shape) == 1  # check audio is loaded as mono
    assert audio.shape[0] == 441000  # Check audio duration in samples is as expected


def test_to_jams():
    default_clipid = "00053d90-e4b9-4045-a2f1-f39efc90cfa9"
    dataset = dcase_birdVox20k.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)
    jam = clip.to_jams()
    # Validate birdVox20k jam schema
    assert jam.validate()

    # validate metadata
    assert jam.file_metadata.duration == 10.0
    assert jam.sandbox.itemid == "00053d90-e4b9-4045-a2f1-f39efc90cfa9"
    assert jam.sandbox.datasetid == "BirdVox-DCASE-20k"
    assert jam.sandbox.hasbird == "1"
