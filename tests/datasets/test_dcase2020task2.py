import numpy as np
import pytest

from tests.test_utils import run_clip_tests

from soundata import annotations
from soundata.datasets import dcase2020task2


TEST_DATA_HOME = "tests/resources/sound_datasets/dcase2020task2"


def test_clip():
    default_clipid = "development.train/fan/normal_id_00_00000000"
    dataset = dcase2020task2.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)

    expected_attributes = {
        "audio_path": (
            "tests/resources/sound_datasets/dcase2020task2/development/fan/train/normal_id_00_00000000.wav"
        ),
        "clip_id": "development.train/fan/normal_id_00_00000000",
    }

    expected_property_types = {
        "split": str,
        "audio": tuple,
        "tags": annotations.Tags,
        "machine_type": str,
        "machine_id": str,
    }

    run_clip_tests(clip, expected_attributes, expected_property_types)


def test_load_audio():
    dataset = dcase2020task2.Dataset(TEST_DATA_HOME)
    clip = dataset.clip("development.train/fan/normal_id_00_00000000")
    audio_path = clip.audio_path
    audio, sr = dcase2020task2.load_audio(audio_path)
    assert sr == 16000
    assert type(audio) is np.ndarray
    assert len(audio) == 16000  # Check audio duration is as expected


def test_load_tags():
    dataset = dcase2020task2.Dataset(TEST_DATA_HOME)
    clip = dataset.clip("development.train/fan/normal_id_00_00000000")

    assert clip.tags.labels == ["normal"]
    assert clip.tags.confidence == 1

    clip = dataset.clip("additional_training.train/pump/normal_id_01_00000000")
    with pytest.raises(FileNotFoundError):
        clip.tags

    clip_eval = dataset.clip("evaluation.test/fan/id_01_00000000")
    assert clip_eval.tags == None


def test_metadata():
    dataset = dcase2020task2.Dataset(TEST_DATA_HOME)
    clip = dataset.clip("development.train/fan/normal_id_00_00000000")

    assert clip.split == "development.train"
    assert clip.machine_type == "fan"
    assert clip.machine_id == "00_00000000"


def test_to_jams():

    default_clipid = "development.train/fan/normal_id_00_00000000"
    dataset = dcase2020task2.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)
    jam = clip.to_jams()

    assert jam.validate()

    # Validate Tags
    tags = jam.search(namespace="tag_open")[0]["data"]

    assert len(tags) == 1
    assert tags[0].time == 0
    assert tags[0].duration == 1.0
    assert tags[0].value == "normal"
    assert tags[0].confidence == 1

    # validate Metadata
    assert jam.file_metadata.duration == 1.0
    assert jam.sandbox.split == "development.train"
    assert jam.sandbox.machine_type == "fan"
    assert jam.sandbox.machine_id == "00_00000000"
