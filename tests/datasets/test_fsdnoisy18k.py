import numpy as np
import pytest

from tests.test_utils import run_clip_tests

from soundata import annotations
from soundata.datasets import fsdnoisy18k

TEST_DATA_HOME = "tests/resources/sound_datasets/fsdnoisy18k"


def test_clip():
    default_clipid = "17"
    dataset = fsdnoisy18k.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)
    expected_attributes = {
        "audio_path": "tests/resources/sound_datasets/fsdnoisy18k/FSDnoisy18k.audio_train/17.wav",
        "clip_id": "17",
    }

    expected_property_types = {
        "audio": tuple,
        "tags": annotations.Tags,
        "split": str,
        "aso_id": str,
        "manually_verified": int,
        "noisy_small": int,
    }

    run_clip_tests(clip, expected_attributes, expected_property_types)

    default_clipid_test = "564"
    clip_test = dataset.clip(default_clipid_test)
    expected_attributes_test = {
        "audio_path": "tests/resources/sound_datasets/fsdnoisy18k/FSDnoisy18k.audio_test/564.wav",
        "clip_id": "564",
    }

    expected_property_types_test = {
        "audio": tuple,
        "tags": annotations.Tags,
        "split": str,
        "aso_id": str,
        "manually_verified": type(None),
        "noisy_small": type(None),
    }

    run_clip_tests(clip_test, expected_attributes_test, expected_property_types_test)


def test_load_audio():
    dataset = fsdnoisy18k.Dataset(TEST_DATA_HOME)
    clip = dataset.clip("17")
    audio_path = clip.audio_path
    audio, sr = fsdnoisy18k.load_audio(audio_path)
    assert sr == 44100
    assert type(audio) is np.ndarray
    assert len(audio.shape) == 1  # check audio is loaded as mono
    assert len(audio) == 47786


def test_to_jams():
    default_clipid = "17"
    dataset = fsdnoisy18k.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)
    jam = clip.to_jams()

    # Validate fsd50k jam schema
    assert jam.validate()

    # Validate Tags
    tags = jam.search(namespace="tag_open")[0]["data"]
    assert len(tags) == 1
    assert [tag.time for tag in tags] == [0.0]
    assert [tag.duration for tag in tags] == [1.0835827664399094]
    assert [tag.value for tag in tags] == ["Walk_or_footsteps"]
    assert [tag.confidence for tag in tags] == [1.0]

    # validate metadata
    assert jam.file_metadata.duration == 1.0835827664399094
    assert jam.sandbox.aso_id == "/m/07pbtc8"
    assert jam.sandbox.manually_verified == 1
    assert jam.sandbox.noisy_small == 0
    assert jam.sandbox.split == "train"
    assert jam.annotations[0].annotation_metadata.data_source == "soundata"


def test_tag():
    default_clipid = "17"
    dataset = fsdnoisy18k.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)
    tag = clip.tags
    assert tag.labels == ["Walk_or_footsteps"]
    assert tag.confidence == [1.0]


def test_metadata():
    # Testing metadata from a training clip
    default_clipid = "17"
    dataset = fsdnoisy18k.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)
    clip_metadata = clip._metadata()[default_clipid]

    assert clip_metadata["tag"] == "Walk_or_footsteps"
    assert clip_metadata["aso_id"] == "/m/07pbtc8"
    assert clip_metadata.get("manually_verified") == 1
    assert clip_metadata.get("noisy_small") == 0
    assert clip_metadata["split"] == "train"

    # Testing metadata from an evaluation clip
    default_clipid = "564"
    dataset = fsdnoisy18k.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)
    clip_metadata = clip._metadata()[default_clipid]

    assert clip_metadata["tag"] == "Walk_or_footsteps"
    assert clip_metadata["aso_id"] == "/m/07pbtc8"
    assert clip_metadata.get("manually_verified") is None
    assert clip_metadata.get("noisy_small") is None
    assert clip_metadata["split"] == "test"

    # Test erroneous filepath to train metadata
    with pytest.raises(FileNotFoundError):
        dataset = fsdnoisy18k.Dataset("a/fake/path/to/the/dataset")
        clip = dataset.clip(default_clipid)
        clip_metadata = clip._metadata()[default_clipid]

    # Test erroneous filepath to test metadata
    with pytest.raises(FileNotFoundError):
        dataset = fsdnoisy18k.Dataset("tests/resources/download")
        clip = dataset.clip(default_clipid)
        clip_metadata = clip._metadata()[default_clipid]
