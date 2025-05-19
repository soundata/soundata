import numpy as np
import pytest

from tests.test_utils import run_clip_tests

from soundata import annotations
from soundata.datasets import fsdnoisy18k
import os

TEST_DATA_HOME = os.path.normpath("tests/resources/sound_datasets/fsdnoisy18k")


def test_clip():
    default_clipid = "17"
    dataset = fsdnoisy18k.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip(default_clipid)
    expected_attributes = {
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/sound_datasets/fsdnoisy18k/"),
            "FSDnoisy18k.audio_train/17.wav",
        ),
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
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/sound_datasets/fsdnoisy18k/"),
            "FSDnoisy18k.audio_test/564.wav",
        ),
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
    dataset = fsdnoisy18k.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip("17")
    audio_path = clip.audio_path
    audio, sr = fsdnoisy18k.load_audio(audio_path)
    assert sr == 44100
    assert type(audio) is np.ndarray
    assert len(audio.shape) == 1  # check audio is loaded as mono
    assert len(audio) == 47786


def test_tag():
    default_clipid = "17"
    dataset = fsdnoisy18k.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip(default_clipid)
    tag = clip.tags
    assert tag.labels == ["Walk_or_footsteps"]
    assert tag.confidence == [1.0]


def test_metadata():
    # Testing metadata from a training clip
    default_clipid = "17"
    dataset = fsdnoisy18k.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip(default_clipid)
    clip_metadata = clip._metadata()[default_clipid]

    assert clip_metadata["tag"] == "Walk_or_footsteps"
    assert clip_metadata["aso_id"] == "/m/07pbtc8"
    assert clip_metadata.get("manually_verified") == 1
    assert clip_metadata.get("noisy_small") == 0
    assert clip_metadata["split"] == "train"

    # Testing metadata from an evaluation clip
    default_clipid = "564"
    dataset = fsdnoisy18k.Dataset(TEST_DATA_HOME, version="test")
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
