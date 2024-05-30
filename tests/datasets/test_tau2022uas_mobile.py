import numpy as np

from tests.test_utils import run_clip_tests

from soundata import annotations
from soundata.datasets import tau2022uas_mobile
import os

TEST_DATA_HOME = os.path.normpath("tests/resources/sound_datasets/tau2022uas_mobile")


def test_clip():
    default_clipid = "airport-lisbon-1000-40000-0-a"
    dataset = tau2022uas_mobile.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip(default_clipid)

    expected_attributes = {
        "audio_path": (
            os.path.join(
                os.path.normpath("tests/resources/sound_datasets/tau2022uas_mobile/"),
                "TAU-urban-acoustic-scenes-2022-mobile-development/audio/airport-lisbon-1000-40000-0-a.wav",
            )
        ),
        "clip_id": "airport-lisbon-1000-40000-0-a",
    }

    expected_property_types = {
        "split": str,
        "audio": tuple,
        "tags": annotations.Tags,
        "city": str,
        "source_label": str,
        "identifier": str,
    }

    run_clip_tests(clip, expected_attributes, expected_property_types)


def test_load_audio():
    default_clipid = "airport-lisbon-1000-40000-0-a"
    dataset = tau2022uas_mobile.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip(default_clipid)
    audio_path = clip.audio_path
    audio, sr = tau2022uas_mobile.load_audio(audio_path)
    assert sr == 44100
    assert type(audio) is np.ndarray
    assert len(audio) == 44100  # Check audio duration is as expected


def test_load_tags():
    default_clipid = "airport-lisbon-1000-40000-0-a"
    dataset = tau2022uas_mobile.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip(default_clipid)
    assert len(clip.tags.labels) == 1
    assert clip.tags.labels[0] == "airport"
    assert np.allclose([1.0], clip.tags.confidence)

    # Evaluation dataset
    eval_default_clipid = "0"
    eval_clip = dataset.clip(eval_default_clipid)
    assert eval_clip.tags is None


def test_load_metadata():
    default_clipid = "airport-lisbon-1000-40000-0-a"
    dataset = tau2022uas_mobile.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip(default_clipid)
    assert clip.split == "2022.development.train"
    assert clip.identifier == "lisbon-1000"
    assert clip.city == "lisbon"
    assert clip.source_label == "a"

    # Evaluation dataset
    eval_default_clipid = "0"
    eval_clip = dataset.clip(eval_default_clipid)
    assert eval_clip.split == "2023.evaluation"
    assert eval_clip.identifier is None
    assert eval_clip.city is None
    assert eval_clip.source_label is None


def test_to_jams():
    default_clipid = "airport-lisbon-1000-40000-0-a"
    dataset = tau2022uas_mobile.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip(default_clipid)
    jam = clip.to_jams()

    assert jam.validate()

    # Validate Tags
    tags = jam.search(namespace="tag_open")[0]["data"]
    assert len(tags) == 1
    assert tags[0].time == 0
    assert tags[0].duration == 1.0
    assert tags[0].value == "airport"
    assert tags[0].confidence == 1

    # validate metadata
    assert jam.file_metadata.duration == 1.0
    assert jam.sandbox.split == "2022.development.train"
    assert jam.sandbox.source_label == "a"
    assert jam.sandbox.identifier == "lisbon-1000"
    assert jam.sandbox.city == "lisbon"
    assert jam.sandbox.scene_label == "airport"
    assert jam.annotations[0].annotation_metadata.data_source == "soundata"
