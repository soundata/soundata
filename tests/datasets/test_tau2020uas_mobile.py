import numpy as np

from tests.test_utils import run_clip_tests

from soundata import annotations
from soundata.datasets import tau2020uas_mobile


TEST_DATA_HOME = "tests/resources/sound_datasets/tau2020uas_mobile"


def test_clip():
    default_clipid = "airport-barcelona-0-0-a"
    dataset = tau2020uas_mobile.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)

    expected_attributes = {
        "audio_path": (
            "tests/resources/sound_datasets/tau2020uas_mobile/TAU-urban-acoustic-scenes-2020-mobile-development/audio/airport-barcelona-0-0-a.wav"
        ),
        "clip_id": "airport-barcelona-0-0-a",
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
    default_clipid = "airport-barcelona-0-0-a"
    dataset = tau2020uas_mobile.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)
    audio_path = clip.audio_path
    audio, sr = tau2020uas_mobile.load_audio(audio_path)
    assert sr == 44100
    assert type(audio) is np.ndarray
    assert len(audio) == 44100  # Check audio duration is as expected


def test_load_tags():
    default_clipid = "airport-barcelona-0-0-a"
    dataset = tau2020uas_mobile.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)
    assert len(clip.tags.labels) == 1
    assert clip.tags.labels[0] == "airport"
    assert np.allclose([1.0], clip.tags.confidence)


def test_load_metadata():
    default_clipid = "airport-barcelona-0-0-a"
    dataset = tau2020uas_mobile.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)
    assert clip.split == "development.train"
    assert clip.identifier == "barcelona-0"
    assert clip.city == "barcelona"
    assert clip.source_label == "a"


# def test_to_jams():

#     # Note: original file is 4 sec, but for testing we've trimmed it to 1 sec
#     default_clipid = "a001"
#     dataset = tut_sound_events_2017.Dataset(TEST_DATA_HOME)
#     clip = dataset.clip(default_clipid)
#     jam = clip.to_jams()

#     # Validate urbansound8k jam schema
#     assert jam.validate()

# # Validate Events
