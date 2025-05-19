import numpy as np

from tests.test_utils import run_clip_tests

from soundata import annotations
from soundata.datasets import tut2017se
import os

TEST_DATA_HOME = os.path.normpath("tests/resources/sound_datasets/tut2017se")


def test_clip():
    default_clipid = "a001"
    dataset = tut2017se.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip(default_clipid)

    expected_attributes = {
        "audio_path": (
            os.path.join(
                os.path.normpath("tests/resources/sound_datasets/tut2017se/"),
                "TUT-sound-events-2017-development/audio/street/a001.wav",
            )
        ),
        "annotations_path": (
            os.path.join(
                os.path.normpath("tests/resources/sound_datasets/tut2017se/"),
                "TUT-sound-events-2017-development/meta/street/a001.ann",
            )
        ),
        "non_verified_annotations_path": (
            os.path.join(
                os.path.normpath("tests/resources/sound_datasets/tut2017se/"),
                "TUT-sound-events-2017-development/non_verified/meta/street/a001.ann",
            )
        ),
        "clip_id": "a001",
    }

    expected_property_types = {
        "split": str,
        "audio": tuple,
        "events": annotations.Events,
        "non_verified_events": annotations.Events,
    }

    run_clip_tests(clip, expected_attributes, expected_property_types)


def test_load_audio():
    dataset = tut2017se.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip("a001")
    audio_path = clip.audio_path
    audio, sr = tut2017se.load_audio(audio_path)
    assert sr == 44100
    assert type(audio) is np.ndarray
    assert len(audio.shape) == 2  # check audio is loaded as stereo
    assert audio.shape[1] == 44100  # Check audio duration is as expected


def test_load_events():
    dataset = tut2017se.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip("a001")
    annotations_path = clip.annotations_path
    annotations = tut2017se.load_events(annotations_path)

    confidence = [1.0] * 3
    intervals = [[1.58921, 2.38382], [3.500767, 4.156693], [4.156693, 14.00307]]
    labels = ["people walking", "people walking", "car"]
    assert np.allclose(confidence, annotations.confidence)
    assert np.allclose(intervals, annotations.intervals)

    for j in range(3):
        assert labels[j] == annotations.labels[j]
