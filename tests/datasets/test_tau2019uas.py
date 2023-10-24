import numpy as np

from tests.test_utils import run_clip_tests

from soundata import annotations
from soundata.datasets import tau2019uas
import os

TEST_DATA_HOME = os.path.normpath("tests/resources/sound_datasets/tau2019uas")


def test_clip():
    default_clipid = "development/airport-barcelona-0-0-a"
    dataset = tau2019uas.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)

    expected_attributes = {
        "audio_path": (
            os.path.join(
                os.path.normpath("tests/resources/sound_datasets/tau2019uas/"),
                "TAU-urban-acoustic-scenes-2019-development/audio/airport-barcelona-0-0-a.wav",
            )
        ),
        "clip_id": "development/airport-barcelona-0-0-a",
    }

    expected_property_types = {
        "split": str,
        "audio": tuple,
        "tags": annotations.Tags,
        "city": str,
        "identifier": str,
    }

    run_clip_tests(clip, expected_attributes, expected_property_types)


def test_load_audio():
    default_clipid = "development/airport-barcelona-0-0-a"
    dataset = tau2019uas.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)
    audio_path = clip.audio_path
    audio, sr = tau2019uas.load_audio(audio_path)
    assert sr == 48000
    assert type(audio) is np.ndarray
    assert len(audio.shape) == 2  # check audio is loaded as stereo
    assert audio.shape[1] == 48000  # Check audio duration is as expected


def test_load_tags():
    # Development dataset
    default_clipid = "development/airport-barcelona-0-0-a"
    dataset = tau2019uas.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)
    assert len(clip.tags.labels) == 1
    assert clip.tags.labels[0] == "airport"
    assert np.allclose([1.0], clip.tags.confidence)

    # Evaluation dataset
    eval_default_clipid = "evaluation/0"
    eval_clip = dataset.clip(eval_default_clipid)
    assert eval_clip.tags is None

    # Leadearboard dataset
    lead_default_clipid = "leaderboard/0"
    lead_clip = dataset.clip(lead_default_clipid)
    assert lead_clip.tags is None


def test_load_metadata():
    # Development dataset
    default_clipid = "development/airport-barcelona-0-0-a"
    dataset = tau2019uas.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)
    assert clip.split == "development.train"
    assert clip.identifier == "barcelona-0"
    assert clip.city == "barcelona"

    # Evaluation dataset
    eval_default_clipid = "evaluation/0"
    eval_clip = dataset.clip(eval_default_clipid)
    assert eval_clip.split == "evaluation"
    assert eval_clip.identifier is None
    assert eval_clip.city is None

    # Leaderboard dataset
    lead_default_clipid = "leaderboard/0"
    lead_clip = dataset.clip(lead_default_clipid)
    assert lead_clip.split == "leaderboard"
    assert lead_clip.identifier is None
    assert lead_clip.city is None


def test_to_jams():
    default_clipid = "development/airport-barcelona-0-0-a"
    dataset = tau2019uas.Dataset(TEST_DATA_HOME)
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
    assert jam.sandbox.split == "development.train"
    assert jam.sandbox.identifier == "barcelona-0"
    assert jam.sandbox.city == "barcelona"
    assert jam.sandbox.scene_label == "airport"
    assert jam.annotations[0].annotation_metadata.data_source == "soundata"
