import os

import numpy as np
from soundata import annotations
from soundata.datasets import singapura
from tests.test_utils import run_clip_tests
import os

TEST_DATA_HOME = os.path.normpath("tests/resources/sound_datasets/singapura")


def test_clip():
    default_clipid = "[b827ebf3744c][2020-08-19T22-46-04Z][manual][---][4edbade2d41d5f80e324ee4f10d401c0][]-135"
    dataset = singapura.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)

    expected_attributes = {
        "audio_path": (
            os.path.join(
                TEST_DATA_HOME,
                "labelled/",
                "2020-08-19/",
                "[b827ebf3744c][2020-08-19T22-46-04Z][manual][---][4edbade2d41d5f80e324ee4f10d401c0][]-135.flac",
            )
        ),
        "annotation_path": os.path.join(
            TEST_DATA_HOME,
            "labels_public/",
            "[b827ebf3744c][2020-08-19T22-46-04Z][manual][---][4edbade2d41d5f80e324ee4f10d401c0][]-135.csv",
        ),
        "clip_id": "[b827ebf3744c][2020-08-19T22-46-04Z][manual][---][4edbade2d41d5f80e324ee4f10d401c0][]-135",
        "sensor_id": "b827ebf3744c",
        "town": "West 2",
        "timestamp": np.datetime64("2020-08-20T06:46:04"),
        "dotw": 4,
    }

    expected_property_types = {
        "audio": np.ndarray,
        "events": annotations.MultiAnnotator,
        "audio_path": str,
        "annotation_path": str,
        "clip_id": str,
        "sensor_id": str,
        "town": str,
        "timestamp": np.datetime64,
        "dotw": int,
    }

    run_clip_tests(clip, expected_attributes, expected_property_types)


def test_load_audio():
    default_clipid = "[b827ebf3744c][2020-08-19T22-46-04Z][manual][---][4edbade2d41d5f80e324ee4f10d401c0][]-135"
    dataset = singapura.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)
    audio_path = clip.audio_path
    audio = singapura.load_audio(audio_path)
    assert type(audio) is np.ndarray
    assert len(audio.shape) == 1  # check audio is loaded correctly
    assert audio.shape[0] == 44100  # Check audio duration is as expected


def test_to_jams():
    default_clipid = "[b827ebf3744c][2020-08-19T22-46-04Z][manual][---][4edbade2d41d5f80e324ee4f10d401c0][]-135"
    dataset = singapura.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)
    jam = clip.to_jams()

    assert jam.validate()


def test_metadata():
    dataset = singapura.Dataset(TEST_DATA_HOME)
    metadata = dataset._metadata
    default_clipid = "[b827ebf3744c][2020-08-19T22-46-04Z][manual][---][4edbade2d41d5f80e324ee4f10d401c0][]-135"

    assert metadata[default_clipid] == {
        "sensor_id": "b827ebf3744c",
        "year": 2020,
        "month": 8,
        "date": 20,
        "day": 4,
        "hour": 6,
        "minute": 46,
        "second": 4,
        "timezone": "SGT",
        "town": "West 2",
    }
