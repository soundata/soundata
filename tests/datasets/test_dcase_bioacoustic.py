import os
import numpy as np

from tests.test_utils import run_clip_tests

from soundata import annotations
from soundata.datasets import dcase_bioacoustic


TEST_DATA_HOME = os.path.normpath("tests/resources/sound_datasets/dcase_bioacoustic")


def test_clip():
    default_clipid = "2015-09-04_08-04-59_unit03"
    dataset = dcase_bioacoustic.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip(default_clipid)

    expected_attributes = {
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/sound_datasets/dcase_bioacoustic"),
            "Development_Set/Training_Set/BV/2015-09-04_08-04-59_unit03.wav",
        ),
        "csv_path": os.path.join(
            os.path.normpath("tests/resources/sound_datasets/dcase_bioacoustic"),
            "Development_Set/Training_Set/BV/2015-09-04_08-04-59_unit03.csv",
        ),
        "clip_id": "2015-09-04_08-04-59_unit03",
    }

    expected_property_types = {
        "split": str,
        "subdataset": str,
        "audio": tuple,
        "events": annotations.Events,
        "events_classes": list,
        "POSevents": annotations.Events,
    }

    run_clip_tests(clip, expected_attributes, expected_property_types)


def test_load_audio():
    dataset = dcase_bioacoustic.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip("2015-09-04_08-04-59_unit03")
    audio_path = clip.audio_path
    audio, sr = dcase_bioacoustic.load_audio(audio_path)
    assert sr == 24000
    assert type(audio) is np.ndarray
    assert len(audio.shape) == 1  # check audio is loaded as mono
    assert audio.shape[0] == 24000  # Check audio duration in samples is as expected


def test_to_jams():
    # Note: for testing we've trimmed the original file to 1 sec
    default_clipid = "2015-09-04_08-04-59_unit03"
    dataset = dcase_bioacoustic.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip(default_clipid)
    jam = clip.to_jams()

    # Validate dcase_bioacoustic jam schema
    assert jam.validate()

    # Validate Events
    events = jam.annotations[0].data
    assert len(events) == 5

    event_onsets = [0.548, 1.028, 5.998, 8.126, 30.707]
    event_offsets = [0.698, 1.178, 6.148, 8.276, 30.857]
    event_labels = [
        "UNK,UNK,UNK,UNK,UNK,UNK,UNK",
        "UNK,UNK,UNK,UNK,UNK,UNK,UNK",
        "UNK,UNK,UNK,UNK,UNK,NEG,NEG",
        "UNK,UNK,UNK,UNK,UNK,NEG,NEG",
        "NEG,NEG,NEG,NEG,POS,NEG,NEG",
    ]

    for e, t, o, l in zip(events, event_onsets, event_offsets, event_labels):
        assert np.allclose(e.time, t)
        assert np.allclose(e.duration, o - t)
        assert np.allclose(e.time, t)

    # validate metadata
    assert jam.file_metadata.duration == 1.0
