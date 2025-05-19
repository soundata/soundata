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
