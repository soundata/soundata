import os
import numpy as np

from tests.test_utils import run_clip_tests

from soundata import annotations
from soundata.datasets import urbansound8k
from tests.test_utils import DEFAULT_DATA_HOME


TEST_DATA_HOME = "tests/resources/sound_datasets/urbansound8k"


def test_clip():
    default_clipid = "135776-2-0-49"
    dataset = urbansound8k.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)

    expected_attributes = {
        "audio_path": "tests/resources/sound_datasets/urbansound8k/audio/fold1/135776-2-0-49.wav",
        "clip_id": "135776-2-0-49",
    }

    expected_property_types = {
        "slice_file_name": str,
        "freesound_id": str,
        "freesound_start_time": float,
        "freesound_end_time": float,
        "salience": int,
        "fold": int,
        "class_id": int,
        "class_label": str,
        "audio": tuple,
        "tags": annotations.Tags,
    }

    run_clip_tests(clip, expected_attributes, expected_property_types)


def test_load_audio():
    dataset = urbansound8k.Dataset(TEST_DATA_HOME)
    clip = dataset.clip("135776-2-0-49")
    audio_path = clip.audio_path
    audio, sr = urbansound8k.load_audio(audio_path)
    assert sr == 44100
    assert type(audio) is np.ndarray
    assert len(audio.shape) == 1  # check audio is loaded as mono
    assert audio.shape[0] == 176400  # Check audio duration in sampels is as expected
