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
        "tags": annotations.Tags
    }

    run_clip_tests(clip, expected_attributes, expected_property_types)
