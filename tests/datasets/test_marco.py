import numpy as np

from tests.test_utils import run_clip_tests

from soundata import annotations
from soundata.datasets import marco


TEST_DATA_HOME = "tests/resources/sound_datasets/marco"


def test_clip():
    default_clipid = "impulse_response/+90deg_011_OCT3D_2_FR"
    dataset = marco.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)

    expected_attributes = {
        "audio_path": (
            "tests/resources/sound_datasets/marco/3D-MARCo Impulse Responses/01_Speaker_+90deg_3m/+90deg_011_OCT3D_2_FR.wav"
        ),
        "clip_id": "impulse_response/+90deg_011_OCT3D_2_FR",
        "source_label": "impulse_response",
        "source_angle": "+90deg",
        "microphone_info": ["OCT3D", "2", "FR"],
    }

    expected_property_types = {"audio": tuple}

    run_clip_tests(clip, expected_attributes, expected_property_types)


def test_load_audio():
    default_clipid = "impulse_response/+90deg_011_OCT3D_2_FR"
    dataset = marco.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)
    audio_path = clip.audio_path
    audio, sr = marco.load_audio(audio_path)
    assert sr == 48000
    assert type(audio) is np.ndarray
    assert len(audio.shape) == 1  # check audio is loaded correctly
    assert audio.shape[0] == 48000 * 1.0  # Check audio duration is as expected


def test_load_metadata():
    # dataset
    default_clipid = "impulse_response/+90deg_011_OCT3D_2_FR"
    dataset = marco.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)
    assert clip.microphone_info == ["OCT3D", "2", "FR"]


def test_to_jams():
    default_clipid = "impulse_response/+90deg_011_OCT3D_2_FR"
    dataset = marco.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)
    jam = clip.to_jams()

    assert jam.validate()

    # validate metadata
    assert jam.file_metadata.duration == 1.0
    assert jam.sandbox.microphone_info == ["OCT3D", "2", "FR"]
    print(jam.annotations)
