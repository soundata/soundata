import os
import numpy as np

from tests.test_utils import run_clip_tests

from soundata import annotations
from soundata.datasets import urbansed


TEST_DATA_HOME = "tests/resources/sound_datasets/urbansed"


def test_clip():
    default_clipid = "soundscape_train_uniform1736"
    dataset = urbansed.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)

    expected_attributes = {
        "audio_path": "tests/resources/sound_datasets/urbansed/audio/train/soundscape_train_uniform1736.wav",
        "jams_path": "tests/resources/sound_datasets/urbansed/annotations/train/soundscape_train_uniform1736.jams",
        "txt_path": "tests/resources/sound_datasets/urbansed/annotations/train/soundscape_train_uniform1736.txt",
        "clip_id": "soundscape_train_uniform1736",
    }

    expected_property_types = {
        "split": str,
        "audio": tuple,
        "events": annotations.Events,
    }

    run_clip_tests(clip, expected_attributes, expected_property_types)


def test_load_audio():
    dataset = urbansed.Dataset(TEST_DATA_HOME)
    clip = dataset.clip("soundscape_train_uniform1736")
    audio_path = clip.audio_path
    audio, sr = urbansed.load_audio(audio_path)
    assert sr == 44100
    assert type(audio) is np.ndarray
    assert len(audio.shape) == 1  # check audio is loaded as mono
    assert audio.shape[0] == 44100  # Check audio duration in samples is as expected


def test_to_jams():

    # Note: original file is 4 sec, but for testing we've trimmed it to 1 sec
    default_clipid = "soundscape_train_uniform1736"
    dataset = urbansed.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)
    jam = clip.to_jams()

    # Validate urbansound8k jam schema
    assert jam.validate()

    # Validate Events
    events = jam.search(namespace="scaper")[0]["data"]
    assert len(events) == 7

    event_times = [0, 1.265813, 1.297431, 1.773775, 3.909915, 4.192884, 9.164048]
    event_durations = [10.0, 0.410168, 1.028956, 2.413936, 3.058655, 1.221705, 0.835952]
    event_labels = [
        "noise",
        "gun_shot",
        "jackhammer",
        "street_music",
        "air_conditioner",
        "jackhammer",
        "dog_bark",
    ]
    event_confidences = [1.0] * 7

    for e, t, d, l, c in zip(
        events, event_times, event_durations, event_labels, event_confidences
    ):
        assert np.allclose(e.time, t)
        assert np.allclose(e.duration, d)
        assert np.allclose(e.time, t)
        assert np.allclose(e.confidence, c)

    # validate metadata
    assert jam.file_metadata.duration == 10.0
    assert jam.file_metadata.jams_version == "0.2.2"

    reg_scaper_sandbox = {
        "polyphony_gini": 0.5436976533071897,
        "protected_labels": [],
        "fg_spec": [
            [
                ["choose", []],
                ["choose", []],
                ["const", 0],
                ["uniform", 0, 10],
                ["uniform", 0.5, 4],
                ["uniform", 6, 30],
                "foreground",
                ["uniform", -3, 3],
                ["uniform", 0.8, 1.2],
            ],
            [
                ["choose", []],
                ["choose", []],
                ["const", 0],
                ["uniform", 0, 10],
                ["uniform", 0.5, 4],
                ["uniform", 6, 30],
                "foreground",
                ["uniform", -3, 3],
                ["uniform", 0.8, 1.2],
            ],
            [
                ["choose", []],
                ["choose", []],
                ["const", 0],
                ["uniform", 0, 10],
                ["uniform", 0.5, 4],
                ["uniform", 6, 30],
                "foreground",
                ["uniform", -3, 3],
                ["uniform", 0.8, 1.2],
            ],
            [
                ["choose", []],
                ["choose", []],
                ["const", 0],
                ["uniform", 0, 10],
                ["uniform", 0.5, 4],
                ["uniform", 6, 30],
                "foreground",
                ["uniform", -3, 3],
                ["uniform", 0.8, 1.2],
            ],
            [
                ["choose", []],
                ["choose", []],
                ["const", 0],
                ["uniform", 0, 10],
                ["uniform", 0.5, 4],
                ["uniform", 6, 30],
                "foreground",
                ["uniform", -3, 3],
                ["uniform", 0.8, 1.2],
            ],
            [
                ["choose", []],
                ["choose", []],
                ["const", 0],
                ["uniform", 0, 10],
                ["uniform", 0.5, 4],
                ["uniform", 6, 30],
                "foreground",
                ["uniform", -3, 3],
                ["uniform", 0.8, 1.2],
            ],
        ],
        "reverb": 0.1,
        "fg_labels": [
            "engine_idling",
            "children_playing",
            "street_music",
            "dog_bark",
            "gun_shot",
            "car_horn",
            "siren",
            "air_conditioner",
            "jackhammer",
            "drilling",
        ],
        "polyphony_max": 2,
        "bg_labels": ["noise"],
        "allow_repeated_source": False,
        "fade_in_len": 0.01,
        "duration": 10,
        "ref_db": -50,
        "n_channels": 1,
        "bg_path": "/scratch/js7561/datasets/scaper_waspaa2017/audio/soundbanks/train/background/",
        "fg_path": "/scratch/js7561/datasets/scaper_waspaa2017/audio/soundbanks/train/foreground/",
        "allow_repeated_label": True,
        "scaper_version": "0.1.0",
        "bg_spec": [
            [
                ["const", "noise"],
                ["choose", []],
                ["const", 0],
                ["const", 0],
                ["const", 10],
                ["const", 0],
                "background",
                None,
                None,
            ]
        ],
        "n_events": 6,
        "fade_out_len": 0.01,
    }
    scaper_sandbox = jam.search(namespace="scaper")[0]["sandbox"]["scaper"]
    for key in reg_scaper_sandbox.keys():
        if key == "polyphony_gini":
            assert np.allclose(scaper_sandbox[key], reg_scaper_sandbox[key])
        else:
            assert scaper_sandbox[key] == reg_scaper_sandbox[key]

    assert jam.annotations[0].annotation_metadata.data_source == "soundata"
