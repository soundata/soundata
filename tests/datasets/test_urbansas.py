import os
import numpy as np

from tests.test_utils import run_clip_tests

from soundata import annotations
from soundata.datasets import urbansas
from tests.test_utils import DEFAULT_DATA_HOME


TEST_DATA_HOME = "tests/resources/sound_datasets/urbansas"


def test_clip():
    default_clipid = "acevedo0103_00_0"
    dataset = urbansas.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)

    expected_attributes = {
        "audio_path": "tests/resources/sound_datasets/urbansas/audio/acevedo0103_00_0.wav",
        "video_path": "tests/resources/sound_datasets/urbansas/video/video_2fps/acevedo0103_00_0.mp4",
        "clip_id": "acevedo0103_00_0",
    }

    expected_property_types = {
        "audio": tuple,
        "video": tuple,
        "city": str,
        "location_id": str,
        "night": bool,
        "non_identifiable_vehicle_sound": bool,
        "events": annotations.Events,
        "video_annotations": annotations.VideoAnnotations,
    }

    run_clip_tests(clip, expected_attributes, expected_property_types)

    assert clip.events.intervals.shape == (2, 2)
    assert clip.city == "montevideo"
    assert not clip.night
    assert clip.location_id == "acevedo0103"
    assert clip.non_identifiable_vehicle_sound


def test_load_audio():
    dataset = urbansas.Dataset(TEST_DATA_HOME)
    clip = dataset.clip("acevedo0103_00_0")
    audio_path = clip.audio_path
    audio, sr = urbansas.load_audio(audio_path)
    assert sr == 44100
    assert type(audio) is np.ndarray
    assert audio.shape[0] == 2  # check audio is loaded as stereo
    assert (
        audio.shape[1] == 10 * 44100
    )  # Check audio duration in samples is as expected


def test_load_video():
    dataset = urbansas.Dataset(TEST_DATA_HOME)
    clip = dataset.clip("acevedo0103_00_0")
    video_path = clip.video_path
    video, fps = urbansas.load_video(video_path)
    assert fps == 2.0
    assert type(video) is list
    assert len(video) == 20
    assert type(video[0]) is np.ndarray
    assert video[0].shape[2] == 3  # check video channels
    assert video[0].shape[1] == 1280  # check video width
    assert video[0].shape[0] == 720  # check video height


def test_to_jams():
    dataset = urbansas.Dataset(TEST_DATA_HOME)
    clip = dataset.clip("acevedo0103_00_0")
    jam = clip.to_jams()

    # Validate urbansound8k jam schema
    assert jam.validate()

    # Validate Events
    events = jam.search(namespace="segment_open")[0]["data"]
    assert len(events) == 2

    assert np.allclose(events[0].time, 0.449)
    assert np.allclose(events[0].duration, 10.0 - 0.449)
    assert events[0].value == "offscreen"
    assert events[0].confidence == 1

    assert np.allclose(events[1].time, 3.583)
    assert np.allclose(events[1].duration, 10.0 - 3.583)
    assert events[1].value == "bus"
    assert events[1].confidence == 1

    # validate metadata
    assert jam.file_metadata.duration == 10.0
    assert jam.sandbox.city == "montevideo"
    assert jam.sandbox.location_id == "acevedo0103"
    assert not jam.sandbox.night
    assert clip.non_identifiable_vehicle_sound
    assert jam.annotations[0].annotation_metadata.data_source == "soundata"
