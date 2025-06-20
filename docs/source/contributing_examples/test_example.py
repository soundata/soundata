import numpy as np

from soundata import annotations
from soundata.datasets import example  # the name of your loader here
from tests.test_utils import run_clip_tests

TEST_DATA_HOME = "tests/resources/sound_datasets/example"


def test_clip():
    default_clipid = "some_id"
    dataset = example.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip(default_clipid)

    expected_attributes = {
        "clip_id": "some_id",
        "audio_path": "tests/resources/sound_datasets/example/audio/some_id.wav",
        "annotation_path": "tests/resources/sound_datasets/example/annotation/some_id.pv",
    }

    # List here all the properties of your loader
    expected_property_types = {
        "tags": annotations.Tags,
        "some_other_annotation": "some_annotation_type",
    }

    run_clip_tests(clip, expected_attributes, expected_property_types)


# Test all the load functions, for instance, the load audio one
def test_load_audio():
    dataset = example.Dataset(TEST_DATA_HOME)
    clip = dataset.clip("some_id")
    audio_path = clip.audio_path
    audio, sr = example.load_audio(audio_path)
    assert sr == 44100
    assert type(audio) is np.ndarray
    assert len(audio.shape) == 1  # check audio is loaded e.g. as mono
    assert audio.shape[0] == 44100  # Check audio duration in samples is as expected


# Test each of the load functions (e.g. Tags, etc)
def test_load_annotation():
    # load a file which exists
    annotation_path = "tests/resources/sound_datasets/dataset/annotation/some_id.pv"
    annotation_data = example.load_annotation(annotation_path)

    # check types
    assert type(annotation_data) == "some_annotation_type"
    assert type(annotation_data.times) is np.ndarray
    # ... etc

    # check values
    assert np.array_equal(annotation_data.times, np.array([0.016, 0.048]))
    # ... etc


def test_metadata():
    data_home = "tests/resources/sound_datasets/dataset"
    dataset = example.Dataset(data_home, version="test")
    metadata = dataset._metadata
    assert metadata["some_id"] == "something"
