import os
import numpy as np

from tests.test_utils import run_clip_tests

from soundata import annotations
from soundata.datasets import fsd50k
from tests.test_utils import DEFAULT_DATA_HOME


TEST_DATA_HOME = "tests/resources/sound_datasets/fsd50k"


def test_clip():
    default_clipid = "64760"
    dataset = fsd50k.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)

    expected_attributes = {
        "audio_path": "tests/resources/sound_datasets/fsd50k/FSD50K.dev_audio/64760.wav",
        "clip_id": "64760",
        "sub_set": "dev",
    }

    expected_property_types = {
        "audio": tuple,
        "labels": annotations.Tags,
        "split": str,
    }

    run_clip_tests(clip, expected_attributes, expected_property_types)


def test_load_audio():
    dataset = fsd50k.Dataset(TEST_DATA_HOME)
    clip = dataset.clip("64760")
    audio_path = clip.audio_path
    audio, sr = fsd50k.load_audio(audio_path)
    assert sr == 44100
    assert type(audio) is np.ndarray
    assert len(audio.shape) == 1  # check audio is loaded as mono


'''
def test_to_jams():

    # Note: original file is 5 sec, but for testing we've trimmed it to 1 sec
    default_clipid = "1-104089-A-22"
    dataset = esc50.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)
    jam = clip.to_jams()

    # Validate esc50 jam schema
    assert jam.validate()

    # Validate Tags
    tags = jam.search(namespace="tag_open")[0]["data"]
    assert len(tags) == 1
    assert tags[0].time == 0
    assert tags[0].duration == 1.0
    assert tags[0].value == "clapping"
    assert tags[0].confidence == 1

    # validate metadata
    assert jam.file_metadata.duration == 1.0
    assert jam.sandbox.filename == "1-104089-A-22.wav"
    assert jam.sandbox.fold == 1
    assert jam.sandbox.target == 22
    assert jam.sandbox.category == "clapping"
    assert jam.sandbox.esc10 == False
    assert jam.sandbox.src_file == "104089"
    assert jam.annotations[0].annotation_metadata.data_source == "soundata"
'''

def test_tags():
    default_clipid = "64760"
    dataset = fsd50k.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)
    tags = clip.labels
    assert tags.labels == [
        'Electric_guitar', 'Guitar', 'Plucked_string_instrument', 'Musical_instrument', 'Music'
    ]


def test_metadata():
    default_clipid = "64760"
    dataset = fsd50k.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)
    clip_metadata = clip._metadata()

    clip_ground_truth = clip_metadata['ground_truth_dev'][default_clipid]
    assert clip_ground_truth['tags'] == [
        'Electric_guitar', 'Guitar', 'Plucked_string_instrument', 'Musical_instrument', 'Music'
    ]
    assert clip_ground_truth['mids'] == [
        '/m/02sgy', '/m/0342h', '/m/0fx80y', '/m/04szw', '/m/04rlf'
    ]
    assert clip_ground_truth['split'] == 'train'

    clip_info = clip_metadata['clips_info_dev'][default_clipid]
    assert clip_info['title'] == 'guitarras_63.wav'
    assert clip_info['description'] == 'electric guitar'
    assert clip_info['tags'] == ['electric', 'guitar']
    assert clip_info['license'] == 'http://creativecommons.org/licenses/sampling+/1.0/'
    assert clip_info['uploader'] == 'casualsamples'

