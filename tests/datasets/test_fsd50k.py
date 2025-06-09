import numpy as np

from tests.test_utils import run_clip_tests

from soundata import annotations, download_utils
from soundata.datasets import fsd50k

import os
import shutil
import pytest

TEST_DATA_HOME = os.path.normpath("tests/resources/sound_datasets/fsd50k")


def test_clip():
    default_clipid = "64760"
    dataset = fsd50k.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip(default_clipid)
    expected_attributes = {
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/sound_datasets/fsd50k/"),
            "FSD50K.dev_audio/64760.wav",
        ),
        "clip_id": "64760",
    }

    expected_property_types = {
        "audio": tuple,
        "tags": annotations.Tags,
        "mids": annotations.Tags,
        "split": str,
        "title": str,
        "description": str,
        "pp_pnp_ratings": dict,
    }

    run_clip_tests(clip, expected_attributes, expected_property_types)


def test_load_audio():
    dataset = fsd50k.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip("64760")
    audio_path = clip.audio_path
    audio, sr = fsd50k.load_audio(audio_path)
    assert sr == 44100
    assert type(audio) is np.ndarray
    assert len(audio.shape) == 1  # check audio is loaded as mono
    assert len(audio) == 75601


def test_labels():
    # For multiple tags
    default_clipid = "64760"
    dataset = fsd50k.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip(default_clipid)
    tags = clip.tags
    assert tags.labels == [
        "Electric_guitar",
        "Guitar",
        "Plucked_string_instrument",
        "Musical_instrument",
        "Music",
    ]
    assert np.array_equal(tags.confidence, [1.0, 1.0, 1.0, 1.0, 1.0])

    mids = clip.mids
    assert mids.labels == ["/m/02sgy", "/m/0342h", "/m/0fx80y", "/m/04szw", "/m/04rlf"]
    assert np.array_equal(mids.confidence, [1.0, 1.0, 1.0, 1.0, 1.0])

    # For a single tag
    default_clipid = "21914"
    dataset = fsd50k.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip(default_clipid)
    tags = clip.tags
    assert tags.labels == ["Crushing"]
    assert np.array_equal(tags.confidence, [1.0])


def test_dev_metadata():
    # Testing metadata from a training clip
    default_clipid = "64760"
    dataset = fsd50k.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip(default_clipid)
    clip_metadata = clip._metadata()

    clip_ground_truth = clip_metadata[default_clipid]["ground_truth"]
    assert clip_ground_truth["tags"] == [
        "Electric_guitar",
        "Guitar",
        "Plucked_string_instrument",
        "Musical_instrument",
        "Music",
    ]
    assert clip_ground_truth["mids"] == [
        "/m/02sgy",
        "/m/0342h",
        "/m/0fx80y",
        "/m/04szw",
        "/m/04rlf",
    ]
    assert clip_ground_truth["split"] == "train"

    clip_info = clip_metadata[default_clipid]["clip_info"]
    assert clip_info["title"] == "guitarras_63.wav"
    assert clip_info["description"] == "electric guitar"
    assert clip_info["tags"] == ["electric", "guitar"]
    assert clip_info["license"] == "http://creativecommons.org/licenses/sampling+/1.0/"
    assert clip_info["uploader"] == "casualsamples"

    clip_pp_pnp = clip_metadata[default_clipid]["pp_pnp_ratings"]
    assert type(clip_pp_pnp) is dict
    assert clip_pp_pnp == {"/m/02sgy": [1.0, 1.0]}

    # Testing metadata from an evaluation clip
    default_clipid = "21914"
    dataset = fsd50k.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip(default_clipid)
    clip_metadata = clip._metadata()
    clip_ground_truth = clip_metadata[default_clipid]["ground_truth"]
    assert clip_ground_truth["split"] == "validation"

    default_clipid = "99"
    dataset = fsd50k.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip(default_clipid)
    clip_metadata = clip._metadata()

    clip_ground_truth = clip_metadata[default_clipid]["ground_truth"]
    assert clip_ground_truth["tags"] == [
        "Chatter",
        "Chewing_and_mastication",
        "Chirp_and_tweet",
        "Traffic_noise_and_roadway_noise",
        "Child_speech_and_kid_speaking",
        "Human_group_actions",
        "Bird_vocalization_and_bird_call_and_bird_song",
        "Bird",
        "Wild_animals",
        "Animal",
        "Motor_vehicle_(road)",
        "Vehicle",
        "Speech",
        "Human_voice",
    ]
    assert clip_ground_truth["mids"] == [
        "/m/07rkbfh",
        "/m/03cczk",
        "/m/07pggtn",
        "/m/0btp2",
        "/m/0ytgt",
        "/t/dd00012",
        "/m/020bb7",
        "/m/015p6",
        "/m/01280g",
        "/m/0jbk",
        "/m/012f08",
        "/m/07yv9",
        "/m/09x0r",
        "/m/09l8g",
    ]
    assert clip_ground_truth["split"] is "test"

    clip_info = clip_metadata[default_clipid]["clip_info"]
    assert type(clip_info) is dict
    assert clip_info["title"] == "manzana_exterior.wav"
    assert clip_info["description"] == "eating apples on a park"
    assert clip_info["tags"] == ["apple", "crunch", "eat", "field-recording", "park"]
    assert clip_info["license"] == "http://creativecommons.org/licenses/by/3.0/"
    assert clip_info["uploader"] == "plagasul"

    clip_pp_pnp = clip_metadata[default_clipid]["pp_pnp_ratings"]
    assert type(clip_pp_pnp) is dict
    assert clip_pp_pnp == {"/m/03cczk": [0.5, 0.5]}


def test_load_vocabulary():
    dataset = fsd50k.Dataset(TEST_DATA_HOME, version="test")

    # Testing load vocabulary function
    fsd50k_to_audioset, audioset_to_fsd50k = dataset.load_fsd50k_vocabulary(
        dataset.vocabulary_path
    )
    assert fsd50k_to_audioset == {
        "Crushing": "/m/07plct2",
        "Electric_guitar": "/m/02sgy",
    }
    assert audioset_to_fsd50k == {
        "/m/07plct2": "Crushing",
        "/m/02sgy": "Electric_guitar",
    }

    # Testing fsd50k to audioset
    fsd50k_to_audioset = dataset.fsd50k_to_audioset
    assert fsd50k_to_audioset["Crushing"] == "/m/07plct2"
    assert fsd50k_to_audioset["Electric_guitar"] == "/m/02sgy"

    # Testing audioset to fsd50k
    audioset_to_fsd50k = dataset.audioset_to_fsd50k
    assert audioset_to_fsd50k["/m/07plct2"] == "Crushing"
    assert audioset_to_fsd50k["/m/02sgy"] == "Electric_guitar"


def test_label_info():
    dataset = fsd50k.Dataset(TEST_DATA_HOME, version="test")

    # Testing label info property
    label_info = dataset.label_info

    assert type(label_info["/m/02sgy"]) is dict
    assert type(label_info["/m/02sgy"]["faq"]) is str
    assert label_info["/m/02sgy"]["examples"] == [4282, 134012]
    assert label_info["/m/02sgy"]["verification_examples"] == [
        63900,
        74871,
        40403,
        97244,
        40474,
        97242,
    ]


def test_vocabularies():
    dataset = fsd50k.Dataset(TEST_DATA_HOME, version="test")

    # Testing load vocabulary function
    fsd50k_to_audioset, audioset_to_fsd50k = dataset.load_fsd50k_vocabulary(
        dataset.vocabulary_path
    )
    assert fsd50k_to_audioset == {
        "Crushing": "/m/07plct2",
        "Electric_guitar": "/m/02sgy",
    }
    assert audioset_to_fsd50k == {
        "/m/07plct2": "Crushing",
        "/m/02sgy": "Electric_guitar",
    }

    # Testing fsd50k to audioset
    fsd50k_to_audioset = dataset.fsd50k_to_audioset
    assert fsd50k_to_audioset["Crushing"] == "/m/07plct2"
    assert fsd50k_to_audioset["Electric_guitar"] == "/m/02sgy"

    # Testing audioset to fsd50k
    audioset_to_fsd50k = dataset.audioset_to_fsd50k
    assert audioset_to_fsd50k["/m/07plct2"] == "Crushing"
    assert audioset_to_fsd50k["/m/02sgy"] == "Electric_guitar"


def test_collection_vocabulary():
    dataset = fsd50k.Dataset(TEST_DATA_HOME, version="test")

    # Testing collection vocabularies
    collection_fsd50k_to_audioset = dataset.collection_fsd50k_to_audioset
    collection_audioset_to_fsd50k = dataset.collection_audioset_to_fsd50k

    assert type(collection_fsd50k_to_audioset) is dict
    assert type(collection_audioset_to_fsd50k) is dict
    assert type(collection_fsd50k_to_audioset["dev"]) is dict
    assert type(collection_fsd50k_to_audioset["eval"]) is dict
    assert type(collection_audioset_to_fsd50k["dev"]) is dict
    assert type(collection_audioset_to_fsd50k["eval"]) is dict

    assert collection_fsd50k_to_audioset["dev"]["Electric_guitar"] == "/m/02sgy"
    assert collection_fsd50k_to_audioset["eval"]["Chatter"] == "/m/07rkbfh"
    assert collection_audioset_to_fsd50k["dev"]["/m/02sgy"] == "Electric_guitar"
    assert collection_audioset_to_fsd50k["eval"]["/m/07rkbfh"] == "Chatter"
