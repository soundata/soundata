import numpy as np
import os

from soundata import annotations
from soundata.datasets import clotho
from tests.test_utils import run_clip_tests

TEST_DATA_HOME = os.path.normpath("tests/resources/sound_datasets/clotho")


def test_clip():
    default_clipid = " Ambience Birds"
    dataset = clotho.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip(default_clipid)

    expected_attributes = {
        "audio_path": os.path.join(
            TEST_DATA_HOME, "clotho_audio_development/ Ambience Birds.wav"
        ),
        "clip_id": " Ambience Birds",
    }

    # List here all the properties of your loader
    expected_property_types = {
        "audio": tuple,
        "file_name": str,
        "keywords": str,
        "sound_id": str,
        "sound_link": str,
        "start_end_samples": str,
        "manufacturer": str,
        "license": str,
        "captions": list,
        "split": str,
    }

    run_clip_tests(clip, expected_attributes, expected_property_types)


def test_properties():
    default_clipid = " Ambience Birds"
    dataset = clotho.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip(default_clipid)

    assert clip.file_name == " Ambience Birds.wav"
    assert clip.keywords == "Ambience;outside;OWI;Birds;night"
    assert clip.sound_id == "327673"
    assert (
        clip.sound_link
        == "https://freesound.org/people/Juan_Merie_Venter/sounds/327673"
    )
    assert clip.start_end_samples == "[11162624, 11932169]"
    assert clip.manufacturer == "Juan_Merie_Venter"
    assert clip.license == "http://creativecommons.org/licenses/by-nc/3.0/"
    assert clip.captions == [
        "A wild assortment of birds are chirping and calling out in nature.",
        "Several different types of bird are tweeting and making calls.",
        "Birds tweeting and chirping happily, engine in the distance.",
        "An assortment of  wild birds are chirping and calling out in nature.",
        "Birds are chirping and making loud bird noises.",
    ]
    assert clip.split == "development"


# Test all the load functions, for instance, the load audio one
def test_load_audio():
    default_clipid = " Ambience Birds"
    dataset = clotho.Dataset(TEST_DATA_HOME, version="test")
    clip = dataset.clip(default_clipid)
    audio_path = clip.audio_path
    audio, sr = clotho.load_audio(audio_path)
    assert sr == 44100
    assert type(audio) is np.ndarray
    assert len(audio.shape) == 1  # check audio is loaded e.g. as mono
    assert audio.shape[0] == 44100  # Check audio duration in samples is as expected


def test_metadata():

    dataset = clotho.Dataset(TEST_DATA_HOME, version="test")

    metadata = dataset._metadata

    assert metadata[" Ambience Birds"]["clip_id"] == " Ambience Birds"
    assert metadata[" Ambience Birds"]["keywords"] == "Ambience;outside;OWI;Birds;night"
    assert metadata[" Ambience Birds"]["sound_id"] == "327673"
    assert (
        metadata[" Ambience Birds"]["sound_link"]
        == "https://freesound.org/people/Juan_Merie_Venter/sounds/327673"
    )
    assert metadata[" Ambience Birds"]["start_end_samples"] == "[11162624, 11932169]"
    assert metadata[" Ambience Birds"]["manufacturer"] == "Juan_Merie_Venter"
    assert (
        metadata[" Ambience Birds"]["license"]
        == "http://creativecommons.org/licenses/by-nc/3.0/"
    )

    assert metadata[" Ambience Birds"]["captions"] == [
        "A wild assortment of birds are chirping and calling out in nature.",
        "Several different types of bird are tweeting and making calls.",
        "Birds tweeting and chirping happily, engine in the distance.",
        "An assortment of  wild birds are chirping and calling out in nature.",
        "Birds are chirping and making loud bird noises.",
    ]
