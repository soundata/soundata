import numpy as np

from tests.test_utils import run_clip_tests

from soundata import annotations
from soundata.datasets import fsd50k

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
        "description": str,
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


def test_to_jams():
    default_clipid = "64760"
    dataset = fsd50k.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)
    jam = clip.to_jams()

    # Validate fsd50k jam schema
    assert jam.validate()

    # Validate Tags
    tags = jam.search(namespace="tag_open")[0]["data"]
    assert len(tags) == 5
    assert [tag.time for tag in tags] == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert [tag.duration for tag in tags] == [
        1.7143083900226757,
        1.7143083900226757,
        1.7143083900226757,
        1.7143083900226757,
        1.7143083900226757,
    ]
    assert [tag.value for tag in tags] == [
        "Electric_guitar",
        "Guitar",
        "Plucked_string_instrument",
        "Musical_instrument",
        "Music",
    ]
    assert [tag.confidence for tag in tags] == [1.0, 1.0, 1.0, 1.0, 1.0]

    # validate metadata
    assert jam.file_metadata.duration == 1.7143083900226757
    assert jam.file_metadata.title == "guitarras_63.wav"
    assert jam.sandbox.mids == [
        "/m/02sgy",
        "/m/0342h",
        "/m/0fx80y",
        "/m/04szw",
        "/m/04rlf",
    ]
    assert jam.sandbox.split == "train"
    assert jam.sandbox.description == "electric guitar"
    assert jam.sandbox.tags == [
        "electric",
        "guitar",
    ]
    assert jam.sandbox.license == "http://creativecommons.org/licenses/sampling+/1.0/"
    assert jam.sandbox.uploader == "casualsamples"
    assert jam.sandbox.pp_pnp_ratings == {"/m/02sgy": [1.0, 1.0]}
    assert jam.annotations[0].annotation_metadata.data_source == "soundata"


def test_tags():
    default_clipid = "64760"
    dataset = fsd50k.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)
    tags = clip.labels
    assert tags.labels == [
        "Electric_guitar",
        "Guitar",
        "Plucked_string_instrument",
        "Musical_instrument",
        "Music",
    ]


def test_dev_metadata():
    default_clipid = "64760"
    dataset = fsd50k.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)
    clip_metadata = clip._metadata()

    clip_ground_truth = clip_metadata["ground_truth_dev"][default_clipid]
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

    clip_info = clip_metadata["clips_info_dev"][default_clipid]
    assert clip_info["title"] == "guitarras_63.wav"
    assert clip_info["description"] == "electric guitar"
    assert clip_info["tags"] == ["electric", "guitar"]
    assert clip_info["license"] == "http://creativecommons.org/licenses/sampling+/1.0/"
    assert clip_info["uploader"] == "casualsamples"

    clip_pp_pnp = clip_metadata["pp_pnp_ratings"][default_clipid]
    assert type(clip_pp_pnp) is dict
    assert clip_pp_pnp == {"/m/02sgy": [1.0, 1.0]}


def test_eval_metadata():
    default_clipid = "99"
    dataset = fsd50k.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)
    clip_metadata = clip._metadata()

    clip_ground_truth = clip_metadata["ground_truth_eval"][default_clipid]
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
    assert clip_ground_truth["split"] is None

    clip_info = clip_metadata["clips_info_eval"][default_clipid]
    assert type(clip_info) is dict
    assert clip_info["title"] == "manzana_exterior.wav"
    assert clip_info["description"] == "eating apples on a park"
    assert clip_info["tags"] == ["apple", "crunch", "eat", "field-recording", "park"]
    assert clip_info["license"] == "http://creativecommons.org/licenses/by/3.0/"
    assert clip_info["uploader"] == "plagasul"

    clip_pp_pnp = clip_metadata["pp_pnp_ratings"][default_clipid]
    assert type(clip_pp_pnp) is dict
    assert clip_pp_pnp == {"/m/03cczk": [0.5, 0.5]}
