import numpy as np

from tests.test_utils import run_clip_tests

from soundata import annotations, download_utils
from soundata.datasets import fsd50k

import os
import shutil

TEST_DATA_HOME = "tests/resources/sound_datasets/fsd50k"


def test_clip():
    default_clipid = "64760"
    dataset = fsd50k.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)
    print(clip)
    expected_attributes = {
        "audio_path": "tests/resources/sound_datasets/fsd50k/FSD50K.dev_audio/64760.wav",
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
    dataset = fsd50k.Dataset(TEST_DATA_HOME)
    clip = dataset.clip("64760")
    audio_path = clip.audio_path
    audio, sr = fsd50k.load_audio(audio_path)
    assert sr == 44100
    assert type(audio) is np.ndarray
    assert len(audio.shape) == 1  # check audio is loaded as mono
    assert len(audio) == 75601


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
    print(jam.file_metadata)
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
    assert jam.sandbox.freesound_tags == [
        "electric",
        "guitar",
    ]
    assert jam.sandbox.license == "http://creativecommons.org/licenses/sampling+/1.0/"
    assert jam.sandbox.uploader == "casualsamples"
    assert jam.sandbox.pp_pnp_ratings == {"/m/02sgy": [1.0, 1.0]}
    assert jam.annotations[0].annotation_metadata.data_source == "soundata"


def test_labels():
    # For multiple tags
    default_clipid = "64760"
    dataset = fsd50k.Dataset(TEST_DATA_HOME)
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
    assert mids.labels == [
        "/m/02sgy",
        "/m/0342h",
        "/m/0fx80y",
        "/m/04szw",
        "/m/04rlf",
    ]
    assert np.array_equal(mids.confidence, [1.0, 1.0, 1.0, 1.0, 1.0])

    # For a single tag
    default_clipid = "21914"
    dataset = fsd50k.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)
    tags = clip.tags
    assert tags.labels == ["Crushing"]
    assert np.array_equal(tags.confidence, [1.0])


def test_dev_metadata():
    # Testing metadata from a training clip
    default_clipid = "64760"
    dataset = fsd50k.Dataset(TEST_DATA_HOME)
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
    dataset = fsd50k.Dataset(TEST_DATA_HOME)
    clip = dataset.clip(default_clipid)
    clip_metadata = clip._metadata()
    clip_ground_truth = clip_metadata[default_clipid]["ground_truth"]
    assert clip_ground_truth["split"] == "validation"

    default_clipid = "99"
    dataset = fsd50k.Dataset(TEST_DATA_HOME)
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
    dataset = fsd50k.Dataset(TEST_DATA_HOME)

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
    dataset = fsd50k.Dataset(TEST_DATA_HOME)

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
    dataset = fsd50k.Dataset(TEST_DATA_HOME)

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
    dataset = fsd50k.Dataset(TEST_DATA_HOME)

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


def test_download_partial(httpserver):

    test_download_home = "tests/resources/sound_datasets/fsd50k_download"
    if os.path.exists(test_download_home):
        shutil.rmtree(test_download_home)

    test_files_path = "tests/resources/download/fsd50k.zip"
    download_utils.unzip(test_files_path, cleanup=False)

    httpserver.serve_content(
        open("tests/resources/download/fsd50k/FSD50K.ground_truth.zip", "rb").read()
    )
    remotes = {
        "ground_truth": download_utils.RemoteFileMetadata(
            filename="1-FSD50K.ground_truth.zip",
            url=httpserver.url,
            checksum="246dd703ab54859e6497eee101e311e7",
        ),
        "metadata": download_utils.RemoteFileMetadata(
            filename="2-FSD50K.ground_truth.zip",
            url=httpserver.url,
            checksum="246dd703ab54859e6497eee101e311e7",
        ),
        "documentation": download_utils.RemoteFileMetadata(
            filename="3-FSD50K.ground_truth.zip",
            url=httpserver.url,
            checksum="246dd703ab54859e6497eee101e311e7",
        ),
    }
    dataset = fsd50k.Dataset(test_download_home)
    dataset.remotes = remotes
    dataset.download(None, False, False)
    assert os.path.exists(os.path.join(test_download_home, "1-FSD50K.ground_truth.zip"))
    assert os.path.exists(os.path.join(test_download_home, "2-FSD50K.ground_truth.zip"))
    assert os.path.exists(os.path.join(test_download_home, "3-FSD50K.ground_truth.zip"))

    if os.path.exists(test_download_home):
        shutil.rmtree(test_download_home)
    dataset.download(["ground_truth"], False, False)
    assert os.path.exists(os.path.join(test_download_home, "1-FSD50K.ground_truth.zip"))
    assert not os.path.exists(
        os.path.join(test_download_home, "2-FSD50K.ground_truth.zip")
    )
    assert not os.path.exists(
        os.path.join(test_download_home, "3-FSD50K.ground_truth.zip")
    )

    if os.path.exists(test_download_home):
        shutil.rmtree(test_download_home)
    dataset.download(["metadata", "documentation"], False, False)
    assert not os.path.exists(
        os.path.join(test_download_home, "1-FSD50K.ground_truth.zip")
    )
    assert os.path.exists(os.path.join(test_download_home, "2-FSD50K.ground_truth.zip"))
    assert os.path.exists(os.path.join(test_download_home, "3-FSD50K.ground_truth.zip"))

    if os.path.exists(test_download_home):
        shutil.rmtree(test_download_home)

    # Test downloading twice with cleanup
    dataset.download(None, False, True)
    dataset.download(None, False, False)

    if os.path.exists(test_download_home):
        shutil.rmtree(test_download_home)

    # Test downloading twice with force overwrite
    dataset.download(None, False, False)
    dataset.download(None, True, False)

    if os.path.exists(test_download_home):
        shutil.rmtree(test_download_home)

    # Test downloading twice with force overwrite and cleanup
    dataset.download(None, False, True)
    dataset.download(None, True, False)

    if os.path.exists(test_download_home):
        shutil.rmtree(test_download_home)

    if os.path.exists(os.path.join("tests/resources/download", "__MACOSX")):
        shutil.rmtree(
            os.path.join("tests/resources/download", "__MACOSX"), ignore_errors=True
        )


def test_merge_and_unzip():
    test_merging_home = "tests/resources/download/fsd50k"
    test_files_path = "tests/resources/download/fsd50k.zip"

    dataset = fsd50k.Dataset(test_merging_home)
    download_utils.unzip(test_files_path, cleanup=False)

    # Test development merge and unzip
    assert not os.path.exists(os.path.join(test_merging_home, "FSD50K.dev_audio"))
    assert os.path.exists(os.path.join(test_merging_home, "FSD50K.dev_audio.zip"))
    assert os.path.exists(os.path.join(test_merging_home, "FSD50K.dev_audio.z01"))
    assert os.path.exists(os.path.join(test_merging_home, "FSD50K.dev_audio.z02"))
    assert os.path.exists(os.path.join(test_merging_home, "FSD50K.dev_audio.z03"))
    assert os.path.exists(os.path.join(test_merging_home, "FSD50K.dev_audio.z04"))
    assert os.path.exists(os.path.join(test_merging_home, "FSD50K.dev_audio.z05"))

    merging_list_dev = [
        "FSD50K.dev_audio.zip",
        "FSD50K.dev_audio.z01",
        "FSD50K.dev_audio.z02",
        "FSD50K.dev_audio.z03",
        "FSD50K.dev_audio.z04",
        "FSD50K.dev_audio.z05",
    ]

    dataset.merge_and_unzip(merging_list=merging_list_dev, cleanup=False)

    assert os.path.exists(os.path.join(test_merging_home, "FSD50K.dev_audio"))
    assert os.path.exists(os.path.join(test_merging_home, "FSD50K.dev_audio.zip"))
    assert os.path.exists(os.path.join(test_merging_home, "FSD50K.dev_audio.z01"))
    assert os.path.exists(os.path.join(test_merging_home, "FSD50K.dev_audio.z02"))
    assert os.path.exists(os.path.join(test_merging_home, "FSD50K.dev_audio.z03"))
    assert os.path.exists(os.path.join(test_merging_home, "FSD50K.dev_audio.z04"))
    assert os.path.exists(os.path.join(test_merging_home, "FSD50K.dev_audio.z05"))

    # Test evaluation merge and unzip
    assert not os.path.exists(os.path.join(test_merging_home, "FSD50K.eval_audio"))
    assert os.path.exists(os.path.join(test_merging_home, "FSD50K.eval_audio.zip"))
    assert os.path.exists(os.path.join(test_merging_home, "FSD50K.eval_audio.z01"))

    merging_list_eval = [
        "FSD50K.eval_audio.zip",
        "FSD50K.eval_audio.z01",
    ]

    dataset.merge_and_unzip(merging_list=merging_list_eval, cleanup=False)

    assert os.path.exists(os.path.join(test_merging_home, "FSD50K.eval_audio/"))
    assert os.path.exists(os.path.join(test_merging_home, "FSD50K.eval_audio.zip"))
    assert os.path.exists(os.path.join(test_merging_home, "FSD50K.eval_audio.z01"))

    if os.path.exists(os.path.join(test_merging_home)):
        shutil.rmtree(test_merging_home, ignore_errors=True)

    if os.path.exists(os.path.join("tests/resources/download", "__MACOSX")):
        shutil.rmtree(
            os.path.join("tests/resources/download", "__MACOSX"), ignore_errors=True
        )


def test_merge_unzip_cleanup():
    test_merging_home = "tests/resources/download/fsd50k"
    test_files_path = "tests/resources/download/fsd50k.zip"

    dataset = fsd50k.Dataset(test_merging_home)
    download_utils.unzip(test_files_path, cleanup=False)

    # Test development merge and unzip
    assert not os.path.exists(os.path.join(test_merging_home, "FSD50K.dev_audio"))
    assert os.path.exists(os.path.join(test_merging_home, "FSD50K.dev_audio.zip"))
    assert os.path.exists(os.path.join(test_merging_home, "FSD50K.dev_audio.z01"))
    assert os.path.exists(os.path.join(test_merging_home, "FSD50K.dev_audio.z02"))
    assert os.path.exists(os.path.join(test_merging_home, "FSD50K.dev_audio.z03"))
    assert os.path.exists(os.path.join(test_merging_home, "FSD50K.dev_audio.z04"))
    assert os.path.exists(os.path.join(test_merging_home, "FSD50K.dev_audio.z05"))

    merging_list_dev = [
        "FSD50K.dev_audio.zip",
        "FSD50K.dev_audio.z01",
        "FSD50K.dev_audio.z02",
        "FSD50K.dev_audio.z03",
        "FSD50K.dev_audio.z04",
        "FSD50K.dev_audio.z05",
    ]

    dataset.merge_and_unzip(merging_list=merging_list_dev, cleanup=True)

    assert os.path.exists(os.path.join(test_merging_home, "FSD50K.dev_audio"))
    assert not os.path.exists(os.path.join(test_merging_home, "FSD50K.dev_audio.zip"))
    assert not os.path.exists(os.path.join(test_merging_home, "FSD50K.dev_audio.z01"))
    assert not os.path.exists(os.path.join(test_merging_home, "FSD50K.dev_audio.z02"))
    assert not os.path.exists(os.path.join(test_merging_home, "FSD50K.dev_audio.z03"))
    assert not os.path.exists(os.path.join(test_merging_home, "FSD50K.dev_audio.z04"))
    assert not os.path.exists(os.path.join(test_merging_home, "FSD50K.dev_audio.z05"))

    # Test evaluation merge and unzip
    assert not os.path.exists(os.path.join(test_merging_home, "FSD50K.eval_audio"))
    assert os.path.exists(os.path.join(test_merging_home, "FSD50K.eval_audio.zip"))
    assert os.path.exists(os.path.join(test_merging_home, "FSD50K.eval_audio.z01"))

    merging_list_eval = [
        "FSD50K.eval_audio.zip",
        "FSD50K.eval_audio.z01",
    ]

    dataset.merge_and_unzip(merging_list=merging_list_eval, cleanup=True)

    assert os.path.exists(os.path.join(test_merging_home, "FSD50K.eval_audio/"))
    assert not os.path.exists(os.path.join(test_merging_home, "FSD50K.eval_audio.zip"))
    assert not os.path.exists(os.path.join(test_merging_home, "FSD50K.eval_audio.z01"))

    if os.path.exists(os.path.join(test_merging_home)):
        shutil.rmtree(test_merging_home, ignore_errors=True)

    if os.path.exists(os.path.join("tests/resources/download", "__MACOSX")):
        shutil.rmtree(
            os.path.join("tests/resources/download", "__MACOSX"), ignore_errors=True
        )
