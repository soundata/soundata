import importlib
import inspect
from inspect import signature
import io
import json
import os
import sys
import pytest
import requests


import soundata
from soundata import core
from tests.test_utils import get_attributes_and_properties

DATASETS = soundata.DATASETS
CUSTOM_TEST_CLIPS = {
    "dcase23_task6a": "development/1",
    "dcase23_task6b": "development/1",
    "dcase_birdVox20k": "00053d90-e4b9-4045-a2f1-f39efc90cfa9",
    "dcase_bioacoustic": "2015-09-04_08-04-59_unit03",
    "esc50": "1-104089-A-22",
    "freefield1010": "64486",
    "fsd50k": "64760",
    "fsdnoisy18k": "17",
    "tau2019uas": "development/airport-barcelona-0-0-a",
    "tau2022uas_mobile": "airport-lisbon-1000-40000-0-a",
    "tau2020uas_mobile": "airport-barcelona-0-0-a",
    "urbansed": "soundscape_train_uniform1736",
    "urbansound8k": "135776-2-0-49",
    "singapura": "[b827ebf3744c][2020-08-19T22-46-04Z][manual][---][4edbade2d41d5f80e324ee4f10d401c0][]-135",
    "tut2017se": "a001",
    "warblrb10k": "759808e5-f824-401e-9058",
}

REMOTE_DATASETS = {}
TEST_DATA_HOME = "tests/resources/sound_datasets"


def test_dataset_attributes():
    for dataset_name in DATASETS:
        dataset = soundata.initialize(
            dataset_name, os.path.join(TEST_DATA_HOME, dataset_name), version="test"
        )

        assert (
            dataset.name == dataset_name
        ), "{}.dataset attribute does not match dataset name".format(dataset_name)
        assert (
            dataset.bibtex is not None
        ), "No BIBTEX information provided for {}".format(dataset_name)
        assert (
            dataset._license_info is not None
        ), "No LICENSE information provided for {}".format(dataset_name)
        assert (
            isinstance(dataset.remotes, dict) or dataset.remotes is None
        ), "{}.REMOTES must be a dictionary".format(dataset_name)
        assert isinstance(dataset._index, dict), "{}.DATA is not properly set".format(
            dataset_name
        )
        assert (
            isinstance(dataset._download_info, str) or dataset._download_info is None
        ), "{}.DOWNLOAD_INFO must be a string".format(dataset_name)
        assert type(dataset._clip_class) == type(
            core.Clip
        ), "{}.Clip must be an instance of core.Clip".format(dataset_name)
        assert callable(dataset.download), "{}.download is not a function".format(
            dataset_name
        )


def test_cite_and_license():
    for dataset_name in DATASETS:
        module = importlib.import_module("soundata.datasets.{}".format(dataset_name))
        dataset = module.Dataset(
            os.path.join(TEST_DATA_HOME, dataset_name), version="test"
        )

        text_trap = io.StringIO()
        sys.stdout = text_trap
        dataset.cite()
        sys.stdout = sys.__stdout__

        text_trap = io.StringIO()
        sys.stdout = text_trap
        dataset.license()
        sys.stdout = sys.__stdout__


KNOWN_ISSUES = {}  # key is module, value is REMOTE key
DOWNLOAD_EXCEPTIONS = ["maestro"]


def test_download(mocker):
    for dataset_name in DATASETS:
        print(dataset_name)
        dataset = soundata.initialize(
            dataset_name, os.path.join(TEST_DATA_HOME, dataset_name), version="test"
        )

        # test parameters & defaults
        assert callable(dataset.download), "{}.download is not callable".format(
            dataset_name
        )
        params = signature(dataset.download).parameters

        expected_params = [
            ("partial_download", None),
            ("force_overwrite", False),
            ("cleanup", False),
        ]
        for exp in expected_params:
            assert exp[0] in params, "{}.download must have {} as a parameter".format(
                dataset_name, exp[0]
            )
            assert (
                params[exp[0]].default == exp[1]
            ), "The default value of {} in {}.download must be {}".format(
                dataset_name, exp[0], exp[1]
            )

        # check that the download method can be called without errors
        if dataset.remotes != {}:
            mock_downloader = mocker.patch.object(dataset, "remotes")
            if dataset_name not in DOWNLOAD_EXCEPTIONS:
                try:
                    dataset.download()
                except:
                    assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

                mocker.resetall()

            # check that links are online
            for key in dataset.remotes:
                # skip this test if it's in known issues
                if dataset_name in KNOWN_ISSUES and key in KNOWN_ISSUES[dataset_name]:
                    continue

                url = dataset.remotes[key].url
                try:
                    request = requests.head(url)
                    assert request.ok, "Link {} for {} does not return OK".format(
                        url, dataset_name
                    )
                except requests.exceptions.ConnectionError:
                    assert False, "Link {} for {} is unreachable".format(
                        url, dataset_name
                    )
                except:
                    assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])
        else:
            try:
                dataset.download()
            except:
                assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])


# This is magically skipped by the the remote fixture `skip_local` in conftest.py
# when tests are run with the --local flag
def test_validate(skip_local):
    for dataset_name in DATASETS:
        dataset = soundata.initialize(
            dataset_name, os.path.join(TEST_DATA_HOME, dataset_name), version="test"
        )

        try:
            dataset.validate()
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

        try:
            dataset.validate(verbose=False)
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])


def test_load_and_clipids():
    for dataset_name in DATASETS:
        dataset = soundata.initialize(
            dataset_name, os.path.join(TEST_DATA_HOME, dataset_name), version="test"
        )

        try:
            clip_ids = dataset.clip_ids
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])
        assert type(clip_ids) is list, "{}.clip_ids() should return a list".format(
            dataset_name
        )
        clipid_len = len(clip_ids)
        # if the dataset has clips, test the loaders
        if dataset._clip_class is not None:
            try:
                choice_clip = dataset.choice_clip()
            except:
                assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])
            assert isinstance(
                choice_clip, core.Clip
            ), "{}.choice_clip must return an instance of type core.Clip".format(
                dataset_name
            )

            try:
                dataset_data = dataset.load_clips()
            except:
                assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

            assert isinstance(
                dataset_data, dict
            ), "{}.load should return a dictionary".format(dataset_name)
            assert (
                len(dataset_data.keys()) == clipid_len
            ), "the dictionary returned {}.load() does not have the same number of elements as {}.clip_ids()".format(
                dataset_name, dataset_name
            )


def test_clip():
    for dataset_name in DATASETS:
        dataset = soundata.initialize(
            dataset_name, os.path.join(TEST_DATA_HOME, dataset_name), version="test"
        )

        # if the dataset doesn't have a clip object, make sure it raises a value error
        # and move on to the next dataset
        if dataset._clip_class is None:
            with pytest.raises(NotImplementedError):
                dataset.clip("~fakeclipid~?!")
            continue

        if dataset_name in CUSTOM_TEST_CLIPS:
            clipid = CUSTOM_TEST_CLIPS[dataset_name]
        else:
            clipid = dataset.clip_ids[0]

        # test data home specified
        try:
            clip_test = dataset.clip(clipid)
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

        assert isinstance(
            clip_test, core.Clip
        ), "{}.clip must be an instance of type core.Clip".format(dataset_name)

        assert hasattr(
            clip_test, "to_jams"
        ), "{}.clip must have a to_jams method".format(dataset_name)

        # test calling all attributes, properties and cached properties
        clip_data = get_attributes_and_properties(clip_test)

        for attr in clip_data["attributes"]:
            ret = getattr(clip_test, attr)

        for prop in clip_data["properties"]:
            ret = getattr(clip_test, prop)

        for cprop in clip_data["cached_properties"]:
            ret = getattr(clip_test, cprop)

        # Validate JSON schema
        try:
            jam = clip_test.to_jams()
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

        assert jam.validate(), "Jams validation failed for {}.clip({})".format(
            dataset_name, clipid
        )

        # will fail if something goes wrong with __repr__
        try:
            text_trap = io.StringIO()
            sys.stdout = text_trap
            print(clip_test)
            sys.stdout = sys.__stdout__
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

        with pytest.raises(ValueError):
            dataset.clip("~fakeclipid~?!")


# This tests the case where there is no data in data_home.
# It makes sure that the clip can be initialized and the
# attributes accessed, but that anything requiring data
# files errors (all properties and cached properties).
def test_clip_placeholder_case():
    data_home_dir = "not/a/real/path"

    for dataset_name in DATASETS:
        data_home = os.path.join(data_home_dir, dataset_name)

        dataset = soundata.initialize(dataset_name, data_home, version="test")

        if not dataset._clip_class:
            continue

        if dataset_name in CUSTOM_TEST_CLIPS:
            clipid = CUSTOM_TEST_CLIPS[dataset_name]
        else:
            clipid = dataset.clip_ids[0]

        try:
            clip_test = dataset.clip(clipid)
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

        clip_data = get_attributes_and_properties(clip_test)

        for attr in clip_data["attributes"]:
            ret = getattr(clip_test, attr)

        for prop in clip_data["properties"]:
            with pytest.raises(Exception):
                ret = getattr(clip_test, prop)

        for cprop in clip_data["cached_properties"]:
            with pytest.raises(Exception):
                ret = getattr(clip_test, cprop)


# for load_* functions which require more than one argument
# module_name : {function_name: {parameter2: value, parameter3: value}}
EXCEPTIONS = {}
SKIP = {}


def test_load_methods():
    for dataset_name in DATASETS:
        dataset = soundata.initialize(
            dataset_name, os.path.join(TEST_DATA_HOME, dataset_name), version="test"
        )

        all_methods = dir(dataset)
        load_methods = [
            getattr(dataset, m) for m in all_methods if m.startswith("load_")
        ]
        for load_method in load_methods:
            method_name = load_method.__name__

            # skip default methods
            if method_name == "load_clips" or method_name == "load_clipgroups":
                continue

            # skip overrides, add to the SKIP dictionary to skip a specific load method
            if dataset_name in SKIP and method_name in SKIP[dataset_name]:
                continue

            if load_method.__doc__ is None:
                raise ValueError(
                    "soundata.datasets.{}.Dataset.{} has no documentation".format(
                        dataset_name, method_name
                    )
                )

            params = [
                p
                for p in signature(load_method).parameters.values()
                if p.default == inspect._empty
            ]  # get list of parameters that don't have defaults

            # add to the EXCEPTIONS dictionary above if your load_* function needs
            # more than one argument.
            if dataset_name in EXCEPTIONS and method_name in EXCEPTIONS[dataset_name]:
                extra_params = EXCEPTIONS[dataset_name][method_name]
                with pytest.raises(IOError):
                    load_method("a/fake/filepath", **extra_params)
            else:
                with pytest.raises(IOError):
                    load_method("a/fake/filepath")


CUSTOM_TEST_MCLIPS = {}


def test_clipgroups():
    data_home_dir = "tests/resources/sound_datasets"

    for dataset_name in DATASETS:
        dataset = soundata.initialize(
            dataset_name, os.path.join(TEST_DATA_HOME, dataset_name), version="test"
        )

        # TODO: Create a .load() Index class method that loads the Index content
        with open(dataset.index_path) as f:
            index_data = json.load(f)
        assert index_data["version"] == "sample"

        # TODO this is currently an opt-in test. Make it an opt out test
        # once #265 is addressed
        if dataset_name in CUSTOM_TEST_MCLIPS:
            clipgroup_id = CUSTOM_TEST_MCLIPS[dataset_name]
        else:
            # there are no clipgroups
            continue

        try:
            clipgroup_default = dataset.ClipGroup(clipgroup_id)
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

        # test data home specified
        data_home = os.path.join(data_home_dir, dataset_name)
        dataset_specific = soundata.initialize(dataset_name, data_home=data_home)
        try:
            clipgroup_test = dataset_specific.ClipGroup(
                clipgroup_id, data_home=data_home
            )
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

        assert isinstance(
            clipgroup_test, core.ClipGroup
        ), "{}.ClipGroup must be an instance of type core.ClipGroup".format(
            dataset_name
        )

        assert hasattr(
            clipgroup_test, "to_jams"
        ), "{}.ClipGroup must have a to_jams method".format(dataset_name)

        # Validate JSON schema
        try:
            jam = clipgroup_test.to_jams()
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

        assert jam.validate(), "Jams validation failed for {}.ClipGroup({})".format(
            dataset_name, clipgroup_id
        )
