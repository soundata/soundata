import itertools
import json
import os
import types

import soundata
from soundata import validate


import pytest

DEFAULT_DATA_HOME = os.path.join(os.getenv("HOME", "/tmp"), "sound_datasets")


def run_clip_tests(clip, expected_attributes, expected_property_types):
    clip_attr = get_attributes_and_properties(clip)

    # test clip attributes
    for attr in clip_attr["attributes"]:
        print("{}: {}".format(attr, getattr(clip, attr)))
        assert expected_attributes[attr] == getattr(clip, attr)

    # test clip property types
    for prop in clip_attr["cached_properties"] + clip_attr["properties"]:
        print("{}: {}".format(prop, type(getattr(clip, prop))))

        is_tested = False

        if prop in expected_property_types:
            assert isinstance(getattr(clip, prop), expected_property_types[prop])
            is_tested = True

        # Previously, this is ignored if the property is already in `expected_property_types`
        # However, this will cause attribute value bugs to silently pass the tests.
        # This is a workaround to prevent this from happening.
        if prop in expected_attributes:
            assert expected_attributes[prop] == getattr(clip, prop)
            is_tested = True

        if not is_tested:
            assert (
                False
            ), "{} not in expected_property_types or expected_attributes".format(prop)


def run_clipgroup_tests(clipgroup):
    clips = getattr(clipgroup, "clips")
    clip_ids = getattr(clipgroup, "clip_ids")
    assert list(clips.keys()) == clip_ids
    for k, clip in clips.items():
        assert getattr(clip, "clip_id") in clip_ids


def get_attributes_and_properties(class_instance):
    attributes = []
    properties = []
    cached_properties = []
    functions = []
    for val in dir(class_instance.__class__):
        if val.startswith("_"):
            continue

        attr = getattr(class_instance.__class__, val)
        if isinstance(attr, soundata.core.cached_property):
            cached_properties.append(val)
        elif isinstance(attr, property):
            properties.append(val)
        elif isinstance(attr, types.FunctionType):
            functions.append(val)
        else:
            raise ValueError("Unknown type {}".format(attr))

    non_attributes = list(
        itertools.chain.from_iterable([properties, cached_properties, functions])
    )
    for val in dir(class_instance):
        if val.startswith("_"):
            continue
        if val not in non_attributes:
            attributes.append(val)
    return {
        "attributes": attributes,
        "properties": properties,
        "cached_properties": cached_properties,
        "functions": functions,
    }


@pytest.fixture
def mock_validated(mocker):
    return mocker.patch.object(validate, "check_validated")


@pytest.fixture
def mock_validator(mocker):
    return mocker.patch.object(validate, "validator")


@pytest.fixture
def mock_validate_index(mocker):
    return mocker.patch.object(validate, "validate_index")


def test_md5(mocker):
    audio_file = b"audio1234"

    expected_checksum = "6dc00d1bac757abe4ea83308dde68aab"

    mocker.patch("builtins.open", new=mocker.mock_open(read_data=audio_file))

    md5_checksum = validate.md5("test_file_path")
    assert expected_checksum == md5_checksum


@pytest.mark.parametrize(
    "test_index,expected_missing,expected_inv_checksum",
    [
        ("test_index_valid.json", {"clips": {}}, {"clips": {}}),
        (
            "test_index_missing_file.json",
            {"clips": {"test_missing": ["tests/resources/test_missing.wav"]}},
            {"clips": {}},
        ),
        (
            "test_index_invalid_checksum.json",
            {"clips": {}},
            {"clips": {"test": ["tests/resources/test.wav"]}},
        ),
    ],
)
def test_validate_index(test_index, expected_missing, expected_inv_checksum):
    index_path = os.path.join("tests/indexes", test_index)
    with open(index_path) as index_file:
        test_index = json.load(index_file)

    missing_files, invalid_checksums = validate.validate_index(
        test_index, "tests/resources/"
    )

    assert expected_missing == missing_files
    assert expected_inv_checksum == invalid_checksums


@pytest.mark.parametrize(
    "missing_files,invalid_checksums",
    [
        ({"clips": {"test": ["tests/resources/test.wav"]}}, {"clips": {}}),
        ({"clips": {}}, {"clips": {}}),
    ],
)
def test_validator(mocker, mock_validate_index, missing_files, invalid_checksums):
    mock_validate_index.return_value = missing_files, invalid_checksums

    m, c = validate.validator("foo", "bar", False)
    assert m == missing_files
    assert c == invalid_checksums
    mock_validate_index.assert_called_once_with("foo", "bar", False)
