"""
This test takes a long time, but it makes sure that the datset can be locally downloaded,
validated successfully, and loaded.
"""

import os
import pytest
import tqdm

from tests.test_utils import get_attributes_and_properties
import soundata


@pytest.fixture()
def dataset(test_dataset, dataset_version):
    if test_dataset == "":
        return None
    elif test_dataset not in soundata.DATASETS:
        raise ValueError("{} is not a dataset in soundata".format(test_dataset))
    data_home = os.path.join("tests/resources/sound_datasets_full", test_dataset)
    return soundata.initialize(test_dataset, data_home, version=dataset_version)


# This is magically skipped by the the remote fixture `skip_remote` in conftest.py
# when tests are run without the --local flag
def test_download(skip_remote, dataset, skip_download):
    if dataset is None:
        pytest.skip()

    # download the dataset
    if not skip_download:
        dataset.download()

        print(
            "If this dataset does not have openly downloadable data, "
            + "follow the instructions printed by the download message and "
            + "rerun this test."
        )


def test_validation(skip_remote, dataset):
    if dataset is None:
        pytest.skip()

    # run validation
    missing_files, invalid_checksums = dataset.validate(verbose=True)

    assert missing_files == {
        key: {} for key in dataset._index.keys() if not key == "version"
    }
    assert invalid_checksums == {
        key: {} for key in dataset._index.keys() if not key == "version"
    }


def test_load(skip_remote, dataset):
    if dataset is None:
        pytest.skip()

    # run load
    all_data = dataset.load_clips()

    assert isinstance(all_data, dict)

    clip_ids = dataset.clip_ids
    assert set(clip_ids) == set(all_data.keys())

    # test that all attributes and properties can be called
    for clip_id in tqdm.tqdm(clip_ids):
        clip = all_data[clip_id]
        clip_data = get_attributes_and_properties(clip)

        for attr in clip_data["attributes"]:
            ret = getattr(clip, attr)

        for prop in clip_data["properties"]:
            ret = getattr(clip, prop)

        for cprop in clip_data["cached_properties"]:
            ret = getattr(clip, cprop)

        jam = clip.to_jams()
        assert jam.validate()


def test_index(skip_remote, dataset):
    if dataset is None:
        pytest.skip()

    okeys = ["clips", "clipgroups", "records"]

    if "version" not in dataset._index.keys():
        raise NotImplementedError("The top-level key 'version' is missing in the index")

    if not any(key in dataset._index.keys() for key in okeys):
        raise NotImplementedError(
            "At least one of the optional top-level keys {} should be in the index".format(
                okeys
            )
        )
