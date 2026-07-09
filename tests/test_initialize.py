import pytest

from soundata import core
from soundata import initialize, list_datasets, dcase_challenge


def test_list_datasets():
    dataset_list = list_datasets()
    assert isinstance(dataset_list, list)
    assert "urbansound8k" in dataset_list
    assert "esc50" in dataset_list
    assert "urbansed" in dataset_list


def test_initialize():
    d = initialize("esc50")
    assert isinstance(d, core.Dataset)
    assert d.name == "esc50"

    with pytest.raises(ValueError):
        initialize("asdfasdfasdfa")


def test_dcase_challenge():
    d = dcase_challenge("dcase2023_task6a")
    assert isinstance(d, core.DCASEWrapper)
    assert d.challenge_name == "dcase2023_task6a"

    with pytest.raises(ValueError):
        dcase_challenge("sdkjlkjfkdjl")
